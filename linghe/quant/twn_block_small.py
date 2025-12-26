import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # BLOCK_M is fixed to 128 by the problem definition (alpha per 128 rows)
        # We tune BLOCK_N (column chunk) and num_warps
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def twn_128_row_accum_kernel(
    W_ptr, Q_ptr, Alpha_ptr,
    M, N,
    stride_wm, stride_wn,
    stride_qm, stride_qn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # 1. Map PID to a block of 128 rows
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    
    # 2. Prepare Row-wise Accumulators (Register file)
    # We maintain these per-row until the end of the loop to avoid excessive cross-lane reduction
    row_sum_abs = tl.zeros([BLOCK_M], dtype=tl.float32)

    # -----------------------------------------------------------------------
    # PASS 1: Calculate Sum(Abs) per row
    # -----------------------------------------------------------------------
    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        mask = (offs_n[None, :] < N)
        
        # Load Tile [128, BLOCK_N]
        w_ptr_curr = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        w = tl.load(w_ptr_curr, mask=mask, other=0.0).to(tl.float32)
        
        # Accumulate into row vector (axis=1 reduction is fast within warp)
        row_sum_abs += tl.sum(tl.abs(w), axis=1)

    # 3. Block-wide Reduction
    # Sum across the 128 rows to get the single block stat
    block_total_abs = tl.sum(row_sum_abs)
    
    # Calculate Threshold
    # We cast the denominator to float32 on device to avoid host-scalar errors
    count = (BLOCK_M * N).to(tl.float32)
    threshold = (block_total_abs / count) * 0.7

    # -----------------------------------------------------------------------
    # PASS 2: Quantize using the calculated Threshold
    # -----------------------------------------------------------------------
    # Again, accumulate stats per row first
    row_masked_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    row_masked_cnt = tl.zeros([BLOCK_M], dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        offs_n = n + tl.arange(0, BLOCK_N)
        mask = (offs_n[None, :] < N)
        
        # Re-Load Tile
        w_ptr_curr = W_ptr + (offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        q_ptr_curr = Q_ptr + (offs_m[:, None] * stride_qm + offs_n[None, :] * stride_qn)
        
        w = tl.load(w_ptr_curr, mask=mask, other=0.0).to(tl.float32)
        w_abs = tl.abs(w)
        
        # Apply Threshold
        keep_mask = w_abs > threshold
        keep_f = keep_mask.to(tl.float32)
        
        # Update row accumulators
        row_masked_sum += tl.sum(w_abs * keep_f, axis=1)
        row_masked_cnt += tl.sum(keep_f, axis=1)
        
        # Quantize and Store
        sign = tl.where(w > 0, 1.0, -1.0)
        q_val = (sign * keep_f).to(tl.bfloat16)
        tl.store(q_ptr_curr, q_val, mask=mask)

    # 4. Final Alpha Calculation
    block_masked_sum = tl.sum(row_masked_sum)
    block_masked_cnt = tl.sum(row_masked_cnt)
    
    alpha = block_masked_sum / tl.maximum(block_masked_cnt, 1.0)
    
    # Store Alpha (one per block)
    tl.store(Alpha_ptr + pid, alpha)


def twn_rowblock128_small_m(W: torch.Tensor):
    """
    Optimized fused kernel with row-wise register accumulation.
    """
    M, N = W.shape
    assert M % 128 == 0, "M must be divisible by 128"
    
    W = W.contiguous()
    Q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M // 128, 1), device=W.device, dtype=torch.float32)

    grid = (M // 128, )
    
    # Fixed BLOCK_M=128 as per logic requirements
    twn_128_row_accum_kernel[grid](
        W, Q, alpha,
        M, N,
        W.stride(0), W.stride(1),
        Q.stride(0), Q.stride(1),
        BLOCK_M=128
    )
    
    return Q, alpha.repeat(1, N // 128)

# --- Minimal Test ---
if __name__ == "__main__":
    M, N = 256, 2048
    W = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    
    # Warmup
    Q, A = twn_rowblock128_optimized(W)
    print(Q.shape, A.shape)
    print(A)
    print(Q)
    
    # Speed Test
    ms = triton.testing.do_bench(lambda: twn_rowblock128_optimized(W))
    gbps = (M * N * 3) * 1e-9 / (ms * 1e-3) # Approx: Read 2 bytes, Write 1 byte
    print(f"Time: {ms:.3f}ms | Bandwidth: {gbps:.2f} GB/s")