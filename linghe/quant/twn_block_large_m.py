import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # BLOCK_M: Number of rows to process in parallel registers. 
        # Low BLOCK_M (4-8) is required because we load the FULL N width.
        triton.Config({'BLOCK_M': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 2}, num_warps=4, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def twn_128_full_row_kernel(
    W_ptr, Q_ptr, Alpha_ptr,
    M, N,
    stride_wm, stride_wn,
    stride_qm, stride_qn,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr  # Must be >= N
):
    # This Program ID handles one group of 128 rows
    pid = tl.program_id(0)
    group_start_row = pid * 128

    # Accumulators for the final alpha (averaged over 128 rows)
    group_masked_sum = 0.0
    group_masked_cnt = 0.0

    # Column offsets: 0 to BLOCK_N (Fixed, full width)
    # We assume BLOCK_N >= N. Mask handles non-power-of-2 N.
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Iterate over the 128 rows in chunks of BLOCK_M
    # e.g., if BLOCK_M=4, we loop 32 times.
    for i in range(0, 128, BLOCK_M):
        # Current row indices
        current_rows = group_start_row + i + tl.arange(0, BLOCK_M)
        mask_m = current_rows < M

        # Pointers for this [BLOCK_M, BLOCK_N] tile
        w_ptrs = W_ptr + (current_rows[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        q_ptrs = Q_ptr + (current_rows[:, None] * stride_qm + offs_n[None, :] * stride_qn)

        # 1. LOAD FULL ROWS (Single Load)
        # We rely on cache/registers here. 
        w = tl.load(w_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)
        
        # 2. COMPUTE THRESHOLD (Per Row)
        # Reduction over N is fast because data is in registers
        row_sum_abs = tl.sum(tl.abs(w), axis=1)
        # Use N for division, not BLOCK_N, to be exact
        threshold = (row_sum_abs / N) * 0.7

        # 3. QUANTIZE (Per Row)
        w_abs = tl.abs(w)
        # Broadcast threshold [BLOCK_M] -> [BLOCK_M, 1] for comparison
        keep_mask = w_abs > threshold[:, None]
        keep_f = keep_mask.to(tl.float32)

        sign = tl.where(w > 0, 1.0, -1.0)
        q_val = (sign * keep_f).to(tl.bfloat16)

        # 4. STORE Q
        tl.store(q_ptrs, q_val, mask=(mask_m[:, None] & mask_n[None, :]))

        # 5. ACCUMULATE STATS FOR ALPHA
        # We sum the kept values for this chunk of rows
        group_masked_sum += tl.sum(tl.sum(w_abs * keep_f, axis=1))
        group_masked_cnt += tl.sum(tl.sum(keep_f, axis=1))

    # 6. FINALIZE ALPHA (One per 128 rows)
    # Note: sum is over all elements in the 128xN block
    alpha = group_masked_sum / tl.maximum(group_masked_cnt, 1.0)
    tl.store(Alpha_ptr + pid, alpha)


def twn_rowblock128_large(W: torch.Tensor):
    M, N = W.shape
    assert M % 128 == 0
    
    # Calculate next power of 2 for BLOCK_N
    # Triton requires constant block sizes. 
    # For a general function, you might pad or branch based on N size.
    # Here we pick a static size large enough for standard LLM layers.
    BLOCK_N = triton.next_power_of_2(N)
    
    W = W.contiguous()
    Q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M // 128, 1), device=W.device, dtype=torch.float32)

    grid = (M // 128, )
    
    twn_128_full_row_kernel[grid](
        W, Q, alpha,
        M, N,
        W.stride(0), W.stride(1),
        Q.stride(0), Q.stride(1),
        BLOCK_N=BLOCK_N
    )
    return Q, alpha.repeat(1, N //128)

if __name__ == "__main__":
    # Test with standard Llama size
    M, N = 2048, 2048 
    W = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    Q, A = twn_rowblock128_fast(W)
    
    print(f"Strategy B Output: {Q.shape}")
    print(A)
    print(Q)
    
    # Quick Benchmark
    ms = triton.testing.do_bench(lambda: twn_rowblock128_fast(W))
    gbps = (M * N * 3) * 1e-9 / (ms * 1e-3)
    print(f"Strategy B Speed: {gbps:.2f} GB/s, {ms} ms")