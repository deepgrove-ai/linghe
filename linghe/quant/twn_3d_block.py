import torch
import triton
import triton.language as tl

# =========================================================================
# 1. 3D MoE KERNEL (Your Code)
# =========================================================================

@triton.autotune(
    configs=[
        # Optimized configs for large N loading
        triton.Config({'BLOCK_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 1}, num_warps=4, num_stages=4), # Fallback for huge N
    ],
    key=['N'],
)
@triton.jit
def twn_moe_3d_kernel(
    W_ptr, Q_ptr, Alpha_ptr,
    M, N,
    stride_we, stride_wm, stride_wn,
    stride_qe, stride_qm, stride_qn,
    stride_ae, stride_am,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    blocks_per_expert = M // 128
    
    cur_expert = pid // blocks_per_expert
    cur_block = pid % blocks_per_expert
    
    start_row = cur_block * 128
    
    off_we = cur_expert * stride_we
    off_qe = cur_expert * stride_qe
    off_ae = cur_expert * stride_ae
    
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    group_masked_sum = 0.0
    group_masked_cnt = 0.0

    for i in range(0, 128, BLOCK_M):
        current_rows = start_row + i + tl.arange(0, BLOCK_M)
        mask_m = current_rows < M
        
        w_ptrs = W_ptr + off_we + (current_rows[:, None] * stride_wm + offs_n[None, :] * stride_wn)
        q_ptrs = Q_ptr + off_qe + (current_rows[:, None] * stride_qm + offs_n[None, :] * stride_qn)

        w = tl.load(w_ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0).to(tl.float32)

        row_sum_abs = tl.sum(tl.abs(w), axis=1)
        threshold = (row_sum_abs / N) * 0.7

        w_abs = tl.abs(w)
        keep_mask = w_abs > threshold[:, None]
        keep_f = keep_mask.to(tl.float32)

        sign = tl.where(w > 0, 1.0, -1.0)
        out_v = (sign * keep_f).to(tl.bfloat16) # Cast to bf16 before store
        
        tl.store(q_ptrs, out_v, mask=(mask_m[:, None] & mask_n[None, :]))

        group_masked_sum += tl.sum(tl.sum(w_abs * keep_f, axis=1))
        group_masked_cnt += tl.sum(tl.sum(keep_f, axis=1))

    alpha = group_masked_sum / tl.maximum(group_masked_cnt, 1.0)
    alpha_ptr_final = Alpha_ptr + off_ae + cur_block * stride_am
    tl.store(alpha_ptr_final, alpha)


def twn_moe_triton(W: torch.Tensor):
    E, M, N = W.shape
    W = W.contiguous()
    Q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((E, M // 128, 1), device=W.device, dtype=torch.float32)

    grid = (E * (M // 128), )
    BLOCK_N = triton.next_power_of_2(N)
    
    twn_moe_3d_kernel[grid](
        W, Q, alpha,
        M, N,
        W.stride(0), W.stride(1), W.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2),
        alpha.stride(0), alpha.stride(1),
        BLOCK_N=BLOCK_N
    )
    return Q, alpha.repeat(1, 1, N//128)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[1024, 2048, 4096, 8192], 
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch (Compiled)', 'Triton (Fused 3D)'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='moe-twn-speed-test',
        args={'E': 8, 'M': 896}, # 8 Experts, 4k Hidden
    )
)
def benchmark(E, M, N, provider):
    W = torch.randn(E, M, N, device='cuda', dtype=torch.bfloat16)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: twn_moe_triton(W), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: twn_moe_triton(W), quantiles=quantiles)
    
    # Bandwidth Calculation
    # Input Read: E*M*N * 2 bytes (bf16)
    # Output Write Q: E*M*N * 1 byte (fp8)
    # Output Write Alpha: E*(M/128) * 4 bytes (fp32) (negligible)
    total_bytes = (E * M * N * 3) 
    gbps = lambda ms: total_bytes * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    print("Running Correctness Check...")
    E, M, N = 16, 768, 2048
    W = torch.randn(E, M, N, device='cuda', dtype=torch.bfloat16)
    
    # Run both
    Q_tri, A_tri = twn_moe_triton(W)
    print(Q_tri.shape, A_tri.shape)
    print(Q_tri[0, : ,:])
    print(A_tri[0, :, :])
    print("\nRunning Speed Test...")
    benchmark.run(show_plots=True, print_data=True)