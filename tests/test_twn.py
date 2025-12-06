from linghe.quant.twn import twn_quant_tensor_fp8, twn_quant_tensor_fp8_triton
from linghe.quant.twn import twn_quant_row_fp8, twn_quant_row_fp8_triton
from linghe.tools.benchmark import benchmark_func
import torch

def test_twn_quant_fp8(M=2048, N=768, bench=False):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096

    dtype = torch.bfloat16
    device = "cuda:0"

    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)

    x_q, alpha = twn_quant_tensor_fp8_triton(x)
    x_q_ref, alpha_ref = twn_quant_tensor_fp8(x)
    print(x_q.dtype, x_q_ref.dtype)
    x_q = x_q.to(torch.float32)
    x_q_ref = x_q_ref.to(torch.float32)
    print(x_q-x_q_ref)
    print(alpha-alpha_ref)
    if bench:
        benchmark_func(twn_quant_tensor_fp8, x, n_repeat=n_repeat, ref_bytes=M * N * 2)
        benchmark_func(twn_quant_tensor_fp8_triton, x, n_repeat=n_repeat, ref_bytes=M * N * 2)
        benchmark_func(twn_quant_row_fp8, x, n_repeat=n_repeat, ref_bytes=M * N * 2)
        benchmark_func(twn_quant_row_fp8_triton, x, n_repeat=n_repeat, ref_bytes=M * N * 2)
    
if __name__ == "__main__":
    test_twn_quant_fp8(bench=True)