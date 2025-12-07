from typing import Optional
import torch
import triton
import triton.language as tl

@torch.compile()
def twn_quant_row_fp8(W):
    # for reference
    dim = 1 if W.ndim == 2 else -1
    # threshold calc
    absW = W.abs()
    th = absW.mean(dim, keepdim=True) * 0.7
    # these should be fusable (loop over with to mask + mult/sum with absW)
    mask = absW > th
    mask_f = mask.to(W.dtype)
    alpha = (absW * mask_f).sum(dim, keepdim=True) / mask_f.sum(dim, keepdim=True).clamp(min=1)
    # now we need to cast the ternary matrixto fp8 and return
    mask_f = (W.sign() * mask_f).to(torch.float8_e4m3fn)
    alpha = alpha.to(torch.float32)
    return mask_f, alpha

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_SIZE': 64},
            num_warps=2,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_SIZE': 128},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {'BLOCK_SIZE': 256},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_SIZE': 512},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=['N'],
)

@triton.jit
def twn_quant_row_fp8_kernel(
    w_ptr,        # *const W
    q_ptr,        # *mut Q (ternary in fp8)
    alpha_ptr,    # *mut alpha (fp32, per-row)
    M, N,         # matrix dims
    stride_wm, stride_wn,
    stride_qm, stride_qn,
    stride_am,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # row id
    if pid >= M:
        return

    # threshold
    n_block = tl.cdiv(N, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_abs = 0.0
    count = 0.0

    for j in range(0, n_block):
        cols = j * BLOCK_SIZE + offsets
        mask_col = cols < N

        offs = pid * stride_wm + cols * stride_wn
        w = tl.load(w_ptr + offs, mask=mask_col, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        sum_abs += tl.sum(absw, axis=0)
        count += tl.sum(mask_col.to(tl.float32), axis=0)

    mean_abs = sum_abs / tl.maximum(count, 1.0)
    th = mean_abs * 0.7

    masked_sum = 0.0
    masked_count = 0.0

    for j in range(0, n_block):
        cols = j * BLOCK_SIZE + offsets
        mask_col = cols < N

        offs = pid * stride_wm + cols * stride_wn
        w = tl.load(w_ptr + offs, mask=mask_col, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        # mask = absW > th
        mask_sel = absw > th
        mask_f = mask_sel.to(tl.float32)

        masked_sum += tl.sum(absw * mask_f, axis=0)
        masked_count += tl.sum(mask_f, axis=0)

        sign = tl.where(w >= 0, 1.0, -1.0)
        ternary = sign * mask_f  # {-1, 0, +1}
        q_block = ternary.to(tl.float8e4m3fn if hasattr(tl, "float8e4m3fn") else tl.bfloat16)

        tl.store(q_ptr + offs, q_block, mask=mask_col)

    alpha = masked_sum / tl.maximum(masked_count, 1.0)
    tl.store(alpha_ptr + pid * stride_am, alpha)


def twn_quant_row_fp8_triton(W: torch.Tensor):
    """
    Row-wise TWN quantization to fp8 (e4m3fn) using the Triton kernel above.

    W: [M, N], any float type (fp16/bf16/fp32)
    Returns:
        Q:     [M, N], dtype=torch.float8_e4m3fn, ternary {-1,0,+1} in fp8
        alpha: [M, 1], dtype=torch.float32
    """
    assert W.ndim == 2, "This wrapper assumes a 2D weight matrix."
    M, N = W.shape
    W = W.contiguous()

    Q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M,), device=W.device, dtype=torch.float32)


    grid = (M,)
    twn_quant_row_fp8_kernel[grid](
        W,
        Q,
        alpha,
        M,
        N,
        W.stride(0),
        W.stride(1),
        Q.stride(0),
        Q.stride(1),
        alpha.stride(0),
        # BLOCK_SIZE=block_size,
    )

    return Q, alpha.view(M, 1)

@triton.jit
def block_mean_128_fused_kernel(
    x_ptr,           # *const float (length M)
    out_ptr,         # *mut float, flattened (M_blk * N_blk)
    M,               # total length of x (number of rows)
    N_blk,           # = N // 128 (number of column blocks)
    BLOCK_M: tl.constexpr,   # 128 (rows per block)
    BLOCK_NB: tl.constexpr,  # max N_blk per program (e.g. 128)
):
    # Each program handles one row-block of 128 elements
    row_block_id = tl.program_id(axis=0)  # 0 .. M_blk-1, where M_blk = M // 128

    # ---- compute mean over 128 rows for this block ----
    row_offs = row_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_rows = row_offs < M

    x = tl.load(x_ptr + row_offs, mask=mask_rows, other=0.0).to(tl.float32)
    block_sum = tl.sum(x, axis=0)
    valid_count = tl.sum(mask_rows, axis=0).to(tl.float32)
    mean = block_sum / valid_count  # scalar

    # ---- broadcast this mean across N_blk columns ----
    cols = tl.arange(0, BLOCK_NB)
    mask_cols = cols < N_blk

    # out is logically [M_blk, N_blk], row-major
    out_row_base = row_block_id * N_blk
    out_offs = out_row_base + cols

    tl.store(out_ptr + out_offs, mean, mask=mask_cols)

@triton.jit
def block_mean_128_kernel(
    x_ptr,          # *const float / bf16 / whatever
    out_ptr,        # *mut float (one mean per block)
    K,              # total length of x
    BLOCK_SIZE: tl.constexpr,  # should be 128 for your case
):
    pid = tl.program_id(axis=0)  # block index
    # Offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < K

    # Load block
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Sum and count valid elements (for tail block)
    block_sum = tl.sum(x, axis=0)
    valid_count = tl.sum(mask, axis=0).to(tl.float32)

    mean = block_sum / valid_count

    # One output per block
    tl.store(out_ptr + pid, mean)

def twn_quant_row_tensor_block_fp8_triton(W: torch.Tensor, block_size = 128):
    """
    Row-wise TWN quantization to fp8 (e4m3fn) using the Triton kernel above.

    W: [M, N], any float type (fp16/bf16/fp32)
    Returns:
        Q:     [M, N], dtype=torch.float8_e4m3fn, ternary {-1,0,+1} in fp8
        alpha: [M, 1], dtype=torch.float32
    """
    assert W.ndim == 2, "This wrapper assumes a 2D weight matrix."
    M, N = W.shape
    W = W.contiguous()

    Q = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M,), device=W.device, dtype=torch.float32)
    real_alpha = torch.empty((M // 128, 1), device=W.device, dtype=torch.float32)


    grid = (M,)
    twn_quant_row_fp8_kernel[grid](
        W,
        Q,
        alpha,
        M,
        N,
        W.stride(0),
        W.stride(1),
        Q.stride(0),
        Q.stride(1),
        alpha.stride(0),
        # BLOCK_SIZE=block_size,
    )
    # idea here is to approx tensor scaling factor as mean (per tensor ternary) and then expand to block size of deepseek fp8
    
    block_mean_128_kernel[grid](
        alpha,
        real_alpha,
        M,
        128,
    )
    
    return Q, real_alpha.repeat(1, N // 128)


if __name__ == "__main__":
    W = torch.randn(8, 8).to(torch.bfloat16).to("cuda")
    Q, alpha = twn_quant_row_fp8_triton(W)
    Q_ref, alpha_ref = twn_quant_row_fp8(W)
    print(Q.dtype, Q_ref.dtype)
    Q = Q.to(torch.float32) * alpha
    Q_ref = Q_ref.to(torch.float32) * alpha_ref.float()
    print(Q-Q_ref)
    print(alpha)
    print(alpha_ref)

