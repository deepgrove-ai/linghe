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

# this is wrong for some reason
@triton.jit
def hadamard_product(
    x_ptr,
    weight_ptr,
    M,
    T,
    n: tl.constexpr,
    W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # adapted from weighted silu

    row_offs = pid * W * T * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]

    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(weight_ptr + row_offs + col_offs, mask=mask).to(tl.float32)
        x = x1 * x2
        tl.store(x_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

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
    alpha = alpha.view(M, 1)
    # did some numerical testing and genuinely just 1 works as wella s any principled scaled
    s = alpha.abs().max() # / 447
    alpha = (alpha / s).to(torch.float8_e4m3fn)
    alpha = alpha.repeat(1, N)
    #print(alpha)
    #print(Q)
    hadamard_product[grid](
        Q,          # x_ptr
        alpha,         # weight_ptr
        M,              # M (total number of rows)
        8,              # T
        N,            # constexpr n
        N // 2,            # constexpr W
    )
    # Q2 = (Q.float() * alpha.float()).to(torch.float8_e4m3fn)
    # print(Q.float() * alpha.float() - Q2.float())

    
    return Q, s

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
    return Q, torch.mean(alpha).expand(M // 128, N // 128)


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