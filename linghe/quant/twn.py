from typing import Optional
import torch
import triton
import triton.language as tl

@torch.compile()
def twn_quant_tensor_fp8(W):
    # for reference
    # threshold calc
    absW = W.abs()
    th = absW.mean() * 0.7
    # these should be fusable (loop over with to mask + mult/sum with absW)
    mask = absW > th
    mask_f = mask.to(W.dtype)
    alpha = (absW * mask_f).sum() / mask_f.sum().clamp(min=1)
    # now we need to cast the ternary matrixto fp8 and return
    ternary = (W.sign() * mask_f).to(torch.float8_e4m3fn)
    alpha = alpha.to(torch.float32)
    return ternary, alpha

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

    # --- First pass: compute mean(|W_row|) to get threshold ---
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

    # --- Second pass: apply mask, accumulate masked sum, and write ternary FP8 ---
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

        # sign * mask_f in fp8
        sign = tl.where(w >= 0, 1.0, -1.0)
        ternary = sign * mask_f  # {-1, 0, +1}
        q_block = ternary.to(tl.bfloat16)

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

    # match your reference’s broadcasting behavior: alpha shape [M, 1]
    return Q, alpha.squeeze() # alpha.view(M, 1)


@triton.jit
def twn_quant_tensor_fp8_kernel(
    x_ptr,             # *const W
    q_ptr,             # *mut ternary (fp8)
    masked_sum_ptr,    # *mut float32 scalar
    masked_count_ptr,  # *mut float32 scalar
    th,                # float32 threshold
    N,                 # total elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask_offs = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask_offs, other=0.0).to(tl.float32)
    absx = tl.abs(x)

    mask = absx > th
    mask_f = mask.to(tl.float32)

    sign = tl.where(x >= 0, 1.0, -1.0)
    ternary = sign * mask_f  # {-1, 0, +1}

    q_block = ternary.to(tl.float16)
    tl.store(q_ptr + offsets, q_block, mask=mask_offs)

    local_masked_sum = tl.sum(absx * mask_f, axis=0)
    local_masked_count = tl.sum(mask_f, axis=0)

    # OLD API: atomic_add(ptr, mask, val)
    tl.atomic_add(masked_sum_ptr, local_masked_sum)
    tl.atomic_add(masked_count_ptr, local_masked_count)

@torch.compile()
def absmean(x):
    return x.abs().mean()

def twn_quant_tensor_fp8_triton(W: torch.Tensor, block_size: int = 128):
    """
    Tensorwise TWN quantization to fp8 (e4m3fn) using Triton.

    Matches:

        absW = W.abs()
        th = absW.mean() * 0.7
        mask = absW > th
        mask_f = mask.to(W.dtype)
        alpha = (absW * mask_f).sum() / mask_f.sum().clamp(min=1)
        ternary = (W.sign() * mask_f).to(torch.float8_e4m3fn)

    Args:
        W: Tensor of any shape and float dtype (fp16/bf16/fp32).
    Returns:
        ternary: same shape as W, dtype=torch.float8_e4m3fn
        alpha:   scalar torch.float32
    """
    assert W.is_cuda, "Triton kernel requires a CUDA tensor"
    x = W.contiguous()
    N = x.numel()

    # --- 1) Compute absW.mean() to get threshold ---
    # sum_abs = torch.zeros(1, device=x.device, dtype=torch.float32)
    # count = torch.zeros(1, device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(N, block_size),)

    # reduce_abs_sum_kernel[grid](
    #     x,
    #     sum_abs,
    #     count,
    #     N,
    #     BLOCK_SIZE=block_size,
    # )

    mean_abs = x.abs().mean() # 
    # mean_abs = sum_abs / count.clamp_min(1.0)
    th = (mean_abs * 0.7).item()  # scalar float passed as meta-arg

    # --- 2) Quantize to ternary fp8 and compute alpha ---
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    masked_sum = torch.zeros(1, device=x.device, dtype=torch.float32)
    masked_count = torch.zeros(1, device=x.device, dtype=torch.float32)

    twn_quant_tensor_fp8_kernel[grid](
        x,
        q,
        masked_sum,
        masked_count,
        th,
        N,
        BLOCK_SIZE=block_size,
    )

    alpha = (masked_sum / masked_count.clamp_min(1.0)).to(torch.float32)

    # reshape q back to W's shape
    q = q.view_as(W)

    # alpha is scalar in your reference; keep it that way
    return q, alpha.squeeze(0)


if __name__ == "__main__":
    W = torch.randn(8, 8).to(torch.bfloat16).to("cuda")
    Q, alpha = twn_quant_tensor_fp8(W)
    Q_ref, alpha_ref = twn_quant_tensor_fp8(W)
    print(Q.dtype, Q_ref.dtype)
    Q = Q.to(torch.float32)
    Q_ref = Q_ref.to(torch.float32)
    print(Q-Q_ref)
    print(alpha-alpha_ref)

