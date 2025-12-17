from posix import wait
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

    return Q, alpha.view(M, 1)

# new kernel for the deepgemm stuff
_TUNE_CONFIGS = [
    triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_K": 512}, num_warps=8, num_stages=3),
]
_MIN_BK = 128  # smallest BLOCK_K among configs (keep in sync)

@triton.autotune(configs=_TUNE_CONFIGS, key=["K"])
@triton.jit
def twn_quant_rowblock128_kernel(
    w_ptr, q_ptr, alpha_ptr,
    N,
    stride_wn, stride_wk,
    stride_qn, stride_qk,
    stride_ab,
    K: tl.constexpr,

    MAX_K_BLOCKS: tl.constexpr,          # <<< passed from host, Python int
    BLOCK_R: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 256,         # autotuned
    OUT_Q: tl.constexpr = True,
    Q_DTYPE: tl.constexpr = tl.bfloat16,
):
    rb = tl.program_id(0)
    row_offs = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    rows_mask = row_offs < N

    k_offs = tl.arange(0, BLOCK_K)

    # pass 1
    sum_abs = 0.0
    count = 0.0
    for kb in tl.static_range(0, MAX_K_BLOCKS):
        cols = kb * BLOCK_K + k_offs
        cols_mask = cols < K
        mask = rows_mask[:, None] & cols_mask[None, :]

        offs = row_offs[:, None] * stride_wn + cols[None, :] * stride_wk
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        sum_abs += tl.sum(tl.sum(absw, axis=0), axis=0)
        count   += tl.sum(tl.sum(mask.to(tl.float32), axis=0), axis=0)

    mean_abs = sum_abs / tl.maximum(count, 1.0)
    th = mean_abs * 0.7

    # pass 2
    masked_sum = 0.0
    masked_cnt = 0.0
    for kb in tl.static_range(0, MAX_K_BLOCKS):
        cols = kb * BLOCK_K + k_offs
        cols_mask = cols < K
        mask = rows_mask[:, None] & cols_mask[None, :]

        offs = row_offs[:, None] * stride_wn + cols[None, :] * stride_wk
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        sel_f = (absw > th).to(tl.float32)
        masked_sum += tl.sum(tl.sum(absw * sel_f, axis=0), axis=0)
        masked_cnt += tl.sum(tl.sum(sel_f, axis=0), axis=0)

        if OUT_Q:
            sign = tl.where(w >= 0.0, 1.0, -1.0)
            q = (sign * sel_f).to(Q_DTYPE)
            tl.store(q_ptr + (row_offs[:, None] * stride_qn + cols[None, :] * stride_qk),
                     q, mask=mask)

    alpha = masked_sum / tl.maximum(masked_cnt, 1.0)
    tl.store(alpha_ptr + rb * stride_ab, alpha)


@triton.autotune(configs=_TUNE_CONFIGS, key=["K"])
@triton.jit
def twn_quant_rowblock128_3d_kernel(
    w_ptr, q_ptr, alpha_ptr,
    G, N,
    stride_wg, stride_wn, stride_wk,
    stride_qg, stride_qn, stride_qk,
    stride_ag, stride_ab,
    K: tl.constexpr,

    MAX_K_BLOCKS: tl.constexpr,
    BLOCK_R: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 256,   # autotuned
    OUT_Q: tl.constexpr = True,
    Q_DTYPE: tl.constexpr = tl.bfloat16,
):
    rb = tl.program_id(0)   # row-block id along N
    g  = tl.program_id(1)   # group id
    if g >= G:
        return

    # rows in this 128-row block
    row_offs = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    rows_mask = row_offs < N

    # columns tile
    k_offs = tl.arange(0, BLOCK_K)

    w_base = w_ptr + g * stride_wg
    q_base = q_ptr + g * stride_qg

    # ---- pass 1: mean(|w|) over [BLOCK_R, K] for this group/block ----
    sum_abs = 0.0
    count = 0.0
    for kb in tl.static_range(0, MAX_K_BLOCKS):
        cols = kb * BLOCK_K + k_offs
        cols_mask = cols < K

        mask = rows_mask[:, None] & cols_mask[None, :]
        offs = row_offs[:, None] * stride_wn + cols[None, :] * stride_wk

        w = tl.load(w_base + offs, mask=mask, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        sum_abs += tl.sum(tl.sum(absw, axis=0), axis=0)
        count   += tl.sum(tl.sum(mask.to(tl.float32), axis=0), axis=0)

    mean_abs = sum_abs / tl.maximum(count, 1.0)
    th = mean_abs * 0.7

    # ---- pass 2: masked alpha + optional Q write ----
    masked_sum = 0.0
    masked_cnt = 0.0
    for kb in tl.static_range(0, MAX_K_BLOCKS):
        cols = kb * BLOCK_K + k_offs
        cols_mask = cols < K

        mask = rows_mask[:, None] & cols_mask[None, :]
        offs = row_offs[:, None] * stride_wn + cols[None, :] * stride_wk

        w = tl.load(w_base + offs, mask=mask, other=0.0).to(tl.float32)
        absw = tl.abs(w)

        sel_f = (absw > th).to(tl.float32)

        masked_sum += tl.sum(tl.sum(absw * sel_f, axis=0), axis=0)
        masked_cnt += tl.sum(tl.sum(sel_f, axis=0), axis=0)

        if OUT_Q:
            sign = tl.where(w >= 0.0, 1.0, -1.0)
            q = (sign * sel_f).to(Q_DTYPE)
            tl.store(q_base + (row_offs[:, None] * stride_qn + cols[None, :] * stride_qk),
                     q, mask=mask)

    alpha = masked_sum / tl.maximum(masked_cnt, 1.0)

    # alpha[g, rb]
    tl.store(alpha_ptr + g * stride_ag + rb * stride_ab, alpha)


def launch_rowblock128_grouped(w, q, alpha, *, out_q=True):
    """
    w:     [G, N, K]
    q:     [G, N, K] (ignored if out_q=False)
    alpha: [G, ceil_div(N,128)] or [G, ceil_div(N,128), 1]
    """
    M, N, K = w.shape
    w = w.contiguous()

    q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M, N // 128), device=w.device, dtype=torch.float32)
    N, K = w.shape
    grid = (triton.cdiv(N, 128),)
    max_k_blocks = triton.cdiv(K, _MIN_BK)

    twn_quant_rowblock128_3d_kernel[grid](
        w, q, alpha,
        G, N,
        w.stride(0), w.stride(1), w.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        alpha.stride(0), alpha.stride(1),
        K=K,
        MAX_K_BLOCKS=max_k_blocks,
        OUT_Q=out_q,
        Q_DTYPE=tl.bfloat16,
    )

def twn_rowblock128(w, *, out_q=True):
    M, N = w.shape
    w = w.contiguous()

    q = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    alpha = torch.empty((M // 128,), device=w.device, dtype=torch.float32)
    N, K = w.shape
    grid = (triton.cdiv(N, 128), M)
    max_k_blocks = triton.cdiv(K, _MIN_BK)
    twn_quant_rowblock128_kernel[grid](
        w, q, alpha,
        N,
        w.stride(0), w.stride(1),
        q.stride(0), q.stride(1),
        alpha.stride(0),
        K=K,
        MAX_K_BLOCKS=max_k_blocks,  
        OUT_Q=out_q,
        Q_DTYPE=tl.bfloat16,
    )
    return q, alpha

if __name__ == "__main__":
    W = torch.randn(256, 128).to(torch.bfloat16).to("cuda")
    Q, alpha = twn_rowblock128(W)
    Q_ref, alpha_ref = twn_quant_row_fp8(W)
    print(Q.dtype, Q_ref.dtype)
    #Q = Q.to(torch.float32) * alpha
    #Q_ref = Q_ref.to(torch.float32) * alpha_ref.float()
    #print(Q-Q_ref)
    print(alpha)
    print(Q)

