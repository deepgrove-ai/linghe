import torch
import triton
import triton.language as tl

@triton.jit
def qk_norm_and_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    B,
    stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    # Load frequencies for RoPE
    # freqs covers D (half head dim), applied to both halves
    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            q0 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            q1 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )

        # 1. Apply RMS Norm
        rms = 1 / tl.sqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q0 *= rms[:, None]
        q1 *= rms[:, None]

        # 2. Apply Weight
        q0 *= q_weight_0
        q1 *= q_weight_1

        # 3. Apply Full RoPE
        # Standard RoPE: rotate_half(x) = [-x2, x1]
        # out = x * cos + rotate_half(x) * sin
        # q0_out = q0 * cos - q1 * sin
        # q1_out = q1 * cos + q0 * sin
        q0_out = q0 * cos - q1 * sin
        q1_out = q1 * cos + q0 * sin

        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :],
            q0_out,
        )
        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + D
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :],
            q1_out,
        )

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H

    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            k0 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            k1 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )

        # 1. Norm
        rms = 1 / tl.sqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k0 *= rms[:, None]
        k1 *= rms[:, None]

        # 2. Weight
        k0 *= k_weight_0
        k1 *= k_weight_1

        # 3. Full RoPE
        k0_out = k0 * cos - k1 * sin
        k1_out = k1 * cos + k0 * sin

        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            k0_out,
        )
        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            k1_out,
        )

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h

    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            v1 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            v0 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            v1 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )

        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            v0,
        )
        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            v1,
        )


def triton_qk_norm_and_rope_forward(
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    H=32,
    h=4,
    eps=1e-6,
    interleaved=True,
    transposed=False,
):
    if transposed:
        L, B, Dim = qkv.shape
    else:
        B, L, Dim = qkv.shape
    stride = qkv.stride(1)
    D = Dim // (H + 2 * h)
    dtype = qkv.dtype
    device = qkv.device
    qo = torch.empty((B, L, H, D), dtype=dtype, device=device)
    ko = torch.empty((B, L, h, D), dtype=dtype, device=device)
    vo = torch.empty((B, L, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (L,)
    qk_norm_and_rope_forward_kernel[grid](
        qkv,
        q_norm_weight,
        k_norm_weight,
        freqs,
        qo,
        ko,
        vo,
        B,
        stride,
        eps,
        H,
        h,
        D // 2,
        interleaved,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return qo, ko, vo


@triton.jit
def qk_norm_and_rope_backward_kernel(
    gq_ptr,
    gk_ptr,
    gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    dqkv_ptr,
    dqw_ptr,
    dkw_ptr,
    B,
    stride,
    grad_stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr

    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        # Load gradients of Output (gq)
        gq_0 = tl.load(
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :]
        )
        gq_1 = tl.load(
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + D
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :]
        )

        # Apply Inverse Full RoPE to Gradients
        # Forward: q0_out = q0*c - q1*s, q1_out = q1*c + q0*s
        # Backward:
        # dq0_rot = gq0 * c + gq1 * s
        # dq1_rot = gq1 * c - gq0 * s
        gq_0_rot = gq_0 * cos + gq_1 * sin
        gq_1_rot = gq_1 * cos - gq_0 * sin

        # Load Original Q for Norm Backward
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            q0 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            q1 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )

        # RMS Norm Backward
        rms = tl.sqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        r = (1 / rms)[:, None]

        # Accumulate weight gradients
        dqw_0 += tl.sum(q0 * gq_0_rot * r, 0)
        dqw_1 += tl.sum(q1 * gq_1_rot * r, 0)

        s = tl.sum(q0 * gq_0_rot * q_w0, 1) + tl.sum(q1 * gq_1_rot * q_w1, 1)

        dq_0 = r * gq_0_rot * q_w0 - r * r * r / DD * q0 * s[:, None]
        dq_1 = r * gq_1_rot * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
            )
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
            )
        else:
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
            )
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
            )

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H

    for i in range(B):
        gk_0 = tl.load(
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        )
        gk_1 = tl.load(
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        )

        # Inverse Full RoPE for K gradients
        gk_0_rot = gk_0 * cos + gk_1 * sin
        gk_1_rot = gk_1 * cos - gk_0 * sin

        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            k0 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )
            k1 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            )

        rms = tl.sqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        r = (1 / rms)[:, None]

        dkw_0 += tl.sum(k0 * gk_0_rot * r, 0)
        dkw_1 += tl.sum(k1 * gk_1_rot * r, 0)

        s = tl.sum(k0 * gk_0_rot * k_w0, 1) + tl.sum(k1 * gk_1_rot * k_w1, 1)

        dk_0 = r * gk_0_rot * k_w0 - r * r * r / DD * k0 * s[:, None]
        dk_1 = r * gk_1_rot * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
            )
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
            )
        else:
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
            )
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
            )
    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # Store V gradients (unchanged, just transposed copy)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):
        v0 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        )

        if TRANSPOSED:
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                v0,
            )
        else:
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                v0,
            )

        v1 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        )

        if TRANSPOSED:
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                v1,
            )
        else:
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                v1,
            )


def triton_qk_norm_and_rope_backward(
    gq,
    gk,
    gv,
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    eps=1e-6,
    interleaved=True,
    transposed=False,
):
    B, L, H, D = gq.shape
    stride = qkv.stride(1)
    h = gk.shape[2]
    num_stages = 5
    num_warps = 1

    dtype = gq.dtype
    device = gq.device
    if transposed:
        dqkv = torch.empty((L, B, (H + 2 * h) * D), dtype=dtype, device=device)
    else:
        dqkv = torch.empty((B, L, (H + 2 * h) * D), dtype=dtype, device=device)
    grad_stride = dqkv.stride(1)

    tmp_dqw = torch.empty((L, D), dtype=torch.float32, device=device)
    tmp_dkw = torch.empty((L, D), dtype=torch.float32, device=device)

    grid = (L,)
    qk_norm_and_rope_backward_kernel[grid](
        gq,
        gk,
        gv,
        qkv,
        q_norm_weight,
        k_norm_weight,
        freqs,
        dqkv,
        tmp_dqw,
        tmp_dkw,
        B,
        stride,
        grad_stride,
        eps,
        H,
        h,
        D // 2,
        interleaved,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    dqw = tmp_dqw.sum(0).to(dtype)
    dkw = tmp_dkw.sum(0).to(dtype)
    return dqkv, dqw, dkw

import torch
import triton
import math
import copy

# Import the kernel functions from the previous cell
# Assuming the previous code is saved in a module or pasted above.
# For this script to run standalone, you would include the kernel code here.
# I will assume `triton_qk_norm_and_rope_forward` and `triton_qk_norm_and_rope_backward` are available.

# ==========================================
# PyTorch Reference Implementation (Qwen Style)
# ==========================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotatry Positional Embedding.
    Standard HF implementation: (x * cos) + (rotate_half(x) * sin)
    """
    # Reshape cos/sin to match q/k dimensions: [B, S, 1, D]
    # Assuming cos/sin are [S, D] or [B, S, D], we need broadcasting
    # The kernel takes freqs, so we simulate the sin/cos generation in the test harness
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class ReferenceQKNormRoPE(torch.nn.Module):
    def __init__(self, hidden_size, head_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.head_dim = head_dim
        # Norm weights are per-channel (head_dim) but typically shared or defined per head?
        # In the kernel: q_norm_weight is size [Dim] or [H*D]?
        # The kernel loads: tl.load(q_norm_weight_ptr + tl.arange(0, D))
        # Wait, the kernel loads `q_weight_0` and `q_weight_1` for each head?
        # No, looking at kernel: `q_weight_0 = tl.load(q_norm_weight_ptr + ...)`
        # The kernel ptr logic for weights suggests shape [head_dim]. 
        # It implies the SAME norm weight is reused across all heads or the input ptr is shifted.
        # Actually, standard QKNorm (e.g. Qwen) usually has shape [H, D] or [Dim].
        # Looking at kernel: `q_weight_0` is loaded ONCE per program? No, `tl.load` uses constant offset `tl.arange(0, D)`.
        # This implies the kernel assumes a SINGLE shared RMSNorm weight vector of size [head_dim] used for ALL heads,
        # OR the user passes a pointer offset per head (which the current kernel does NOT do).
        # Based on the code provided: `q_norm_weight_ptr` is static.
        # It loads `0..D` and `D..2D`. So weight shape is [head_dim].
        # This is LayerNorm over the head dimension, shared across heads? 
        # Or usually QKNorm is `(B, S, H, D)` -> norm over D. Weight is `[H, D]` or just `[D]`.
        # The kernel code loads `q_norm_weight_ptr + tl.arange` and `q_norm_weight_ptr + D + ...`.
        # It essentially treats the weight as having size [head_dim].
        
        self.q_norm_weight = torch.nn.Parameter(torch.ones(head_dim))
        self.k_norm_weight = torch.nn.Parameter(torch.ones(head_dim))

    def forward(self, q, k, freqs):
        # q, k: [B, S, H, D]
        
        # 1. RMS Norm
        # Manual RMS Norm to match kernel exact arithmetic
        # Kernel: rms = 1 / sqrt(mean(x^2) + eps)
        # x_norm = x * rms * weight
        
        q_rms = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
        q_norm = q * q_rms * self.q_norm_weight
        
        k_rms = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.eps)
        k_norm = k * k_rms * self.k_norm_weight
        
        # 2. RoPE
        # Generate cos/sin from freqs
        # freqs: [S, D//2] -> Kernel loads freqs and computes cos/sin
        # Kernel logic: 
        # freqs = load(freqs_ptr + pid * D...) where pid is Sequence Length index? 
        # pid = program_id(0). grid = (L,). So pid maps to Sequence Length `S`.
        # So freqs is shape [S, D//2].
        
        # Broadcasting: [S, 1, D//2] -> [B, S, H, D//2]
        # We need full head dim for rotation.
        # The kernel loads D elements of freqs. Wait, D in kernel is `head_dim // 2`.
        # cos = tl.cos(freqs).
        # q0 (first half) * cos - q1 * sin.
        # This implies cos/sin are applied to both halves identically.
        
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2) # [1, S, 1, D//2]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2) # [1, S, 1, D//2]
        
        # Expand to full head dim for convenient mult
        # The logic: x_full = [x1, x2]. 
        # x1_new = x1*c - x2*s
        # x2_new = x2*c + x1*s
        # We can implement this via complex numbers or the rotate_half helper.
        # rotate_half produces [-x2, x1].
        # (x * c) + (rotate_half(x) * s)
        # = [x1*c, x2*c] + [-x2*s, x1*s]
        # = [x1*c - x2*s, x2*c + x1*s]. Matches kernel exactly.
        
        # We need to concat cos/sin to shape [..., D]
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        
        q_out, k_out = apply_rotary_pos_emb(q_norm, k_norm, cos, sin)
        
        return q_out, k_out

# ==========================================
# Testing & Benchmarking Harness
# ==========================================

def run_tests():
    torch.manual_seed(42)
    
    # Configuration
    B = 2      # Batch size
    S = 128    # Sequence length
    H = 16     # Query Heads
    h = 4      # KV Heads
    D = 64     # Head Dim
    dtype = torch.float16
    device = "cuda"
    
    # Inputs
    # QKV Packed: [B, S, (H + 2h) * D]
    total_dim = (H + 2*h) * D
    qkv = torch.randn((B, S, total_dim), dtype=dtype, device=device, requires_grad=True)
    
    # Weights (RMS Norm weights)
    # The kernel treats these as simple vectors of size [D]
    q_norm_w = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    k_norm_w = torch.randn(D, dtype=dtype, device=device, requires_grad=True)
    
    # Frequencies for RoPE
    # Shape [S, D//2]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float().to(device) / D))
    t = torch.arange(S, device=device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq) # [S, D//2]
    # Kernel expects freqs to be contiguous in memory for load
    freqs = freqs.contiguous()
    
    # ==========================================
    # Run PyTorch Reference
    # ==========================================
    ref_model = ReferenceQKNormRoPE(total_dim, D).to(device)
    # Manually assign weights to ensure exact match
    ref_model.q_norm_weight = torch.nn.Parameter(q_norm_w.clone())
    ref_model.k_norm_weight = torch.nn.Parameter(k_norm_w.clone())
    
    # Slice QKV manually for reference
    # Layout: [B, S, (H+h+h)*D]. Assuming interleaved=False in creation, 
    # but strictly the kernel assumes specific layouts based on flags.
    # The kernel logic "interleaved=True" implies: [q, q, ..., k, v, q, ...] ?
    # Actually kernel logic: `row_offs` calculation suggests how heads are packed.
    # Let's test `interleaved=False` (Contiguous Q, then K, then V) first.
    
    q_start = 0
    k_start = H * D
    v_start = (H + h) * D
    
    # Important: The reference logic below assumes non-interleaved packing
    q_ref_in = qkv[..., :k_start].view(B, S, H, D)
    k_ref_in = qkv[..., k_start:v_start].view(B, S, h, D)
    v_ref_in = qkv[..., v_start:].view(B, S, h, D)
    
    q_ref_in.retain_grad()
    k_ref_in.retain_grad()
    
    # Forward
    q_ref_out, k_ref_out = ref_model(q_ref_in, k_ref_in, freqs)
    v_ref_out = v_ref_in.clone() # V is just passed through (or transposed)
    
    # Loss
    loss_ref = (q_ref_out.sum() + k_ref_out.sum() + v_ref_out.sum())
    loss_ref.backward()
    
    grad_qkv_ref = qkv.grad.clone()
    grad_qw_ref = ref_model.q_norm_weight.grad.clone()
    grad_kw_ref = ref_model.k_norm_weight.grad.clone()
    
    # ==========================================
    # Run Triton Kernel
    # ==========================================
    # Reset gradients
    qkv.grad = None
    q_norm_w.grad = None
    k_norm_w.grad = None
    
    # The wrapper expects transposed=False -> [B, S, Dim]
    # interleaved=False -> Q block, K block, V block
    qo_tri, ko_tri, vo_tri = triton_qk_norm_and_rope_forward(
        qkv, q_norm_w, k_norm_w, freqs,
        H=H, h=h, eps=1e-6,
        interleaved=False, transposed=False
    )
    
    # Verify Forward
    print(f"Checking Forward Pass...")
    if torch.allclose(qo_tri.float(), q_ref_out.float(), atol=1e-3, rtol=1e-3):
        print("✅ Q Output Match")
    else:
        print("❌ Q Output Mismatch")
        print("Diff:", (qo_tri - q_ref_out).abs().max().item())

    if torch.allclose(ko_tri.float(), k_ref_out.float(), atol=1e-3, rtol=1e-3):
        print("✅ K Output Match")
    else:
        print("❌ K Output Mismatch")
        
    # Backward
    # We need to synthesize gradients matching the implicit loss used in Reference
    # Reference loss was sum(). So we pass ones_like.
    gq = torch.ones_like(qo_tri)
    gk = torch.ones_like(ko_tri)
    gv = torch.ones_like(vo_tri)
    
    dqkv_tri, dqw_tri, dkw_tri = triton_qk_norm_and_rope_backward(
        gq, gk, gv, qkv,
        q_norm_w, k_norm_w, freqs,
        eps=1e-6, interleaved=False, transposed=False
    )
    
    print(f"Checking Backward Pass...")
    if torch.allclose(dqkv_tri.float(), grad_qkv_ref.float(), atol=1e-3, rtol=1e-3):
        print("✅ QKV Gradient Match")
    else:
        print("❌ QKV Gradient Mismatch")
        print("Diff:", (dqkv_tri - grad_qkv_ref).abs().max().item())
        
    if torch.allclose(dqw_tri.float(), grad_qw_ref.float(), atol=1e-2, rtol=1e-2):
        print("✅ Q Norm Weight Grad Match")
    else:
        print("❌ Q Norm Weight Grad Mismatch")
        print("Diff:", (dqw_tri - grad_qw_ref).abs().max().item())
        
    if torch.allclose(dkw_tri.float(), grad_kw_ref.float(), atol=1e-2, rtol=1e-2):
        print("✅ K Norm Weight Grad Match")
    else:
        print("❌ K Norm Weight Grad Mismatch")

    # ==========================================
    # Benchmarking
    # ==========================================
    print("\nStarting Benchmarks...")
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['S'],  # argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(1, 11)],  
            line_arg='provider',  
            line_vals=['torch', 'triton'],  
            line_names=['PyTorch', 'Triton'],  
            styles=[('blue', '-'), ('green', '-')],  
            ylabel='ms', 
            plot_name='qk_norm_rope_fwd',
            args={'B': B, 'H': H, 'h': h, 'D': D}, 
        )
    )
    def benchmark(B, S, H, h, D, provider):
        total_dim = (H + 2*h) * D
        qkv = torch.randn((B, S, total_dim), dtype=torch.float16, device=device)
        q_norm_w = torch.randn(D, dtype=torch.float16, device=device)
        k_norm_w = torch.randn(D, dtype=torch.float16, device=device)
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2).float().to(device) / D))
        t = torch.arange(S, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq).contiguous()
        
        quantiles = [0.5, 0.2, 0.8]
        
        if provider == 'torch':
            ref_model = ReferenceQKNormRoPE(total_dim, D).to(device)
            ref_model.q_norm_weight = torch.nn.Parameter(q_norm_w)
            ref_model.k_norm_weight = torch.nn.Parameter(k_norm_w)
            
            # Slice manually overhead included as part of standard torch usage
            def torch_func():
                q_in = qkv[..., :H*D].view(B, S, H, D)
                k_in = qkv[..., H*D:(H+h)*D].view(B, S, h, D)
                return ref_model(q_in, k_in, freqs)
                
            ms, min_ms, max_ms = triton.testing.do_bench(torch_func, quantiles=quantiles)
            
        if provider == 'triton':
            def triton_func():
                return triton_qk_norm_and_rope_forward(
                    qkv, q_norm_w, k_norm_w, freqs,
                    H=H, h=h, eps=1e-6,
                    interleaved=False, transposed=False
                )
            ms, min_ms, max_ms = triton.testing.do_bench(triton_func, quantiles=quantiles)
            
        return ms, max_ms, min_ms

    benchmark.run(show_plots=False, print_data=True)

if __name__ == "__main__":
    run_tests()