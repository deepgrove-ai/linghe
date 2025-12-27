import torch
import triton
import triton.language as tl


@triton.jit
def qk_norm_and_full_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    B,
    stride_seq,
    stride_batch,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D_HALF: tl.constexpr,
    DD: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)

    # 1. Load Frequencies
    freqs = tl.load(freqs_ptr + pid * D_HALF + tl.arange(0, D_HALF)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    # 2. Load Weights
    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D_HALF)).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D_HALF + tl.arange(0, D_HALF)).to(tl.float32)
    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D_HALF)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D_HALF + tl.arange(0, D_HALF)).to(tl.float32)

    w = H // h

    # 3. Handle Interleaving
    if INTERLEAVED:
        row_offs_q = tl.arange(0, H) + (tl.arange(0, H) // w) * 2
        q_base_ptr = qkv_ptr
    else:
        row_offs_q = tl.arange(0, H)
        q_base_ptr = qkv_ptr

    # Loop over Batch
    for i in range(B):
        if TRANSPOSED:
            base = q_base_ptr + pid * stride_seq + i * stride_batch
        else:
            base = q_base_ptr + i * stride_batch + pid * stride_seq

        q0 = tl.load(base + DD * row_offs_q[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)
        q1 = tl.load(base + D_HALF + DD * row_offs_q[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)

        ss = tl.sum(q0 * q0, axis=1) + tl.sum(q1 * q1, axis=1)
        rms = tl.rsqrt(ss / DD + eps)
        q0 = q0 * rms[:, None] * q_w0[None, :]
        q1 = q1 * rms[:, None] * q_w1[None, :]

        out0 = q0 * cos[None, :] - q1 * sin[None, :]
        out1 = q1 * cos[None, :] + q0 * sin[None, :]

        out_base = qo_ptr + i * L * H * DD + pid * H * DD
        tl.store(out_base + DD * tl.arange(0, H)[:, None] + tl.arange(0, D_HALF)[None, :], out0)
        tl.store(out_base + D_HALF + DD * tl.arange(0, H)[:, None] + tl.arange(0, D_HALF)[None, :], out1)

    # Handle K
    if INTERLEAVED:
        row_offs_kv = tl.arange(0, h) * (w + 2)
        k_base_ptr = qkv_ptr + DD * w
    else:
        row_offs_kv = tl.arange(0, h)
        k_base_ptr = qkv_ptr + DD * H

    for i in range(B):
        if TRANSPOSED:
            base = k_base_ptr + pid * stride_seq + i * stride_batch
        else:
            base = k_base_ptr + i * stride_batch + pid * stride_seq

        k0 = tl.load(base + DD * row_offs_kv[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)
        k1 = tl.load(base + D_HALF + DD * row_offs_kv[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)

        ss = tl.sum(k0 * k0, axis=1) + tl.sum(k1 * k1, axis=1)
        rms = tl.rsqrt(ss / DD + eps)
        k0 = k0 * rms[:, None] * k_w0[None, :]
        k1 = k1 * rms[:, None] * k_w1[None, :]

        out0 = k0 * cos[None, :] - k1 * sin[None, :]
        out1 = k1 * cos[None, :] + k0 * sin[None, :]

        out_base = ko_ptr + i * L * h * DD + pid * h * DD
        tl.store(out_base + DD * tl.arange(0, h)[:, None] + tl.arange(0, D_HALF)[None, :], out0)
        tl.store(out_base + D_HALF + DD * tl.arange(0, h)[:, None] + tl.arange(0, D_HALF)[None, :], out1)

    # Handle V
    if INTERLEAVED:
        v_base_ptr = qkv_ptr + DD * w + DD
    else:
        v_base_ptr = qkv_ptr + DD * H + DD * h

    for i in range(B):
        if TRANSPOSED:
            base = v_base_ptr + pid * stride_seq + i * stride_batch
        else:
            base = v_base_ptr + i * stride_batch + pid * stride_seq

        v0 = tl.load(base + DD * row_offs_kv[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)
        v1 = tl.load(base + D_HALF + DD * row_offs_kv[:, None] + tl.arange(0, D_HALF)[None, :]).to(tl.float32)

        out_base = vo_ptr + i * L * h * DD + pid * h * DD
        tl.store(out_base + DD * tl.arange(0, h)[:, None] + tl.arange(0, D_HALF)[None, :], v0)
        tl.store(out_base + D_HALF + DD * tl.arange(0, h)[:, None] + tl.arange(0, D_HALF)[None, :], v1)


def triton_qk_norm_and_rope_forward(qkv, q_norm_weight, k_norm_weight, freqs, H=32, h=4, eps=1e-6, interleaved=True, transposed=False, num_warps=4, num_stages=3):
    if transposed:
        L, B, Dim = qkv.shape
        stride_seq = qkv.stride(0)
        stride_batch = qkv.stride(1)
    else:
        B, L, Dim = qkv.shape
        stride_batch = qkv.stride(0)
        stride_seq = qkv.stride(1)

    head_dim = Dim // (H + 2 * h)
    D_HALF = head_dim // 2

    qo = torch.empty((B, L, H, head_dim), dtype=qkv.dtype, device=qkv.device)
    ko = torch.empty((B, L, h, head_dim), dtype=qkv.dtype, device=qkv.device)
    vo = torch.empty((B, L, h, head_dim), dtype=qkv.dtype, device=qkv.device)

    grid = (L,)
    qk_norm_and_full_rope_forward_kernel[grid](
        qkv, q_norm_weight, k_norm_weight, freqs, qo, ko, vo, B, stride_seq, stride_batch, eps,
        H=H, h=h, D_HALF=D_HALF, DD=head_dim, INTERLEAVED=interleaved, TRANSPOSED=transposed,
        num_warps=num_warps, num_stages=num_stages,
    )
    return qo, ko, vo


# -----------------------------
# Backward Kernel
# -----------------------------
@triton.jit
def qk_norm_and_full_rope_backward_kernel(
    gq_ptr, gk_ptr, gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr, k_norm_weight_ptr,
    freqs_ptr,
    dqkv_ptr,
    tmp_dqw_ptr, tmp_dkw_ptr,
    B,
    stride_seq,
    stride_batch,
    eps,
    H: tl.constexpr, h: tl.constexpr, D_HALF: tl.constexpr, DD: tl.constexpr,
    INTERLEAVED: tl.constexpr, TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    offs_d = tl.arange(0, D_HALF)

    # Load shared data
    freqs = tl.load(freqs_ptr + pid * D_HALF + offs_d).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)

    q_w0 = tl.load(q_norm_weight_ptr + offs_d).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D_HALF + offs_d).to(tl.float32)
    k_w0 = tl.load(k_norm_weight_ptr + offs_d).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D_HALF + offs_d).to(tl.float32)

    w = H // h

    # Accumulators for weight gradients
    dqw0 = tl.zeros((D_HALF,), dtype=tl.float32)
    dqw1 = tl.zeros((D_HALF,), dtype=tl.float32)
    dkw0 = tl.zeros((D_HALF,), dtype=tl.float32)
    dkw1 = tl.zeros((D_HALF,), dtype=tl.float32)

    # --- Q Gradients ---
    if INTERLEAVED:
        row_offs_q = tl.arange(0, H) + (tl.arange(0, H) // w) * 2
        q_base_ptr = qkv_ptr
        dq_base_ptr = dqkv_ptr
    else:
        row_offs_q = tl.arange(0, H)
        q_base_ptr = qkv_ptr
        dq_base_ptr = dqkv_ptr

    for i in range(B):
        # Load Grad
        gq_base = gq_ptr + i * L * H * DD + pid * H * DD
        gq0 = tl.load(gq_base + DD * tl.arange(0, H)[:, None] + offs_d[None, :]).to(tl.float32)
        gq1 = tl.load(gq_base + D_HALF + DD * tl.arange(0, H)[:, None] + offs_d[None, :]).to(tl.float32)

        # Inverse RoPE
        gg0 = gq0 * cos[None, :] + gq1 * sin[None, :]
        gg1 = gq1 * cos[None, :] - gq0 * sin[None, :]

        # Load Input
        if TRANSPOSED:
            base = q_base_ptr + pid * stride_seq + i * stride_batch
            dq_out = dq_base_ptr + pid * stride_seq + i * stride_batch
        else:
            base = q_base_ptr + i * stride_batch + pid * stride_seq
            dq_out = dq_base_ptr + i * stride_batch + pid * stride_seq

        q0 = tl.load(base + DD * row_offs_q[:, None] + offs_d[None, :]).to(tl.float32)
        q1 = tl.load(base + D_HALF + DD * row_offs_q[:, None] + offs_d[None, :]).to(tl.float32)

        # RMS Backward
        ss = tl.sum(q0 * q0, axis=1) + tl.sum(q1 * q1, axis=1)
        inv_rms = tl.rsqrt(ss / DD + eps)
        r = inv_rms[:, None]

        dqw0 += tl.sum(gg0 * (q0 * r), axis=0)
        dqw1 += tl.sum(gg1 * (q1 * r), axis=0)

        s = tl.sum(q0 * gg0 * q_w0[None, :], axis=1) + tl.sum(q1 * gg1 * q_w1[None, :], axis=1)
        r3_over_n = (inv_rms * inv_rms * inv_rms) / DD
        
        dq0 = r * (gg0 * q_w0[None, :]) - r3_over_n[:, None] * q0 * s[:, None]
        dq1 = r * (gg1 * q_w1[None, :]) - r3_over_n[:, None] * q1 * s[:, None]

        tl.store(dq_out + DD * row_offs_q[:, None] + offs_d[None, :], dq0)
        tl.store(dq_out + D_HALF + DD * row_offs_q[:, None] + offs_d[None, :], dq1)

    tl.store(tmp_dqw_ptr + pid * DD + offs_d, dqw0)
    tl.store(tmp_dqw_ptr + pid * DD + D_HALF + offs_d, dqw1)

    # --- K Gradients ---
    if INTERLEAVED:
        row_offs_kv = tl.arange(0, h) * (w + 2)
        k_base_ptr = qkv_ptr + DD * w
        dk_base_ptr = dqkv_ptr + DD * w
    else:
        row_offs_kv = tl.arange(0, h)
        k_base_ptr = qkv_ptr + DD * H
        dk_base_ptr = dqkv_ptr + DD * H

    for i in range(B):
        gk_base = gk_ptr + i * L * h * DD + pid * h * DD
        gk0 = tl.load(gk_base + DD * tl.arange(0, h)[:, None] + offs_d[None, :]).to(tl.float32)
        gk1 = tl.load(gk_base + D_HALF + DD * tl.arange(0, h)[:, None] + offs_d[None, :]).to(tl.float32)

        gg0 = gk0 * cos[None, :] + gk1 * sin[None, :]
        gg1 = gk1 * cos[None, :] - gk0 * sin[None, :]

        if TRANSPOSED:
            base = k_base_ptr + pid * stride_seq + i * stride_batch
            dq_out = dk_base_ptr + pid * stride_seq + i * stride_batch
        else:
            base = k_base_ptr + i * stride_batch + pid * stride_seq
            dq_out = dk_base_ptr + i * stride_batch + pid * stride_seq

        k0 = tl.load(base + DD * row_offs_kv[:, None] + offs_d[None, :]).to(tl.float32)
        k1 = tl.load(base + D_HALF + DD * row_offs_kv[:, None] + offs_d[None, :]).to(tl.float32)

        ss = tl.sum(k0 * k0, axis=1) + tl.sum(k1 * k1, axis=1)
        inv_rms = tl.rsqrt(ss / DD + eps)
        r = inv_rms[:, None]

        dkw0 += tl.sum(gg0 * (k0 * r), axis=0)
        dkw1 += tl.sum(gg1 * (k1 * r), axis=0)

        s = tl.sum(k0 * gg0 * k_w0[None, :], axis=1) + tl.sum(k1 * gg1 * k_w1[None, :], axis=1)
        r3_over_n = (inv_rms * inv_rms * inv_rms) / DD

        dk0 = r * (gg0 * k_w0[None, :]) - r3_over_n[:, None] * k0 * s[:, None]
        dk1 = r * (gg1 * k_w1[None, :]) - r3_over_n[:, None] * k1 * s[:, None]

        tl.store(dq_out + DD * row_offs_kv[:, None] + offs_d[None, :], dk0)
        tl.store(dq_out + D_HALF + DD * row_offs_kv[:, None] + offs_d[None, :], dk1)

    tl.store(tmp_dkw_ptr + pid * DD + offs_d, dkw0)
    tl.store(tmp_dkw_ptr + pid * DD + D_HALF + offs_d, dkw1)

    # --- V Gradients ---
    if INTERLEAVED:
        dv_base_ptr = dqkv_ptr + DD * w + DD
    else:
        dv_base_ptr = dqkv_ptr + DD * H + DD * h

    for i in range(B):
        gv_base = gv_ptr + i * L * h * DD + pid * h * DD
        gv0 = tl.load(gv_base + DD * tl.arange(0, h)[:, None] + offs_d[None, :]).to(tl.float32)
        gv1 = tl.load(gv_base + D_HALF + DD * tl.arange(0, h)[:, None] + offs_d[None, :]).to(tl.float32)

        if TRANSPOSED:
            dq_out = dv_base_ptr + pid * stride_seq + i * stride_batch
        else:
            dq_out = dv_base_ptr + i * stride_batch + pid * stride_seq

        tl.store(dq_out + DD * row_offs_kv[:, None] + offs_d[None, :], gv0)
        tl.store(dq_out + D_HALF + DD * row_offs_kv[:, None] + offs_d[None, :], gv1)


def triton_qk_norm_and_rope_backward(gq, gk, gv, qkv, q_norm_weight, k_norm_weight, freqs, H=32, h=4, eps=1e-6, interleaved=True, transposed=False, num_warps=4, num_stages=3):
    gq = gq.contiguous()
    gk = gk.contiguous()
    gv = gv.contiguous()
    dqkv = torch.empty_like(qkv, memory_format=torch.contiguous_format)
    
    if transposed:
        L, B, Dim = qkv.shape
        stride_seq = qkv.stride(0)
        stride_batch = qkv.stride(1)
    else:
        B, L, Dim = qkv.shape
        stride_batch = qkv.stride(0)
        stride_seq = qkv.stride(1)

    head_dim = Dim // (H + 2 * h)
    D_HALF = head_dim // 2

    tmp_dqw = torch.empty((L, head_dim), device=qkv.device, dtype=torch.float32)
    tmp_dkw = torch.empty((L, head_dim), device=qkv.device, dtype=torch.float32)

    grid = (L,)
    qk_norm_and_full_rope_backward_kernel[grid](
        gq, gk, gv, qkv, q_norm_weight, k_norm_weight, freqs, dqkv, tmp_dqw, tmp_dkw, B,
        stride_seq, stride_batch, eps,
        H=H, h=h, D_HALF=D_HALF, DD=head_dim, INTERLEAVED=interleaved, TRANSPOSED=transposed,
        num_warps=num_warps, num_stages=num_stages,
    )

    dqw = tmp_dqw.sum(0).to(q_norm_weight.dtype)
    dkw = tmp_dkw.sum(0).to(k_norm_weight.dtype)
    return dqkv, dqw, dkw