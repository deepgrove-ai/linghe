"""
FP8 Scaling-Aware Transpose Implementation in Triton

Based on the paper "FP8-Flow-MoE: A Casting-Free FP8 Recipe without Double Quantization Error"

Scale Layout (row-wise quantization with groups along columns):
- Input [M, N] with scales [M, N // group_size]
- Output [N, M] with scales [N, M // group_size]

This implementation uses torch.float8_e4m3fn for FP8 tensors and float32 for scales.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple


FP8_E4M3_MAX = 448.0
DEFAULT_GROUP_SIZE = 128


def _get_fp8_dtype():
    """Get the FP8 E4M3 dtype."""
    if hasattr(torch, 'float8_e4m3fn'):
        return torch.float8_e4m3fn
    raise RuntimeError("PyTorch 2.1+ with CUDA required for float8_e4m3fn")

@triton.jit
def _fp8_transpose_kernel(
    input_ptr,
    input_scale_ptr,
    output_ptr,
    output_scale_ptr,
    M, N,
    num_input_groups,   # N // group_size
    num_output_groups,  # M // group_size
    GROUP_SIZE: tl.constexpr,
):
    """
    Scaling-aware FP8 transpose kernel.
    
    Processes GROUP_SIZE x GROUP_SIZE blocks.
    For each block:
    1. Load scales for all rows (same input group index)
    2. Find Smax = max of these scales  
    3. Compute k = log2(Smax) - log2(row_scale)
    4. Adjust FP8 exponents: E_new = E - k
    5. Transpose and store
    6. Store Smax as output scale
    """
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, GROUP_SIZE)
    block_m = pid // num_n_blocks
    block_n = pid % num_n_blocks
    
    m_start = block_m * GROUP_SIZE
    n_start = block_n * GROUP_SIZE
    
    # Scale group indices
    input_group = block_n    # Column group in input
    output_group = block_m   # Row group in output (transposed)
    
    m_offs = tl.arange(0, GROUP_SIZE)
    n_offs = tl.arange(0, GROUP_SIZE)
    
    m_idx = m_start + m_offs
    n_idx = n_start + n_offs
    
    m_mask = m_idx < M
    n_mask = n_idx < N
    
    # Load input scales: [M, num_input_groups]
    # We need scales[m_idx, input_group]
    scale_ptrs = input_scale_ptr + m_idx * num_input_groups + input_group
    scales = tl.load(scale_ptrs, mask=m_mask, other=1.0)
    
    # Find max scale in block
    s_max = tl.max(scales)
    
    # Compute k = log2(s_max) - log2(scale) for each row
    # For power-of-2 scales: use float32 exponent extraction
    scale_bits = scales.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF).to(tl.int32) - 127
    
    smax_bits = s_max.to(tl.uint32, bitcast=True)
    smax_exp = ((smax_bits >> 23) & 0xFF).to(tl.int32) - 127
    
    k = smax_exp - scale_exp  # [GROUP_SIZE], k >= 0
    
    # Load FP8 data block as uint8
    data_ptrs = input_ptr + m_idx[:, None] * N + n_idx[None, :]
    mask_2d = m_mask[:, None] & n_mask[None, :]
    data = tl.load(data_ptrs, mask=mask_2d, other=0).to(tl.uint8)
    
    # Parse FP8 E4M3: SEEEEMMM
    signs = (data >> 7) & 1
    exps = ((data >> 3) & 0xF).to(tl.int32)
    mants = data & 0x7
    
    # Adjust exponents per row
    k_2d = k[:, None]
    new_exps = exps - k_2d
    
    # Clamp to valid range [0, 15]
    new_exps = tl.maximum(new_exps, 0)
    new_exps = tl.minimum(new_exps, 15)
    
    # Preserve zeros (exp=0 stays 0)
    is_zero = (exps == 0)
    new_exps = tl.where(is_zero, exps, new_exps)
    
    # Reassemble FP8
    new_data = (signs << 7) | (new_exps.to(tl.uint8) << 3) | mants
    
    # Transpose
    new_data_T = tl.trans(new_data)
    
    # Store transposed: output[n_idx, m_idx]
    out_ptrs = output_ptr + n_idx[:, None] * M + m_idx[None, :]
    mask_2d_T = n_mask[:, None] & m_mask[None, :]
    tl.store(out_ptrs, new_data_T, mask=mask_2d_T)
    
    # Store output scale: all rows in output block get s_max
    out_scale_ptrs = output_scale_ptr + n_idx * num_output_groups + output_group
    tl.store(out_scale_ptrs, tl.full([GROUP_SIZE], s_max, dtype=tl.float32), mask=n_mask)


def fp8_scaling_aware_transpose(
    input_fp8: torch.Tensor,
    input_scales: torch.Tensor,
    group_size: int = DEFAULT_GROUP_SIZE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaling-aware FP8 transpose.
    
    Args:
        input_fp8: [M, N] tensor, dtype float8_e4m3fn
        input_scales: [M, N // group_size] tensor, dtype float32
        group_size: Elements per scale group (default 128)
        
    Returns:
        output_fp8: [N, M] tensor, dtype float8_e4m3fn
        output_scales: [N, M // group_size] tensor, dtype float32
    """
    fp8_dtype = _get_fp8_dtype()
    
    if input_fp8.dtype == fp8_dtype:
        input_bits = input_fp8.view(torch.uint8)
    elif input_fp8.dtype == torch.uint8:
        input_bits = input_fp8
    else:
        raise TypeError(f"Expected float8_e4m3fn or uint8, got {input_fp8.dtype}")
    
    assert input_scales.dtype == torch.float32
    assert input_bits.is_contiguous() and input_scales.is_contiguous()
    
    M, N = input_bits.shape
    num_input_groups = (N + group_size - 1) // group_size
    num_output_groups = (M + group_size - 1) // group_size
    
    assert input_scales.shape == (M, num_input_groups), \
        f"Expected scales shape ({M}, {num_input_groups}), got {input_scales.shape}"
    
    output_bits = torch.empty((N, M), dtype=torch.uint8, device=input_bits.device)
    output_scales = torch.empty((N, num_output_groups), dtype=torch.float32, device=input_bits.device)
    
    grid = (triton.cdiv(M, group_size) * triton.cdiv(N, group_size),)
    
    _fp8_transpose_kernel[grid](
        input_bits, input_scales,
        output_bits, output_scales,
        M, N,
        num_input_groups, num_output_groups,
        GROUP_SIZE=group_size,
    )
    
    return output_bits.view(fp8_dtype), output_scales
