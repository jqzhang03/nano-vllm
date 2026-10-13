"""Segmented INT4 expert GEMMs for MoE layers.

Expert weights remain in their per-module packed buffers. Small device-side
pointer tables let one Triton launch address all experts without stacking or
duplicating a large quantized layer. The gate/up kernel computes both
projections in one launch and emits ``[gate | up]`` for the SwiGLU step.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _segment_int4_mm_kernel(
    a_ptr, w_ptrs_ptr, scale_ptrs_ptr, y_ptr, offs_ptr, counts_ptr,
    R, K, O, E, num_groups,
    stride_am, stride_ak,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP: tl.constexpr, SCALE_DTYPE: tl.constexpr,
):
    e = tl.program_id(0)
    tm = tl.program_id(1)
    tn = tl.program_id(2)
    n_e = tl.load(counts_ptr + e)
    if tm * BM >= n_e:
        return

    off = tl.load(offs_ptr + e)
    rows = off + tm * BM + tl.arange(0, BM)
    rmask = rows < off + n_e
    cols = tn * BN + tl.arange(0, BN)
    cmask = cols < O
    w_base = tl.load(w_ptrs_ptr + e).to(tl.pointer_type(tl.int8))
    s_base = tl.load(scale_ptrs_ptr + e).to(tl.pointer_type(SCALE_DTYPE))
    w_stride = K // 2
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BK) * BK, BK):
        for gg in tl.static_range(0, BK // GROUP):
            jj = gg * (GROUP // 2) + tl.arange(0, GROUP // 2)
            k_even = k + 2 * jj
            kmask = k_even < K
            a_even = tl.load(
                a_ptr + rows[:, None] * stride_am + k_even[None, :] * stride_ak,
                mask=rmask[:, None] & kmask[None, :], other=0.0)
            a_odd = tl.load(
                a_ptr + rows[:, None] * stride_am
                + (k_even[None, :] + 1) * stride_ak,
                mask=rmask[:, None] & kmask[None, :], other=0.0)
            packed = tl.load(
                w_base + cols[None, :] * w_stride + (k // 2 + jj)[:, None],
                mask=kmask[:, None] & cmask[None, :], other=0)
            group = (k + gg * GROUP) // GROUP
            scale = tl.load(
                s_base + cols * num_groups + group,
                mask=cmask & (group < num_groups), other=1.0).to(tl.float32)
            low = ((packed & 0x0F) - 8).to(tl.float32) * scale[None, :]
            high = (((packed >> 4) & 0x0F) - 8).to(tl.float32) * scale[None, :]
            elem_ty = a_ptr.dtype.element_ty
            acc += tl.dot(a_even, low.to(elem_ty), out_dtype=tl.float32)
            acc += tl.dot(a_odd, high.to(elem_ty), out_dtype=tl.float32)

    out = y_ptr + rows[:, None] * O + cols[None, :]
    tl.store(out, acc.to(a_ptr.dtype.element_ty), mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _segment_int4_gup_kernel(
    a_ptr, gate_ptrs_ptr, gate_scale_ptrs_ptr,
    up_ptrs_ptr, up_scale_ptrs_ptr, y_ptr, offs_ptr, counts_ptr,
    R, K, I, E, num_groups,
    stride_am, stride_ak,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP: tl.constexpr, SCALE_DTYPE: tl.constexpr,
):
    """Run gate and up tiles in one launch without doubling CTA accumulators."""
    e = tl.program_id(0)
    tm = tl.program_id(1)
    tn = tl.program_id(2)
    n_e = tl.load(counts_ptr + e)
    if tm * BM >= n_e:
        return

    off = tl.load(offs_ptr + e)
    rows = off + tm * BM + tl.arange(0, BM)
    rmask = rows < off + n_e
    n_tiles = tl.cdiv(I, BN)
    is_up = tn >= n_tiles
    n_tile = tl.where(is_up, tn - n_tiles, tn)
    cols = n_tile * BN + tl.arange(0, BN)
    cmask = cols < I
    gate_base = tl.load(gate_ptrs_ptr + e).to(tl.pointer_type(tl.int8))
    gate_scale_base = tl.load(gate_scale_ptrs_ptr + e).to(
        tl.pointer_type(SCALE_DTYPE))
    up_base = tl.load(up_ptrs_ptr + e).to(tl.pointer_type(tl.int8))
    up_scale_base = tl.load(up_scale_ptrs_ptr + e).to(tl.pointer_type(SCALE_DTYPE))
    w_base = tl.where(is_up, up_base, gate_base)
    scale_base = tl.where(is_up, up_scale_base, gate_scale_base)
    w_stride = K // 2
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BK) * BK, BK):
        for gg in tl.static_range(0, BK // GROUP):
            jj = gg * (GROUP // 2) + tl.arange(0, GROUP // 2)
            k_even = k + 2 * jj
            kmask = k_even < K
            a_even = tl.load(
                a_ptr + rows[:, None] * stride_am + k_even[None, :] * stride_ak,
                mask=rmask[:, None] & kmask[None, :], other=0.0)
            a_odd = tl.load(
                a_ptr + rows[:, None] * stride_am
                + (k_even[None, :] + 1) * stride_ak,
                mask=rmask[:, None] & kmask[None, :], other=0.0)
            packed = tl.load(
                w_base + cols[None, :] * w_stride + (k // 2 + jj)[:, None],
                mask=kmask[:, None] & cmask[None, :], other=0)
            group = (k + gg * GROUP) // GROUP
            scale = tl.load(
                scale_base + cols * num_groups + group,
                mask=cmask & (group < num_groups), other=1.0).to(tl.float32)
            low = ((packed & 0x0F) - 8).to(tl.float32) * scale[None, :]
            high = (((packed >> 4) & 0x0F) - 8).to(tl.float32) * scale[None, :]
            elem_ty = a_ptr.dtype.element_ty
            acc += tl.dot(a_even, low.to(elem_ty), out_dtype=tl.float32)
            acc += tl.dot(a_odd, high.to(elem_ty), out_dtype=tl.float32)

    out_offset = tl.where(is_up, I, 0)
    out = y_ptr + rows[:, None] * (2 * I) + (out_offset + cols[None, :])
    tl.store(out, acc.to(a_ptr.dtype.element_ty),
             mask=rmask[:, None] & cmask[None, :])


def _scale_pointer_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise TypeError(f"unsupported INT4 scale dtype: {dtype}")


def _validate(a: torch.Tensor, offsets: torch.Tensor, counts: torch.Tensor,
              group_size: int) -> None:
    if a.ndim != 2:
        raise ValueError(f"expected activations [R, K], got {tuple(a.shape)}")
    if a.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"INT4 MoE GEMM expects bf16/fp16 activations, got {a.dtype}")
    if group_size <= 0 or group_size % 2 or 128 % group_size:
        raise ValueError(f"INT4 group size must be even and divide 128, got {group_size}")
    if offsets.device != a.device or counts.device != a.device:
        raise ValueError("INT4 MoE offsets/counts must be on the activation device")


def moe_segment_int4_mm(a: torch.Tensor, weight_ptrs: torch.Tensor,
                        scale_ptrs: torch.Tensor, offsets: torch.Tensor,
                        counts: torch.Tensor, out_features: int,
                        group_size: int, *, BM: int = 16, BN: int = 128,
                        BK: int = 128, num_warps: int = 4,
                        max_tokens_per_expert: int | None = None) -> torch.Tensor:
    """Apply per-expert packed INT4 weights to expert-sorted activation rows."""
    _validate(a, offsets, counts, group_size)
    R, K = a.shape
    E = counts.numel()
    if K % group_size:
        raise ValueError(f"K={K} must be divisible by INT4 group size {group_size}")
    num_groups = K // group_size
    y = torch.empty((R, out_features), device=a.device, dtype=a.dtype)
    if R == 0 or E == 0:
        return y
    max_n = (int(counts.max()) if max_tokens_per_expert is None
             else max_tokens_per_expert)
    if max_n == 0:
        return y
    grid = (E, triton.cdiv(max_n, BM), triton.cdiv(out_features, BN))
    _segment_int4_mm_kernel[grid](
        a, weight_ptrs, scale_ptrs, y, offsets, counts,
        R, K, out_features, E, num_groups, a.stride(0), a.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP=group_size,
        SCALE_DTYPE=_scale_pointer_type(a.dtype), num_warps=num_warps,
        num_stages=2,
    )
    return y


def moe_segment_int4_gup(a: torch.Tensor,
                         gate_ptrs: torch.Tensor, gate_scale_ptrs: torch.Tensor,
                         up_ptrs: torch.Tensor, up_scale_ptrs: torch.Tensor,
                         offsets: torch.Tensor, counts: torch.Tensor,
                         intermediate_size: int, group_size: int, *,
                         BM: int = 16, BN: int = 128, BK: int = 128,
                         num_warps: int = 4,
                         max_tokens_per_expert: int | None = None) -> torch.Tensor:
    """Fused gate/up projections; returns ``[R, 2*I]`` as ``[gate | up]``."""
    _validate(a, offsets, counts, group_size)
    R, K = a.shape
    E = counts.numel()
    if K % group_size:
        raise ValueError(f"K={K} must be divisible by INT4 group size {group_size}")
    num_groups = K // group_size
    y = torch.empty((R, 2 * intermediate_size), device=a.device, dtype=a.dtype)
    if R == 0 or E == 0:
        return y
    max_n = (int(counts.max()) if max_tokens_per_expert is None
             else max_tokens_per_expert)
    if max_n == 0:
        return y
    grid = (E, triton.cdiv(max_n, BM), 2 * triton.cdiv(intermediate_size, BN))
    _segment_int4_gup_kernel[grid](
        a, gate_ptrs, gate_scale_ptrs, up_ptrs, up_scale_ptrs, y,
        offsets, counts, R, K, intermediate_size, E, num_groups,
        a.stride(0), a.stride(1),
        BM=BM, BN=BN, BK=BK, GROUP=group_size,
        SCALE_DTYPE=_scale_pointer_type(a.dtype), num_warps=num_warps,
        num_stages=2,
    )
    return y
