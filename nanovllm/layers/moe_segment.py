"""Triton 真段式 grouped GEMM（MoE 专家批量）——offsets/counts 驱动，无 padding。

与 padded-bmm（cuBLAS batched）的实测边界（benchmarks/_moe_triton_segment.py，
K=512/O=1536/E∈{8,128}，本机 sm_120）：
- 小段（decode 类 R<512、段长 < BM）→ bmm 快 3-10×（本内核 BM 地板 + 空程序）；
- 大 E（128）prefill → 持平（1.05-1.14×；cuBLAS 128-batch 已高效）；
- **小 E + 极长段**（E=8、R=32768、段 ~4K 行）→ 本内核 0.70×（快 43%）——
  batched bmm 对"少 batch 每 batch 超大"的切分不如单一大 GEMM。
结论：默认路径仍是 padded-bmm；MoE(segment_backend=True) 显式启用本内核
（当前窗口窄，主要为后续内核优化/无 cuBLAS 场景保留）。

正确性：4 场景（E8/E8-big/decode-like/E128）vs 逐段 torch 参考 rel err = 0.0（位级）。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def moe_segment_mm_kernel(
    a_ptr, w_ptr, y_ptr, offs_ptr, counts_ptr,
    R, K, O, E,
    stride_ak,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y[段e行, O] = a[段e行, K] @ w[e]（w [E, K, O] 行主序，3D 堆叠转置布局）。

    程序 (e, tm, tn)：专家 e 的第 tm 个 M 块（BM 行）× 第 tn 个 N 块（BN 列）。
    段 e = [offs[e], offs[e]+counts[e])；段内行不足 BM 的行掩码 0（贡献 0，非 padding
    计算——块数由 max_n 决定，空 M 块提前返回）。
    """
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
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * K * O
    for k in range(0, K, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < K
        a_tile = tl.load(a_ptr + rows[:, None] * stride_ak + kk[None, :],
                         mask=rmask[:, None] & kmask[None, :], other=0.0)
        w_tile = tl.load(w_base + kk[:, None] * O + cols[None, :],
                         mask=kmask[:, None] & cmask[None, :], other=0.0)
        acc += tl.dot(a_tile, w_tile)
    y_ptrs = y_ptr + rows[:, None] * O + cols[None, :]
    tl.store(y_ptrs, acc.to(a_ptr.dtype.element_ty),
             mask=rmask[:, None] & cmask[None, :])


def moe_segment_mm(a: torch.Tensor, w3d: torch.Tensor, offs: torch.Tensor,
                   counts: torch.Tensor, BM: int = 64, BN: int = 128,
                   BK: int = 64, num_warps: int = 4) -> torch.Tensor:
    """host 封装。a [R, K]（段连续）；w3d [E, K, O]；offs/counts [E]（device int）；
    返回 y [R, O]。"""
    R, K = a.shape
    E, _, O = w3d.shape
    y = torch.empty(R, O, device=a.device, dtype=a.dtype)
    max_n = int(counts.max())
    if R == 0 or max_n == 0:
        return y
    nt_m = (max_n + BM - 1) // BM
    nt_n = (O + BN - 1) // BN
    grid = (E, nt_m, nt_n)
    moe_segment_mm_kernel[grid](
        a, w3d, y, offs, counts, R, K, O, E, a.stride(0),
        BM=BM, BN=BN, BK=BK, num_warps=num_warps)
    return y
