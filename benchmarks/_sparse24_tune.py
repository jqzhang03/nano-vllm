"""2:4 稀疏内核调优：扫 tile/warps/BLOCK_K（4路拆分 + 掩码重建权重块）。

软件 2:4 的 MMA 数与稠密相同（无硬件稀疏MMA），赢面只在权重带宽（0.625×），
因此小 M、大 N 的形状最可能有正收益（lm_head / gate_up）。
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEV = "cuda"
torch.manual_seed(0)


@triton.jit
def gemm_sparse24_kernel_t(
    a_ptr, v_ptr, idx_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_vn, stride_vk, stride_in, stride_ik, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, BLOCK_K // 4)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + 4 * offs_g)[None, :] * stride_ak
        a0 = tl.load(a_ptrs, mask=m_mask, other=0.0)
        a1 = tl.load(a_ptrs + stride_ak, mask=m_mask, other=0.0)
        a2 = tl.load(a_ptrs + 2 * stride_ak, mask=m_mask, other=0.0)
        a3 = tl.load(a_ptrs + 3 * stride_ak, mask=m_mask, other=0.0)
        v_ptrs = v_ptr + offs_n[None, :] * stride_vn + (k // 2 + 2 * offs_g)[:, None] * stride_vk
        v_lo = tl.load(v_ptrs, mask=n_mask, other=0.0)
        v_hi = tl.load(v_ptrs + stride_vk, mask=n_mask, other=0.0)
        i_ptrs = idx_ptr + offs_n[None, :] * stride_in + (k // 4 + offs_g)[:, None] * stride_ik
        b8 = tl.load(i_ptrs, mask=n_mask, other=0)
        i_lo = b8 & 0x3
        i_hi = (b8 >> 2) & 0x3
        vp0 = tl.where(i_lo == 0, v_lo, 0.0) + tl.where(i_hi == 0, v_hi, 0.0)
        acc += tl.dot(a0, vp0.to(tl.bfloat16), out_dtype=tl.float32)
        vp1 = tl.where(i_lo == 1, v_lo, 0.0) + tl.where(i_hi == 1, v_hi, 0.0)
        acc += tl.dot(a1, vp1.to(tl.bfloat16), out_dtype=tl.float32)
        vp2 = tl.where(i_lo == 2, v_lo, 0.0) + tl.where(i_hi == 2, v_hi, 0.0)
        acc += tl.dot(a2, vp2.to(tl.bfloat16), out_dtype=tl.float32)
        vp3 = tl.where(i_lo == 3, v_lo, 0.0) + tl.where(i_hi == 3, v_hi, 0.0)
        acc += tl.dot(a3, vp3.to(tl.bfloat16), out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        v_ptrs += (BLOCK_K // 2) * stride_vk
        i_ptrs += (BLOCK_K // 4) * stride_ik

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def prune_2_4(w):
    w4 = w.view(*w.shape[:-1], w.shape[-1] // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    return (w4 * mask).view(w.shape)


def pack_2_4(w):
    N, K = w.shape
    w4 = w.view(N, K // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    v = torch.gather(w4 * mask, -1, keep).view(N, K // 2).contiguous()
    idx = keep.to(torch.uint8).view(N, K // 2)
    idx_packed = (idx[:, 0::2] | (idx[:, 1::2] << 2)).contiguous()
    return v, idx_packed


def run_sparse(a, v, idx, bm, bn, bk, warps, stages):
    M, K = a.shape
    N = v.shape[0]
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_sparse24_kernel_t[grid](
        a, v, idx, out, M, N, K,
        a.stride(0), a.stride(1), v.stride(0), v.stride(1), idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return out


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# 正确性
for M, K, N in [(8, 1024, 4096), (256, 1024, 4096)]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    v, idx = pack_2_4(w)
    ref = F.linear(x, prune_2_4(w))
    out = run_sparse(x, v, idx, 16, 128, 128, 4, 2)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"correctness M={M} K={K} N={N}: max err={err:.6f}")
    assert err < 0.01

print("\n=== 性能（us） ===")
cases = [(8, 1024, 4096, "gate_up M=8"), (256, 1024, 4096, "gate_up M=256"),
         (8, 1024, 151936, "lm_head M=8"), (256, 1024, 151936, "lm_head M=256"),
         (8, 1024, 3072, "qkv M=8"), (8, 4096, 1024, "down M=8")]
for M, K, N, name in cases:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    v, idx = pack_2_4(w)
    t_dense = bench(lambda: F.linear(x, w))
    print(f"[{name}] cuBLAS {t_dense * 1e3:8.1f}us")
    for bm, bn, bk, warps, stages in [(16, 128, 128, 4, 2), (16, 128, 256, 4, 2),
                                      (32, 128, 128, 4, 2), (16, 256, 128, 8, 2),
                                      (64, 128, 128, 4, 2), (16, 128, 128, 4, 3),
                                      (32, 256, 128, 8, 2), (64, 256, 128, 8, 2)]:
        try:
            t = bench(lambda: run_sparse(x, v, idx, bm, bn, bk, warps, stages))
            print(f"    s24 bm={bm} bn={bn} bk={bk} w={warps} s={stages}: {t * 1e3:8.1f}us  "
                  f"{t_dense / t:.2f}x")
        except Exception as e:
            print(f"    s24 bm={bm} bn={bn} bk={bk} w={warps} s={stages}: FAIL {type(e).__name__}: {str(e)[:80]}")
