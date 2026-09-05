"""Triton 真段式 grouped GEMM（MoE 专家批量）开发脚本。

目标：offsets/counts 驱动的段式 kernel（无 padding），替代 padded-bmm。
与 padded-bmm 对比找赢/亏边界（段长 vs BM）；验证正确性（vs bmm / reference）。

用法: python benchmarks/_moe_triton_segment.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton
import triton.language as tl


@triton.jit
def moe_segment_mm_kernel(
    a_ptr, w_ptr, y_ptr, offs_ptr, counts_ptr,
    R, K, O, E,
    stride_ak,                 # a 行步长（连续 → 1，但保留通用）
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """段式 grouped GEMM：y[段e行, O] = a[段e行, K] @ w[e, K, O]（w 为 [E, K, O] 行主序）。

    程序 (e, tm, tn)：处理专家 e 的第 tm 个 M 块（BM 行）与第 tn 个 N 块（BN 列）。
    段 e = [offs[e], offs[e]+counts[e])；空程序（tm·BM ≥ counts[e]）提前返回。
    """
    e = tl.program_id(0)
    tm = tl.program_id(1)
    tn = tl.program_id(2)
    n_e = tl.load(counts_ptr + e)
    if tm * BM >= n_e:
        return
    off = tl.load(offs_ptr + e)
    rows = off + tm * BM + tl.arange(0, BM)          # [BM]
    rmask = rows < off + n_e
    cols = tn * BN + tl.arange(0, BN)                # [BN]
    cmask = cols < O
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    w_base = w_ptr + e.to(tl.int64) * K * O
    for k in range(0, K, BK):
        kk = k + tl.arange(0, BK)
        kmask = kk < K
        a_tile = tl.load(a_ptr + rows[:, None] * stride_ak + kk[None, :],
                         mask=rmask[:, None] & kmask[None, :], other=0.0)  # [BM, BK]
        w_tile = tl.load(w_base + kk[:, None] * O + cols[None, :],
                         mask=kmask[:, None] & cmask[None, :], other=0.0)  # [BK, BN]
        acc += tl.dot(a_tile, w_tile)
    y_ptrs = y_ptr + rows[:, None] * O + cols[None, :]
    tl.store(y_ptrs, acc.to(a_ptr.dtype.element_ty),
             mask=rmask[:, None] & cmask[None, :])


def moe_segment_mm(a, w3d, offs, counts, BM=64, BN=128, BK=64, num_warps=4):
    """host 封装。a [R, K]（段连续）；w3d [E, K, O]；y [R, O]；offs/counts [E] int。"""
    R, K = a.shape
    E, _, O = w3d.shape
    y = torch.empty(R, O, device=a.device, dtype=a.dtype)
    max_n = int(counts.max())
    nt_m = (max_n + BM - 1) // BM
    nt_n = (O + BN - 1) // BN
    grid = (E, nt_m, nt_n)
    moe_segment_mm_kernel[grid](
        a, w3d, y, offs, counts, R, K, O, E, a.stride(0),
        BM=BM, BN=BN, BK=BK, num_warps=num_warps)
    return y


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print("=== Triton 段式 grouped GEMM ===\n")

    # 1) 正确性：随机专家分配（段长不均），对照 torch 分段 bmm 参考
    for (R, K, O, E, tag) in [(256, 512, 1536, 8, "E8"),
                              (4096, 512, 1536, 8, "E8-big"),
                              (96, 512, 1536, 8, "decode-like"),
                              (4096, 512, 768, 128, "E128")]:
        a = torch.randn(R, K, device=dev, dtype=torch.float16)
        w = torch.randn(E, K, O, device=dev, dtype=torch.float16) * 0.05
        # 随机专家标签 → counts/offs
        if tag == "decode-like":
            labels = torch.randint(0, E, (R,), device=dev)
        else:
            labels = torch.randint(0, E, (R,), device=dev)
        counts = torch.bincount(labels, minlength=E)
        offs = torch.cumsum(counts, 0) - counts
        order = torch.argsort(labels, stable=True)
        a_sorted = a[order]
        y = moe_segment_mm(a_sorted, w, offs.to(dev), counts.to(dev))
        # 参考：每段 bmm（显式循环，E 小）
        y_ref = torch.empty_like(y)
        for e in range(E):
            s, n = int(offs[e]), int(counts[e])
            if n:
                y_ref[s:s + n] = a_sorted[s:s + n] @ w[e]
        rel = (y.float() - y_ref.float()).norm() / y_ref.float().norm()
        print(f"correct {tag}: R={R} K={K} O={O} E={E} "
              f"counts={counts[:6].tolist()}... rel_norm_err={rel:.3e} "
              f"({'PASS' if rel < 1e-3 else 'FAIL'})")

    # 2) 基准：triton 段式 vs padded-bmm（同输入）
    print("\n=== triton 段式 vs padded bmm（H512/I768 专家语义: K=512, O=1536, E=8, k=2）===")
    K, O, E = 512, 1536, 8

    def bench(fn, iters=50, warmup=10, reps=5):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e2 = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e2.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e2) / iters)
        return statistics.median(ts)

    for R in [24, 96, 512, 2048, 8192]:
        labels = torch.randint(0, E, (R,), device=dev)
        counts = torch.bincount(labels, minlength=E)
        offs = torch.cumsum(counts, 0) - counts
        order = torch.argsort(labels, stable=True)
        a = torch.randn(R, K, device=dev, dtype=torch.float16)
        a_s = a[order]
        w = torch.randn(E, K, O, device=dev, dtype=torch.float16) * 0.05
        max_n = int(counts.max())
        # padded bmm
        dst = torch.zeros(E, max_n, K, device=dev, dtype=torch.float16)
        tt = bench(lambda: moe_segment_mm(a_s, w, offs.to(dev), counts.to(dev)))
        time.sleep(0.3)
        tb = bench(lambda: torch.bmm(dst[:, :max_n], w))
        print(f"R={R:5d} (max_n={max_n:4d}): triton {tt:7.3f} ms | "
              f"padded-bmm {tb:7.3f} ms | triton/bmm {tt / tb:.2f}x")

    # 真段式理论价值场景：大 E + prefill 大 R（每段长段，padding 相对小）
    print("\n=== 大 E 大 R（真段式价值场景）===")
    for (E2, R2, K2, O2) in [(128, 32768, 512, 1536), (128, 4096, 512, 1536),
                             (8, 32768, 512, 1536)]:
        labels = torch.randint(0, E2, (R2,), device=dev)
        counts = torch.bincount(labels, minlength=E2)
        offs = torch.cumsum(counts, 0) - counts
        a = torch.randn(R2, K2, device=dev, dtype=torch.float16)
        w = torch.randn(E2, K2, O2, device=dev, dtype=torch.float16) * 0.05
        max_n = int(counts.max())
        dst = torch.zeros(E2, max_n, K2, device=dev, dtype=torch.float16)
        tt = bench(lambda: moe_segment_mm(a, w, offs.to(dev), counts.to(dev)))
        time.sleep(0.3)
        tb = bench(lambda: torch.bmm(dst[:, :max_n], w))
        pad_waste = E2 * max_n / R2
        print(f"E={E2} R={R2:6d} max_n={max_n:5d} (padding {pad_waste:.2f}x): "
              f"triton {tt:7.3f} | bmm {tb:7.3f} ms | triton/bmm {tt / tb:.2f}x")


if __name__ == "__main__":
    main()
