"""关键数字高迭代复测（阶段方法论：异常值必须复测）。

复测项（均与文档声称一致才算过）：
- v1a/v2a/v2b 的 4096^3、16384^3、M=64 吞吐
- bank 消融 pad/nopad 比（文档 ~1.88x @ 4096^3）
- split-K S=4 @ M=64（文档 ~1.66x vs v2b）
- persistent @ 4096^3（文档 ~0.89x vs v2b，亏损）
iters=200 取中位数（5 次独立计时取中位，抗噪声）。

用法: python benchmarks/_cuda_final_verify.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cuda_common import load_ext

import torch


def bench_median(fn, iters=200, warmup=10, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) / iters)
    return statistics.median(times)


def tflops(ms, M, N, K):
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ext = load_ext("_cuda_gemm_dev_ext", os.path.join(here, "_cuda_gemm_dev.cu"))
    print("=== 高迭代复测（iters=200 中位数）===\n")
    results = {}

    for M, N, K, tag in [(4096, 4096, 4096, "4096^3"),
                         (16384, 4096, 4096, "16384x4096"),
                         (64, 4096, 4096, "M=64")]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        t = bench_median(lambda: torch.matmul(a, b))
        cub = tflops(t, M, N, K)
        row = f"M={M} N={N} K={K} ({tag}): cuBLAS {cub:6.1f}"
        for name, fn, iters in [("v1a", ext.gemm_fma_naive, 30),
                                ("v2a", ext.gemm_mma16x64, 200),
                                ("v2b", ext.gemm_v2b, 200)]:
            tm = bench_median(lambda: fn(a, b), iters=iters)
            row += f" | {name} {tflops(tm, M, N, K):6.1f} ({tm / t:5.1%})"
        print(row)
        results[tag] = row

    # bank 消融
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    tp = bench_median(lambda: ext.gemm_v2b_pad(a, b))
    tn = bench_median(lambda: ext.gemm_v2b_nopad(a, b))
    print(f"\nbank 消融 4096^3: pad {tflops(tp, 4096, 4096, 4096):.1f} | "
          f"nopad {tflops(tn, 4096, 4096, 4096):.1f} TFLOPS | "
          f"pad/nopad = {tn / tp:.3f}")

    # split-K @ M=64
    a = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    tv = bench_median(lambda: ext.gemm_v2b(a, b))
    ts = bench_median(lambda: ext.gemm_v2b_splitk(a, b, 4))
    print(f"splitK S=4 @ M=64: v2b {tflops(tv, 64, 4096, 4096):.1f} | "
          f"S4 {tflops(ts, 64, 4096, 4096):.1f} TFLOPS | vs v2b {tv / ts:.3f}x")

    # persistent @ 4096^3
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    tv = bench_median(lambda: ext.gemm_v2b(a, b))
    tp = bench_median(lambda: ext.gemm_v2b_persist(a, b, 108))
    print(f"persist108 @ 4096^3: v2b {tflops(tv, 4096, 4096, 4096):.1f} | "
          f"persist {tflops(tp, 4096, 4096, 4096):.1f} TFLOPS | vs v2b {tp / tv:.3f}x")
    print("\n复测完成")


if __name__ == "__main__":
    main()
