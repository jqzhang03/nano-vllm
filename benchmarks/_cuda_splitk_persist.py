"""split-K 与 persistent kernel 验证：正确性 + 小 M 大 K 吞吐对比。

split-K 目标场景：M 小 → C-block 数 < SM 数（M=64, N=4096 → 32 blocks < 36 SM），
K 切成 S 段增加并行度。persistent：固定 grid 循环 tile，测去 block 调度收益。

用法: python benchmarks/_cuda_splitk_persist.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cuda_common import load_ext

import torch


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ext = load_ext("_cuda_gemm_dev_ext", os.path.join(here, "_cuda_gemm_dev.cu"))
    torch.manual_seed(0)
    print("=== split-K / persistent kernel ===\n")

    # 1) 正确性
    for M, N, K in [(64, 128, 64), (256, 256, 128), (64, 512, 256),
                    (128, 256, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        ref = (a.double() @ b.double())
        for S in [1, 2, 4]:
            if K % (32 * S):
                continue
            c = ext.gemm_v2b_splitk(a, b, S)
            rel = (c.double() - ref).abs().max().item() / ref.abs().max().item()
            ok = "PASS" if rel < 1e-2 else "FAIL"
            print(f"splitk S={S} {M}x{N}x{K}: rel={rel:.2e} {ok}")
        c = ext.gemm_v2b_persist(a, b, 8)
        rel = (c.double() - ref).abs().max().item() / ref.abs().max().item()
        print(f"persist {M}x{N}x{K}: rel={rel:.2e} "
              f"({'PASS' if rel < 1e-2 else 'FAIL'})")
    print()

    # 2) 小 M 大 K：split-K 扫描（vs v2b 与 cuBLAS）
    for M, N, K in [(64, 4096, 4096), (256, 4096, 4096), (128, 8192, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        t_cub = bench(lambda: torch.matmul(a, b))
        t_v2b = bench(lambda: ext.gemm_v2b(a, b))
        fl = 2.0 * M * N * K / 1e12
        print(f"M={M} N={N} K={K}: cuBLAS {fl / (t_cub * 1e-3):6.1f} | "
              f"v2b {fl / (t_v2b * 1e-3):6.1f} TFLOPS")
        for S in [2, 4, 8]:
            if K % (32 * S):
                continue
            t_s = bench(lambda: ext.gemm_v2b_splitk(a, b, S))
            print(f"  splitk S={S}: {fl / (t_s * 1e-3):6.1f} TFLOPS "
                  f"(vs v2b {t_v2b / t_s:5.2f}x)")
        # persistent: 不同 grid 大小
        for nb in [36, 72, 108, 144]:
            t_p = bench(lambda: ext.gemm_v2b_persist(a, b, nb))
            print(f"  persist nb={nb}: {fl / (t_p * 1e-3):6.1f} TFLOPS "
                  f"(vs v2b {t_v2b / t_p:5.2f}x)")

    # 3) 大形状 sanity：persistent 在 4096^3（tiles 2048）
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t_v2b = bench(lambda: ext.gemm_v2b(a, b))
    t_p = bench(lambda: ext.gemm_v2b_persist(a, b, 108))
    fl = 2.0 * 4096 ** 3 / 1e12
    print(f"\n4096^3: v2b {fl / (t_v2b * 1e-3):.1f} | persist108 "
          f"{fl / (t_p * 1e-3):.1f} TFLOPS (vs v2b {t_v2b / t_p:.3f}x)")


if __name__ == "__main__":
    main()
