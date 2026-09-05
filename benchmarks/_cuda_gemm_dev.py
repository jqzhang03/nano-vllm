"""CUDA C 显式 SMEM fp16 GEMM 开发驱动：正确性 + 基准 + SASS 分析。

用法: python benchmarks/_cuda_gemm_dev.py
当前内核: gemm_fma_naive（v1a，教学地板）
对照: torch.matmul（cuBLAS fp16）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cuda_common import load_ext

import torch


def bench(fn, iters=20, warmup=5):
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


KERNELS = {
    "v1a_fma_naive": ("gemm_fma_naive", (16, 16, 16)),
    "v2a_mma16x64": ("gemm_mma16x64", (16, 64, 32)),
    "v2b_mma_8tile": ("gemm_v2b", (64, 128, 32)),
}


def check_correct(ext, name, fn, M, N, K, tol=1e-2):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = fn(a, b)
    ref = (a.double() @ b.double())
    err = (c.double() - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    ok = rel < tol
    print(f"  {name:14s} {M}x{N}x{K}: rel={rel:.3e} ({'PASS' if ok else 'FAIL'})")
    return ok


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ext = load_ext("_cuda_gemm_dev_ext", os.path.join(here, "_cuda_gemm_dev.cu"))
    print("=== CUDA C 显式 SMEM fp16 GEMM 开发 ===")

    # 1) 正确性：多形状覆盖（小/多块路径/大 K）
    ok = True
    shapes = [(16, 16, 16), (256, 256, 128), (512, 384, 320),
              (1024, 1024, 4096), (4096, 4096, 4096),
              (128, 128, 64), (384, 256, 96)]
    for name, (attr, div) in KERNELS.items():
        fn = getattr(ext, attr)
        for M, N, K in shapes:
            if M % div[0] or N % div[1] or K % div[2]:
                continue
            ok &= check_correct(ext, name, fn, M, N, K)
    print()

    # 2) 基准 vs cuBLAS
    for name, (attr, div) in KERNELS.items():
        fn = getattr(ext, attr)
        print(f"--- {name} ---")
        for M, N, K in [(4096, 4096, 4096), (64, 4096, 4096), (16384, 4096, 4096),
                        (256, 4096, 4096)]:
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)
            t_cub = bench(lambda: torch.matmul(a, b))
            t_my = bench(lambda: fn(a, b))
            fl = 2.0 * M * N * K / 1e12
            print(f"M={M} N={N} K={K}: cuBLAS {fl / (t_cub * 1e-3):7.1f} TFLOPS | "
                  f"{name} {fl / (t_my * 1e-3):7.1f} TFLOPS "
                  f"({t_cub / t_my:5.1%} of cuBLAS time)")
        print()

    # 3) SASS 分析（每个内核生成什么）
    print("=== SASS 分析 ===")
    os.system(f"python {os.path.join(here, '_sass_stats.py')} {ext.__file__}")
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
