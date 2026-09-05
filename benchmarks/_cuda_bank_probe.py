"""bank-conflict 消融：BsT 行距 pad(34) vs 无 pad(32) 的吞吐对比。

BsT[n][k] 转置布局的写路径（B 加载散写）在无 pad 时每 2 个 n 撞一次 bank
（行距 32 half → 行地址 = n*16 word → bank 只有 2 个值交替）；行距 34 时
17n mod 32 遍历 16 个不同 bank。本脚本验证该理论差是否可测。

用法: python benchmarks/_cuda_bank_probe.py
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
    print("=== BsT bank-conflict 消融：pad(BS=34) vs 无 pad(BS=32) ===\n")

    for M, N, K in [(128, 128, 64), (256, 256, 128), (4096, 4096, 4096),
                    (16384, 4096, 4096), (64, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        cp = ext.gemm_v2b_pad(a, b)
        cn = ext.gemm_v2b_nopad(a, b)
        ref = (a.double() @ b.double())
        rel_p = (cp.double() - ref).abs().max().item() / ref.abs().max().item()
        rel_n = (cn.double() - ref).abs().max().item() / ref.abs().max().item()
        t_p = bench(lambda: ext.gemm_v2b_pad(a, b))
        t_n = bench(lambda: ext.gemm_v2b_nopad(a, b))
        fl = 2.0 * M * N * K / 1e12
        print(f"M={M} N={N} K={K}: pad   {fl / (t_p * 1e-3):6.1f} TFLOPS (rel {rel_p:.1e}) | "
              f"nopad {fl / (t_n * 1e-3):6.1f} TFLOPS (rel {rel_n:.1e}) | "
              f"pad/nopad = {t_n / t_p:.3f}")


if __name__ == "__main__":
    main()
