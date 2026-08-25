"""复测 tile 搜索的关键发现（高迭代 + 中位数，排除小 M 测量噪声）。

验证项：
  1. int4 M=8:  BM16/BN128/w4/s2（当前）vs BM16/BN64/w8/s3（搜索最优 +20%）
  2. int4 M=4096: BM64/BN256/w8/s2（当前）vs BM16/BN128/w4/s2（搜索最优 +21%）
  3. fp8 M=4096: 当前 vs 搜索最优（fp8 大 M 是否同样偏好小 tile）
  4. fp8 M=8: 当前 vs 搜索最优
另：打印每配置的 occupancy 理论（regs/thread → blocks/SM）。

用法: python benchmarks/_kernel_verify.py
"""
import statistics

import torch
import triton

from nanovllm.layers.linear import gemm_int4_kernel, gemm_fp8_kernel

REGS_PER_SM = 65536


def bench_median(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def launch_int4(a, n, k, bm, bn, warps, stages):
    w = torch.randint(-8, 8, (n, k // 2), device=a.device, dtype=torch.int8)
    sc = torch.randn(n, n // 128, device=a.device, dtype=a.dtype).abs() + 0.01
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(a.shape[0], bm) * triton.cdiv(n, bn),)
    gemm_int4_kernel[grid](
        a, w, out, sc, a.shape[0], n, k, n // 128,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages)
    return out


def launch_fp8(a, n, k, bm, bn, warps, stages):
    w = torch.randn(n, k, device=a.device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    sc = torch.randn(n, 1, device=a.device, dtype=torch.float32).abs() + 0.01
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(a.shape[0], bm) * triton.cdiv(n, bn),)
    gemm_fp8_kernel[grid](
        a, w, out, sc, a.shape[0], n, k,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages)
    return out


def occ(regs, threads):
    """理论 occupancy：regs/SM 与 threads/SM 的 min。"""
    b_by_reg = REGS_PER_SM // (regs * threads) if regs > 0 else 99
    b_by_thr = 1536 // threads
    return min(b_by_reg, b_by_thr)


def compare(name, launch, a, n, k, configs):
    print(f"--- {name} (M={a.shape[0]} N={n} K={k}) ---")
    results = []
    for bm, bn, warps, stages in configs:
        ms = bench_median(lambda: launch(a, n, k, bm, bn, warps, stages))
        tf = 2 * a.shape[0] * n * k / (ms * 1e-3) / 1e12
        results.append((ms, tf, bm, bn, warps, stages))
        print(f"  {ms*1000:8.1f}µs  {tf:6.1f} TFLOPS  BM{bm} BN{bn} w{warps} s{stages}")
    results.sort()
    best = results[0]
    print(f"  → 最优: BM{best[2]} BN{best[3]} w{best[4]} s{best[5]} "
          f"({best[1]:.1f} TFLOPS)；最差/最优 = {results[-1][0]/best[0]:.2f}x\n")
    return results


def main():
    torch.manual_seed(0)
    dev = "cuda"
    N = K = 4096
    for m, kern, cur, cand in (
            (8, "int4", (16, 128, 4, 2), (16, 64, 8, 3)),
            (4096, "int4", (64, 256, 8, 2), (16, 128, 4, 2)),
            (4096, "fp8", (64, 256, 8, 2), (16, 128, 4, 2)),
            (8, "fp8", (16, 128, 4, 2), (16, 64, 8, 3)),
            (64, "fp8", (16, 128, 4, 2), (16, 128, 4, 2))):
        a = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        launch = launch_int4 if kern == "int4" else launch_fp8
        # 当前 + 候选 + 候选邻近（排除局部极值）
        configs = [cur, cand, (16, 128, 4, 3), (32, 128, 4, 2), (64, 128, 4, 2),
                   (16, 128, 8, 2), (32, 64, 8, 3), (16, 256, 4, 2)]
        configs = list(dict.fromkeys(configs))
        compare(f"{kern} M={m}", launch, a, N, K, configs)
        del a
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
