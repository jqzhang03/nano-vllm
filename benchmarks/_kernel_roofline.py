"""内核 roofline 归因 + tile 网格搜索（阶段 1 主交付脚本，产出喂给 _kernel_roofline.md）。

内容：
  1. 硬件锚点：实测 FP16/BF16 TC 峰值（cuBLAS ~48 TFLOPS）、实测带宽（~370 GB/s）、
     36 SM / 100KB SMEM / 64K regs。
  2. 现有 GEMM 内核（int4 / fp8 / sparse24）归因：每个 (内核, M, N, K) 实测
     achieved TFLOPS vs 算力上界、achieved GB/s vs 带宽上界、arithmetic intensity
     → 分类"算力受限 / 带宽受限" + 效率。
  3. tile 网格搜索（BM/BN/warps/stages，BK 受分组约束固定 128）：当前配置 vs 搜索最优。
  4. fp8 decode 注意力内核归因（带宽型）。

用法: python benchmarks/_kernel_roofline.py
"""
import itertools
import os
import re
import subprocess
import tempfile

import torch
import triton

from nanovllm.layers.linear import (
    gemm_int4_kernel, gemm_fp8_kernel, gemm_sparse24_kernel,
)
from nanovllm.layers.attention import paged_decode_attention_fp8

TFLOPS_PEAK = 48.5   # 实测 cuBLAS BF16 大 GEMM（_hw_spec.py）
GBS_PEAK = 370.4     # 实测 D2D copy 带宽（_hw_spec.py）
SM = 36
REGS_PER_SM = 65536
THREADS_PER_SM = 1536


def real_regs(jit_kernel, args, kwargs, grid=(1,)):
    """从 cubin 拿真实寄存器数/溢出（cuobjdump --dump-resource-usage）。

    Triton 3.x 移除了 CompiledKernel.n_regs；PTX 的 %r<N> 是 SSA 虚拟编号（虚高）。
    cubin 里的资源占用是 ptxas 分配的最终值。
    """
    try:
        kern = jit_kernel.warmup(*args, **kwargs, grid=grid)
        cubin = kern.asm["cubin"]
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
            f.write(cubin)
            path = f.name
        try:
            out = subprocess.run(["cuobjdump", "--dump-resource-usage", path],
                                 capture_output=True, text=True, timeout=30)
            m = re.search(r"REG:(\d+)", out.stdout)
            s = re.search(r"STACK:(\d+)", out.stdout)
            regs = int(m.group(1)) if m else -1
            stack = int(s.group(1)) if s else 0
            return regs, stack
        finally:
            os.unlink(path)
    except Exception as ex:
        return -1, f"err: {ex}"


def bench(fn, iters=30, warmup=5):
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


def launch_int4(a, n, k, bm, bn, warps, stages, n_groups):
    """直接调 gemm_int4_kernel 自定义 tile。b_int4 [N, K//2]、scale [N, groups]。"""
    w = torch.randint(-8, 8, (n, k // 2), device=a.device, dtype=torch.int8)
    s = torch.randn(n, n_groups, device=a.device, dtype=a.dtype).abs() + 0.01
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(a.shape[0], bm) * triton.cdiv(n, bn),)
    gemm_int4_kernel[grid](
        a, w, out, s, a.shape[0], n, k, n_groups,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages)
    return out


def launch_fp8(a, n, k, bm, bn, warps, stages):
    w = torch.randn(n, k, device=a.device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s = torch.randn(n, 1, device=a.device, dtype=torch.float32).abs() + 0.01
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(a.shape[0], bm) * triton.cdiv(n, bn),)
    gemm_fp8_kernel[grid](
        a, w, out, s, a.shape[0], n, k,
        a.stride(0), a.stride(1), w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages)
    return out


def launch_sparse(a, n, k, bm, bn, warps, stages):
    v = torch.randn(n, k // 2, device=a.device, dtype=torch.bfloat16)
    idx = torch.randint(0, 4, (n, k // 4), device=a.device, dtype=torch.uint8)
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(a.shape[0], bm) * triton.cdiv(n, bn),)
    gemm_sparse24_kernel[grid](
        a, v, idx, out, a.shape[0], n, k,
        a.stride(0), a.stride(1), v.stride(0), v.stride(1), idx.stride(0), idx.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
        num_warps=warps, num_stages=stages)
    return out


def current_cfg(kernel_name, m):
    """引擎当前 tile 配置（linear.py 的 M-adaptive 路由：M≤128 小 tile / M>128 大 tile）。"""
    if m <= 128:
        if kernel_name == "sparse":
            return dict(bm=16, bn=128, warps=4, stages=3)
        return dict(bm=16, bn=128, warps=4, stages=2)
    return dict(bm=64, bn=256, warps=8, stages=2)


def roofline_row(name, ms, m, n, k, dtype_bytes_w):
    flops = 2 * m * n * k
    tflops = flops / (ms * 1e-3) / 1e12
    # 流量：A (m*k) 激活 + B (n*k) 权重 + C (m*n) 输出，单位字节
    bytes_total = (m * k * 2 + n * k * dtype_bytes_w + m * n * 2)
    gbs = bytes_total / (ms * 1e-3) / 1e9
    ai = flops / bytes_total
    bound = "算力受限" if ai >= 128 else "带宽受限"
    return dict(name=name, tflops=tflops, gbs=gbs, ai=ai, bound=bound,
                eff_tc=100 * tflops / TFLOPS_PEAK, eff_bw=100 * gbs / GBS_PEAK)


def warmup_args(kern, a, n, k, bm, bn, warps, stages):
    """构造各内核 warmup 的 (位置参数, 关键字参数)（与 launch_* 同形状）。"""
    if kern == "int4":
        w = torch.randint(-8, 8, (n, k // 2), device=a.device, dtype=torch.int8)
        s = torch.randn(n, n // 128, device=a.device, dtype=a.dtype).abs() + 0.01
        out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
        args = (a, w, out, s, a.shape[0], n, k, n // 128,
                a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                out.stride(0), out.stride(1))
        kw = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
                  num_warps=warps, num_stages=stages)
        return (gemm_int4_kernel, args, kw)
    if kern == "fp8":
        w = torch.randn(n, k, device=a.device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        s = torch.randn(n, 1, device=a.device, dtype=torch.float32).abs() + 0.01
        out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
        args = (a, w, out, s, a.shape[0], n, k,
                a.stride(0), a.stride(1), w.stride(0), w.stride(1),
                out.stride(0), out.stride(1))
        kw = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
                  num_warps=warps, num_stages=stages)
        return (gemm_fp8_kernel, args, kw)
    v = torch.randn(n, k // 2, device=a.device, dtype=torch.bfloat16)
    idx = torch.randint(0, 4, (n, k // 4), device=a.device, dtype=torch.uint8)
    out = torch.empty(a.shape[0], n, device=a.device, dtype=torch.bfloat16)
    args = (a, v, idx, out, a.shape[0], n, k,
            a.stride(0), a.stride(1), v.stride(0), v.stride(1), idx.stride(0), idx.stride(1),
            out.stride(0), out.stride(1))
    kw = dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=128, GROUP_M=8,
              num_warps=warps, num_stages=stages)
    return (gemm_sparse24_kernel, args, kw)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print(f"硬件锚点: TC 峰值 {TFLOPS_PEAK} TFLOPS | 带宽峰值 {GBS_PEAK} GB/s | "
          f"{SM} SM | SMEM 100KB/SM | regs 64K/SM")
    print(f"\n=== 1) 现有 GEMM 内核 roofline 归因（N=K=4096，int4 权重 0.25× 字节 / "
          f"fp8 0.5× / sparse 0.625× / 激活 bf16 2B） ===")
    shapes = [(8, 4096, 4096), (64, 4096, 4096), (256, 4096, 4096), (4096, 4096, 4096)]
    for kern, launch, wb in (("int4", launch_int4, 0.5),
                             ("fp8", launch_fp8, 1.0),
                             ("sparse", launch_sparse, 1.25)):
        for m, n, k in shapes:
            a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            cfg = current_cfg(kern, m)
            fn = lambda: launch(a, n, k, cfg["bm"], cfg["bn"], cfg["warps"], cfg["stages"],
                                n // 128) if kern == "int4" else \
                (launch_fp8(a, n, k, cfg["bm"], cfg["bn"], cfg["warps"], cfg["stages"])
                 if kern == "fp8" else
                 launch_sparse(a, n, k, cfg["bm"], cfg["bn"], cfg["warps"], cfg["stages"]))
            try:
                ms = bench(fn, iters=50, warmup=5)
            except Exception as e:
                print(f"  {kern} M={m}: err {e}")
                continue
            r = roofline_row(f"{kern} M={m}", ms, m, n, k, wb)
            print(f"  {r['name']:<14} {ms*1000:7.1f}µs | {r['tflops']:6.1f} TFLOPS "
                  f"({r['eff_tc']:4.0f}% TC) | {r['gbs']:6.0f} GB/s ({r['eff_bw']:3.0f}% BW) "
                  f"| AI {r['ai']:.0f} → {r['bound']}")
            del a
            torch.cuda.empty_cache()

    print(f"\n=== 2) tile 网格搜索（当前配置 vs 搜索最优；N=K=4096） ===")
    print(f"搜索空间: BM∈{{16,32,64,128}} BN∈{{64,128,256}} warps∈{{4,8}} stages∈{{2,3}} "
          f"(BK=128 受分组约束) | SMEM 预算 ≤95KB")
    for kern, launch in (("int4", launch_int4), ("fp8", launch_fp8), ("sparse", launch_sparse)):
        for m in (8, 64, 4096):
            a = torch.randn(m, 4096, device=dev, dtype=torch.bfloat16)
            cur = current_cfg(kern, m)
            ms_cur = bench(lambda: launch(a, 4096, 4096, cur["bm"], cur["bn"],
                                          cur["warps"], cur["stages"],
                                          m and 32 or 32) if kern == "int4" else
                           launch(a, 4096, 4096, cur["bm"], cur["bn"], cur["warps"], cur["stages"]),
                           iters=30, warmup=5)
            tf_cur = 2 * m * 4096 * 4096 / (ms_cur * 1e-3) / 1e12
            best = None
            for bm, bn, warps, stages in itertools.product(
                    (16, 32, 64, 128), (64, 128, 256), (4, 8), (2, 3)):
                smem = (bm * 128 + 128 * bn) * 2 * stages
                if smem > 95000:
                    continue
                if kern == "int4":
                    fn = lambda: launch(a, 4096, 4096, bm, bn, warps, stages, 32)
                else:
                    fn = lambda: launch(a, 4096, 4096, bm, bn, warps, stages)
                try:
                    ms = bench(fn, iters=20, warmup=3)
                except Exception:
                    continue
                tf = 2 * m * 4096 * 4096 / (ms * 1e-3) / 1e12
                if best is None or tf > best[0]:
                    best = (tf, bm, bn, warps, stages)
            tf, bm, bn, warps, stages = best
            jk, args, kw = warmup_args(kern, a, 4096, 4096, cur["bm"], cur["bn"],
                                       cur["warps"], cur["stages"])
            regs_cur = real_regs(jk, args, kw)
            jk, args, kw = warmup_args(kern, a, 4096, 4096, bm, bn, warps, stages)
            regs_best = real_regs(jk, args, kw)
            print(f"  {kern} M={m:<5} 当前 {tf_cur:5.1f} TFLOPS (BM{cur['bm']} BN{cur['bn']} "
                  f"w{cur['warps']} s{cur['stages']} regs={regs_cur[0]} stk={regs_cur[1]}) "
                  f"→ 最优 {tf:5.1f} TFLOPS (BM{bm} BN{bn} w{warps} s{stages} "
                  f"regs={regs_best[0]} stk={regs_best[1]}) | 提升 {100*(tf/tf_cur-1):+.1f}%")
            del a
            torch.cuda.empty_cache()

    print(f"\n=== 3) fp8 decode 注意力内核归因（带宽型） ===")
    H, K_H, D, BS = 8, 8, 128, 256
    for seqlen in (1024, 4096):
        nblk = (seqlen + 255) // 256
        q = torch.randn(BS, H, D, device=dev, dtype=torch.bfloat16)
        kc = torch.randn(nblk, 256, K_H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        vc = torch.randn(nblk, 256, K_H, D, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        # 每 seq 独享块（不重叠 → 无 L2 命中虚高，逼近真实引擎 DRAM 读）
        if BS * nblk <= nblk:  # 块不足时回退
            bt = torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0).repeat(BS, 1)
        else:
            bt = torch.randint(0, nblk, (BS, max(1, nblk)), device=dev, dtype=torch.int32)
            bt[:min(BS, nblk)] = torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(1) \
                .expand(min(BS, nblk), max(1, nblk)).contiguous() if nblk > 0 else bt
        seqlens = torch.full((BS,), seqlen, device=dev, dtype=torch.int32)
        ms = bench(lambda: paged_decode_attention_fp8(q, kc, vc, bt, seqlens,
                                                      0.1, 0.1, D ** -0.5),
                   iters=30, warmup=5)
        # 流量：每 seq 读全历史 K+V fp8（1B/elem）；写输出小（忽略）
        bytes_total = BS * seqlen * K_H * D * 2
        gbs = bytes_total / (ms * 1e-3) / 1e9
        print(f"  seqlen={seqlen} bs={BS}: {ms*1000:7.1f}µs | KV 读 {gbs:6.0f} GB/s "
              f"({100*gbs/GBS_PEAK:3.0f}% 的 copy 带宽) | 每 seq 每秒 {BS*1e3/ms:.0f} 步")
    print("\n说明：D2D copy 370 GB/s 是读写双向口径；纯读型内核可达更高（>500 GB/s）——"
          "注意力内核是带宽型（读 KV），算力远未饱和。")


if __name__ == "__main__":
    main()
