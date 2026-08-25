"""手写 SMEM-tiled FP16 GEMM 的开发与调优（阶段 1：内核与性能工程）。

路线：naive → num_stages 软件流水（多缓冲）→ GROUP_M L2 swizzle → tile/线程网格搜索。
指标：实测 TFLOPS vs cuBLAS（本机 ~48 TFLOPS）、n_regs/n_spills（Triton CompiledKernel）、
SMEM 用量（理论，含 stages 多缓冲）。
形状：大 4096³（计算受限区，测 TC 效率）与 decode 形态 M=64/K=4096/N=4096（带宽受限区）。

用法: python benchmarks/_matmul_dev.py
"""
import itertools

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_fp16(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """SMEM-tiled FP16 GEMM：C[M,N] = A[M,K] @ B[K,N]。

    - 每个 program 计算 BLOCK_M×BLOCK_N 的 C 块；K 循环内 tl.load 进 SMEM（由 Triton
      自动分配），tl.dot 用 TensorCore MMA；
    - num_stages>1 时 Triton 生成多缓冲软件流水（load k+1 与 dot k 重叠）；
    - GROUP_M swizzle：把同"行组"的 (pid_m, pid_n) 相邻调度 → 相邻 C 块共享 A 的行
      分片，提高 L2 命中（Triton 教程标准技巧）。
    """
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
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b_tile = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(a_ptr.dtype.element_ty),
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def launch(a, b, M, N, K, bm, bn, bk, group_m, warps, stages):
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    matmul_fp16[grid](a, b, c, M, N, K,
                      a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m,
                      num_warps=warps, num_stages=stages)
    return c


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


def kernel_stats(bm, bn, bk, group_m, warps, stages):
    """用 warmup 拿编译后内核的 n_regs/n_spills（不启动）。"""
    try:
        k = matmul_fp16.warmup(
            torch.empty(1, 1, device="cuda", dtype=torch.float16),
            torch.empty(1, 1, device="cuda", dtype=torch.float16),
            torch.empty(1, 1, device="cuda", dtype=torch.float16),
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m,
            num_warps=warps, num_stages=stages,
            grid=(1,))
        smem = 0
        try:
            smem = k.metadata.shared
        except Exception:
            smem = -1
        return k.n_regs, k.n_spills, smem
    except Exception as ex:
        return -1, -1, f"warmup err: {ex}"


def main():
    torch.manual_seed(0)
    dev = "cuda"
    print("=== 手写 SMEM-tiled FP16 GEMM vs cuBLAS ===\n")
    shapes = [(4096, 4096, 4096), (64, 4096, 4096), (256, 4096, 4096), (16384, 4096, 4096)]
    for M, N, K in shapes:
        a = torch.randn(M, K, device=dev, dtype=torch.float16)
        b = torch.randn(K, N, device=dev, dtype=torch.float16)
        ms = bench(lambda: torch.matmul(a, b))
        ref_tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
        print(f"--- M={M} N={N} K={K} | cuBLAS {ref_tflops:.1f} TFLOPS ---")
        best = None
        for bm, bn, bk, warps, stages in itertools.product(
                (64, 128), (128, 256), (64, 128), (4, 8), (2, 3, 4)):
            if bm * bk * 2 + bk * bn * 2 > 95000:  # SMEM 预算（100KB，留余量）
                continue
            try:
                ms2 = bench(lambda: launch(a, b, M, N, K, bm, bn, bk, 8, warps, stages),
                            iters=10, warmup=3)
            except Exception:
                continue
            tf = 2 * M * N * K / (ms2 * 1e-3) / 1e12
            if best is None or tf > best[0]:
                best = (tf, bm, bn, bk, warps, stages)
        tf, bm, bn, bk, warps, stages = best
        nr, ns, smem = kernel_stats(bm, bn, bk, 8, warps, stages)
        print(f"  最优手写: {tf:.1f} TFLOPS = {100*tf/ref_tflops:.1f}% cuBLAS  "
              f"(BM{bm} BN{bn} BK{bk} w{warps} s{stages} | regs={nr} spills={ns} smem={smem})")
        # 当前配置对比（引擎 int4/fp8 用的小 M 形态类似）
        del a, b
        torch.cuda.empty_cache()

    # ---- 逐配置明细（8192³ 太大，用 4096³ 看 n_regs/spills 与性能的关系） ----
    print("\n=== 4096³ 逐配置明细（看寄存器压力/溢出的影响） ===")
    M = N = K = 4096
    a = torch.randn(M, K, device=dev, dtype=torch.float16)
    b = torch.randn(K, N, device=dev, dtype=torch.float16)
    ms = bench(lambda: torch.matmul(a, b), iters=10)
    ref = 2 * M * N * K / (ms * 1e-3) / 1e12
    rows = []
    for bm, bn, bk, warps, stages in itertools.product(
            (64, 128), (128, 256), (64, 128), (4, 8), (2, 3, 4)):
        if bm * bk * 2 + bk * bn * 2 > 95000:
            continue
        try:
            ms2 = bench(lambda: launch(a, b, M, N, K, bm, bn, bk, 8, warps, stages),
                        iters=10, warmup=3)
        except Exception:
            continue
        tf = 2 * M * N * K / (ms2 * 1e-3) / 1e12
        rows.append((tf, bm, bn, bk, warps, stages))
    rows.sort(reverse=True)
    for tf, bm, bn, bk, warps, stages in rows[:12]:
        print(f"  {tf:5.1f} TFLOPS ({100*tf/ref:4.1f}% cuBLAS)  BM{bm} BN{bn} BK{bk} "
              f"w{warps} s{stages}")
    print(f"\n（cuBLAS 基准 = {ref:.1f} TFLOPS；最优配置的寄存器/占用分析见下）")

    # ---- 最优配置的寄存器 / SMEM / occupancy 分析（cuobjdump + 理论） ----
    print("\n=== 最优配置的资源分析（BM64 BN128 BK128 w4 s2 vs 其它） ===")
    import os, re, subprocess, tempfile
    for bm, bn, bk, warps, stages in ((64, 128, 128, 4, 2), (64, 128, 128, 8, 2),
                                      (128, 128, 64, 4, 2), (128, 128, 128, 8, 3)):
        try:
            kobj = matmul_fp16.warmup(
                a, b, torch.empty(M, N, device=dev, dtype=torch.float16),
                M, N, K, K, 1, N, 1, N, 1,
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
                num_warps=warps, num_stages=stages, grid=(1,))
            smem_meta = getattr(kobj.metadata, "shared", -1)
            cubin = kobj.asm["cubin"]
            with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
                f.write(cubin)
                path = f.name
            try:
                out = subprocess.run(["cuobjdump", "--dump-resource-usage", path],
                                     capture_output=True, text=True, timeout=30)
                m = re.search(r"REG:(\d+)", out.stdout)
                regs = int(m.group(1)) if m else -1
            finally:
                os.unlink(path)
            # 理论占用：regs/thread → 每 SM 能放的 block 数
            threads = warps * 32
            regs_per_block = regs * threads if regs > 0 else -1
            blocks_by_regs = (65536 // regs_per_block) if regs_per_block > 0 else -1
            blocks_by_smem = (100000 // smem_meta) if smem_meta > 0 else -1
            occ = min(blocks_by_regs, blocks_by_smem) if blocks_by_regs > 0 and blocks_by_smem > 0 else -1
            print(f"  BM{bm} BN{bn} BK{bk} w{warps} s{stages}: regs/thread={regs} "
                  f"SMEM/block={smem_meta}B 理论占用 blocks/SM = min(regs:{blocks_by_regs}, "
                  f"smem:{blocks_by_smem}) = {occ} ({100*occ*threads/1536:.0f}% 线程占用)")
        except Exception as e:
            print(f"  BM{bm} BN{bn} BK{bk} w{warps} s{stages}: err {e}")

    # ---- GROUP_M swizzle 消融（L2 优化证据） ----
    print("\n=== GROUP_M swizzle 消融（L2 命中优化，4096³ BM64 BN128 BK128 w4 s2） ===")
    for gm in (1, 2, 8, 32):
        ms2 = bench(lambda: launch(a, b, M, N, K, 64, 128, 128, gm, 4, 2), iters=20)
        tf = 2 * M * N * K / (ms2 * 1e-3) / 1e12
        print(f"  GROUP_M={gm:<3} {tf:5.1f} TFLOPS")

    # ---- stages 流水消融（SMEM 多缓冲 vs 容量） ----
    print("\n=== num_stages 流水消融（4096³ BM64 BN128 BK128 w4） ===")
    for st in (1, 2, 3, 4):
        try:
            ms2 = bench(lambda: launch(a, b, M, N, K, 64, 128, 128, 8, 4, st), iters=20)
            tf = 2 * M * N * K / (ms2 * 1e-3) / 1e12
            print(f"  stages={st} {tf:5.1f} TFLOPS")
        except Exception as e:
            print(f"  stages={st}: err {type(e).__name__}")


if __name__ == "__main__":
    main()
