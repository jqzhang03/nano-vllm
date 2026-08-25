"""采集本机 GPU 硬件规格 + 大 GEMM 实测峰值标定（roofline 归因的输入）。

输出：
  1. torch 设备属性（SM 数、每 SM 线程/SMEM/寄存器上限、L2、显存）
  2. 大矩阵 GEMM（cuBLAS）实测 TFLOPS——作为"实测可达算力峰值"（含 TensorCore 效率）
  3. 大内存拷贝实测带宽——作为"实测可达带宽峰值"
  4. fp16 GEMM 的 arithmetic intensity 参考线

用法: python benchmarks/_hw_spec.py
"""
import torch
import torch.nn.functional as F


def bench(fn, iters=20, warmup=5):
    """返回 fn 的平均毫秒（CUDA events，同步）。"""
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
    props = torch.cuda.get_device_properties(0)
    print("=== torch device properties ===")
    for k in ("name", "total_memory", "multi_processor_count", "max_threads_per_multi_processor",
              "max_threads_per_block", "shared_memory_per_block", "shared_memory_per_multiprocessor",
              "regs_per_multiprocessor", "regs_per_block", "l2_cache_size", "major", "minor"):
        print(f"  {k}: {getattr(props, k, 'n/a')}")
    print(f"  compute capability: {props.major}.{props.minor}")

    dev = "cuda"
    torch.manual_seed(0)

    # ---- 1) 实测带宽：大张量拷贝（读写各一次 → 2*N bytes / 时间） ----
    N = 256 * 1024 * 1024  # 1GB bf16 元素? 256M * 2B = 512MB
    a = torch.randn(N, device=dev, dtype=torch.bfloat16)
    b = torch.empty_like(a)
    ms = bench(lambda: b.copy_(a))
    bw = 2 * a.numel() * a.element_size() / (ms * 1e-3) / 1e9
    print(f"\n=== 实测带宽 ===")
    print(f"  D2D copy 512MB: {ms:.3f} ms -> {bw:.1f} GB/s（读写各 1 次计 2×N bytes）")

    # ---- 2) 实测 FP16 GEMM 峰值（cuBLAS，大 M 无带宽压力） ----
    print(f"\n=== 实测 FP16 GEMM 峰值（cuBLAS） ===")
    for M, Nn, K in ((4096, 4096, 4096), (8192, 8192, 8192), (16384, 8192, 8192)):
        x = torch.randn(M, K, device=dev, dtype=torch.float16)
        w = torch.randn(Nn, K, device=dev, dtype=torch.float16)
        ms = bench(lambda: F.linear(x, w))
        flops = 2 * M * Nn * K
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"  M{Nn}xK{K}xN{Nn}: {ms:.3f} ms -> {tflops:.1f} TFLOPS")
        del x, w
        torch.cuda.empty_cache()

    # ---- 3) BF16 GEMM 峰值（引擎实际 dtype） ----
    print(f"\n=== 实测 BF16 GEMM 峰值（cuBLAS） ===")
    M = Nn = K = 8192
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = torch.randn(Nn, K, device=dev, dtype=torch.bfloat16)
    ms = bench(lambda: F.linear(x, w))
    tflops = 2 * M * Nn * K / (ms * 1e-3) / 1e12
    print(f"  M=N=K=8192: {ms:.3f} ms -> {tflops:.1f} TFLOPS")

    # ---- 4) FP32 GEMM（非 TC 参考） ----
    x = torch.randn(M, K, device=dev)
    w = torch.randn(Nn, K, device=dev)
    ms = bench(lambda: F.linear(x, w))
    tflops = 2 * M * Nn * K / (ms * 1e-3) / 1e12
    print(f"  FP32 M=N=K=8192: {ms:.3f} ms -> {tflops:.1f} TFLOPS")

    # ---- 5) arithmetic intensity 参考 ----
    print(f"\n=== roofline 参考 ===")
    print(f"  实测带宽峰值: {bw:.1f} GB/s（BF16 D2D copy）")
    print(f"  实测 FP16 TC 峰值: ~{tflops:.0f} TFLOPS 量级（看上方大形状）")
    for ai in (32, 64, 128, 256, 512):
        print(f"  AI={ai} FLOP/byte 时带宽边界算力 = {bw * ai / 1000:.1f} TFLOPS"
              f"（若 < TC 峰值则此形态是带宽受限）")


if __name__ == "__main__":
    main()
