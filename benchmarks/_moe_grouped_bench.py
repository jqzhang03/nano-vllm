"""grouped vs loop 后端基准：组织税是否消除 + 不均衡 padding 代价。

- T 扫描（k=2, 均衡路由）：loop 版时间应近似平坦（~3.5ms 固定组织税），
  grouped 版应随 T 增长（批量 GEMM 主导）→ 固定税消除的证据；
- 不均衡（强制 100% → e0）：grouped 的 padded 批量放大到 E·max_n 行
  （vs 需要 R=T·k）→ 时间应明显劣化（padding 浪费量化）；
- 引擎吞吐对比（toy decode）见 _moe_engine_bench.py（自动后端）。

用法: python benchmarks/_moe_grouped_bench.py
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.moe import MoE


def bench_median(fn, iters=50, warmup=10, reps=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters)
    return statistics.median(ts)


def settle():
    torch.cuda.synchronize()
    time.sleep(0.5)


def main():
    H, I, E, k = 512, 768, 8, 2
    moe = MoE(H, I, num_experts=E, top_k=k).cuda().half()
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    moe._ensure_grouped_weights()
    assert moe._grouped_ok, "grouped weights should be available (non-quant)"
    print("=== grouped vs loop：T 扫描（k=2 均衡，ms/forward）===")
    for T in [64, 256, 1024, 4096]:
        x = torch.randn(T, H, device="cuda", dtype=torch.float16)
        tg = bench_median(lambda: moe._forward_grouped(x))
        settle()
        tl = bench_median(lambda: moe._forward_loop(x))
        settle()
        print(f"  T={T:5d}: grouped {tg:8.3f} | loop {tl:8.3f} ms | loop/grouped {tl / tg:5.2f}x")

    # 不均衡：强制全部 → e0（gate 权重只让 e0 有分）
    print("\n=== 不均衡（100% → e0）：padded 批量放大代价 ===")
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(1.0)
    T = 1024
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)
    tg_imb = bench_median(lambda: moe._forward_grouped(x))
    settle()
    tl_imb = bench_median(lambda: moe._forward_loop(x))
    with torch.no_grad():
        moe.gate.weight.uniform_(-0.05, 0.05)
    xb = torch.randn(T, H, device="cuda", dtype=torch.float16)
    tg_bal = bench_median(lambda: moe._forward_grouped(xb))
    settle()
    # 理论放大：R = T*k，grouped 计算行 = E*max_n
    with torch.no_grad():
        vals, idx = moe._route(x)
    counts = torch.bincount(idx.reshape(-1), minlength=E)
    print(f"  路由分布 counts={counts.tolist()} max={int(counts.max())} "
          f"R={T * k} → 计算放大 {E * int(counts.max()) / (T * k):.1f}x（理论）")
    print(f"  T=1024: imbalanced grouped {tg_imb:7.3f} | imbalanced loop {tl_imb:7.3f} | "
          f"balanced grouped {tg_bal:7.3f} ms")


if __name__ == "__main__":
    main()
