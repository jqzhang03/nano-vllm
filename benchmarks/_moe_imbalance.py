"""MoE 负载分布观察：循环专家实现（串行逐专家）对负载不均是否敏感？

假设（待验证）：本实现专家**串行**处理（每层 E 次 gather+FFN+scatter，空专家跳过）
→ 总时间 ≈ Σ 专家时间 ≈ 总激活行数 × FFN 单行成本 / GEMM 效率 → 对"某专家收到
大部分 token"的分布**不敏感**（不同于并行分组实现的 max-专家瓶颈）。代价是每层
E 次 kernel 启动 + 小 GEMM 效率低。对比：同一 T 下 balanced（随机路由）vs
imbalanced（强制全部 token → expert 0，k=1）耗时。

用法: python benchmarks/_moe_imbalance.py
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.moe import MoE


def bench_median(fn, iters=30, warmup=5, reps=3):
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


def main():
    import time
    H, I, E, T = 512, 768, 8, 1024
    print("=== 循环专家实现的负载分布观察（单位 ms/iter）===\n")

    def settle():
        torch.cuda.synchronize()
        time.sleep(0.5)

    for k, tag in [(1, "k=1"), (2, "k=2"), (8, "k=8 (全专家=dense化)")]:
        moe = MoE(H, I, num_experts=E, top_k=k).cuda().half()
        with torch.no_grad():
            for p in moe.parameters():
                p.uniform_(-0.05, 0.05)
        x = torch.randn(T, H, device="cuda", dtype=torch.float16)
        tb = bench_median(lambda: moe(x))
        settle()
        with torch.no_grad():
            moe.gate.weight.zero_()
            moe.gate.weight[0].fill_(1.0)
        ti = bench_median(lambda: moe(x))
        settle()
        with torch.no_grad():
            for p in moe.gate.parameters():
                p.uniform_(-0.05, 0.05)
        with torch.no_grad():
            _, idxb = moe._route(x)
        cnt_b = torch.bincount(idxb.reshape(-1), minlength=E).tolist()
        print(f"{tag}: balanced {tb:7.3f} ms | imbalanced(100%→e0) {ti:7.3f} ms "
              f"| imbal/bal = {ti / tb:.3f} | bal dist {cnt_b}")

    # T 扫描（k=2）：验证总激活行线性
    moe = MoE(H, I, num_experts=E, top_k=2).cuda().half()
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    print("\nT 扫描（k=2, balanced, ms/iter）:")
    for T in [64, 256, 1024, 4096]:
        x = torch.randn(T, H, device="cuda", dtype=torch.float16)
        t = bench_median(lambda: moe(x))
        settle()
        print(f"  T={T:5d}: {t:8.3f} ms")

    # dense 对照：相同激活 FLOPs（I_dense = k × I_moe）——循环专家的组织税
    from nanovllm.layers.linear import ColumnParallelLinear, RowParallelLinear
    from torch import nn
    import torch.nn.functional as F

    class DenseMLP(nn.Module):
        def __init__(self, H, I_d):
            super().__init__()
            self.gate_proj = ColumnParallelLinear(H, I_d, bias=False)
            self.up_proj = ColumnParallelLinear(H, I_d, bias=False)
            self.down_proj = RowParallelLinear(I_d, H, bias=False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    moe8 = MoE(H, I, num_experts=8, top_k=2).cuda().half()
    dense = DenseMLP(H, 2 * I).cuda().half()   # 激活 FLOPs = MoE k=2
    with torch.no_grad():
        for p in moe8.parameters():
            p.uniform_(-0.05, 0.05)
        for p in dense.parameters():
            p.uniform_(-0.05, 0.05)
    print("\n同激活 FLOPs 对照（MoE E=8 k=2 vs dense inter=2×I, ms/iter）:")
    for T in [256, 1024, 4096]:
        x = torch.randn(T, H, device="cuda", dtype=torch.float16)
        tm = bench_median(lambda: moe8(x))
        settle()
        td = bench_median(lambda: dense(x))
        settle()
        print(f"  T={T:5d}: MoE {tm:7.3f} ms | dense {td:7.3f} ms "
              f"| MoE/dense = {tm / td:.2f}x（>1 = 循环专家的组织税）")


if __name__ == "__main__":
    main()
