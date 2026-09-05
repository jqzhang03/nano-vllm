"""诊断：MoE forward 计时为何秒级——单次 wall/event 分布 vs 循环。

怀疑：bench_median 的 Event 计时被什么放大，或单 forward 本身慢（kernel/时钟）。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.moe import MoE


def main():
    H, I, E, k, T = 512, 768, 8, 2, 1024
    moe = MoE(H, I, num_experts=E, top_k=k).cuda().half()
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    x = torch.randn(T, H, device="cuda", dtype=torch.float16)

    # 单次 forward（含一切冷启动）
    for i in range(3):
        s = time.perf_counter()
        moe(x)
        torch.cuda.synchronize()
        print(f"single forward #{i}: wall {(time.perf_counter() - s) * 1e3:.2f} ms")

    # 预热后 20 次：wall 与 event
    for _ in range(10):
        moe(x)
    torch.cuda.synchronize()
    s = time.perf_counter()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20):
        moe(x)
    e1.record()
    torch.cuda.synchronize()
    print(f"20 forwards: wall {(time.perf_counter() - s) * 1e3:.2f} ms "
          f"| event {e0.elapsed_time(e1):.2f} ms")

    # 各算子单独计时（定位慢在哪个算子）
    import torch.nn.functional as F
    with torch.no_grad():
        logits = moe.gate(x)
    probs = logits.float().softmax(-1)
    top_vals, top_idx = probs.topk(k, -1)
    vals = top_vals.half()
    idx = top_idx.reshape(-1)
    w = vals.reshape(-1)
    tgt = torch.arange(T, device="cuda").unsqueeze(1).expand(T, k).reshape(-1)

    def t(name, fn):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        print(f"  {name:30s}: {(time.perf_counter() - s) * 1e3 / 100:.4f} ms/iter")

    t("router+topk", lambda: moe.gate(x).float().softmax(-1).topk(k, -1))
    t("mask+any", lambda: (idx == 0).any())
    t("gather", lambda: x.index_select(0, tgt[idx == 0]))
    out = torch.zeros_like(x)
    xs = x.index_select(0, tgt[idx == 0])
    t("expert ffn", lambda: moe.experts[0](xs))
    t("index_add", lambda: out.index_add_(0, tgt[idx == 0], xs))


if __name__ == "__main__":
    main()
