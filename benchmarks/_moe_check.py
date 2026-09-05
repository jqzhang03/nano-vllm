"""MoE 层对照检查：forward（gather/index_add 路径）vs reference（全行掩码路径）。

期望：数学同构 + 同求和顺序（e 升序、x dtype 累加）→ 位级一致或仅 GEMM
K-归约尾差（阈值 1e-4 相对；记账错会放大到 1e-2 级）。
覆盖：E ∈ {2, 8, 64}、k ∈ {1, 2, 4}、norm_topk_prob 开/关、T 边界（1/非 k 倍数/大批）、
0-token 专家出现（E > T×k 保证存在空专家）、全零输入（router 均匀）。

用法: python benchmarks/_moe_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.moe import MoE


def check(hidden, inter, E, k, T, norm, seed):
    torch.manual_seed(seed)
    moe = MoE(hidden, inter, num_experts=E, top_k=k, norm_topk_prob=norm)
    # 项目线性层权重是 torch.empty（依赖 loader 填充）→ 测试需自行初始化
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    moe = moe.cuda().half()
    x = torch.randn(T, hidden, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        y_fwd = moe(x)
        y_ref = moe.reference(x)
    diff = (y_fwd.float() - y_ref.float()).abs().max().item()
    scale = max(1.0, y_ref.float().abs().max().item())
    ok = diff <= 1e-4 * scale
    with torch.no_grad():
        _, idx = moe._route(x)
    used = idx.reshape(-1).unique().numel()
    tag = "PASS" if ok else "FAIL"
    print(f"E={E:3d} k={k} T={T:4d} norm={int(norm)}: max_diff={diff:.3e} "
          f"({tag}) experts_used={used}/{E}")
    return ok


def main():
    print("=== MoE 层对照：forward vs reference ===\n")
    ok = True
    cases = [
        # (hidden, inter, E, k, T, norm)
        (32, 64, 2, 1, 1, 0),     # 边界：T=1, k=1, E=2
        (32, 64, 2, 2, 5, 0),     # k=E（全专家）
        (64, 128, 8, 2, 7, 0),    # T 非 k 倍数 → 空专家
        (64, 128, 8, 2, 100, 0),
        (64, 128, 8, 2, 100, 1),  # norm_topk_prob
        (128, 512, 64, 4, 3, 0),  # 大 E 稀疏（3×4=12 < 64 → 52 空专家）
        (128, 512, 64, 4, 1000, 1),
        (32, 64, 4, 2, 128, 0),
    ]
    for c in cases:
        ok &= check(*c, seed=0)
    # 全零输入（无 bias → router logits 全 0 → softmax 均匀 → topk 前 k 个）
    moe = MoE(16, 32, num_experts=4, top_k=1).cuda().half()
    with torch.no_grad():
        for p in moe.parameters():
            p.uniform_(-0.05, 0.05)
    x = torch.zeros(32, 16, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        d = (moe(x).float() - moe.reference(x).float()).abs().max().item()
    ok &= d == 0.0
    print(f"全零输入（router 均匀）: max_diff={d:.3e} ({'PASS' if d == 0 else 'FAIL'})")

    # Triton 真段式后端（segment_backend=True → forward 走 moe_segment）
    print("\n--- segment_backend（Triton 段式）---")
    for (E, k, T) in [(8, 2, 100), (64, 4, 3), (8, 2, 4096)]:
        moe = MoE(512, 768, num_experts=E, top_k=k,
                  segment_backend=True).cuda().half()
        with torch.no_grad():
            for p in moe.parameters():
                p.uniform_(-0.05, 0.05)
        x = torch.randn(T, 512, device="cuda", dtype=torch.float16)
        with torch.no_grad():
            y_seg = moe(x)                     # 走 triton 段式
            y_ref = moe.reference(x)
        diff = (y_seg.float() - y_ref.float()).abs().max().item()
        scale = max(1.0, y_ref.float().abs().max().item())
        ok2 = diff <= 1e-4 * scale
        ok &= ok2
        print(f"seg E={E:3d} k={k} T={T:5d}: max_diff={diff:.3e} "
              f"({'PASS' if ok2 else 'FAIL'})")
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
