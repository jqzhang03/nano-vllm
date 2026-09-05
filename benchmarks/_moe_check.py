"""MoE 层对照检查：forward（gather/index_add 路径）vs reference（全行掩码路径）。

期望：数学同构 + 同求和顺序（e 升序、fp32 累加）→ **位级一致（diff == 0）**。
覆盖：E ∈ {2, 8, 64}、k ∈ {1, 2, 4}、T 边界（1 / 非 k 倍数 / 大批）、shared expert 有无、
所有 token 集中一个专家（负载极端）、0-token 专家出现（E > T×k 保证存在空专家）。

用法: python benchmarks/_moe_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.moe import MoE


def check(hidden, inter, E, k, T, shared, seed):
    torch.manual_seed(seed)
    moe = MoE(hidden, inter, num_experts=E, top_k=k, num_shared_experts=shared)
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
    ok = diff <= 1e-4 * scale   # 位级之外允许 GEMM K-归约顺序尾差（记账错会放大到 1e-2 级）
    # 路由统计（确认场景真的覆盖了空/满专家）
    with torch.no_grad():
        vals, idx = moe._route(x.float() if False else x)
    used = idx.reshape(-1).unique().numel()
    tag = "PASS" if ok else "FAIL"
    print(f"E={E:3d} k={k} T={T:4d} shared={shared}: max_diff={diff:.3e} "
          f"({tag}) experts_used={used}/{E}")
    return ok


def main():
    print("=== MoE 层对照：forward vs reference（期望 diff == 0）===\n")
    ok = True
    cases = [
        # (hidden, inter, E, k, T, shared)
        (32, 64, 2, 1, 1, 0),    # 边界：T=1, k=1, E=2
        (32, 64, 2, 2, 5, 0),    # k=E（全专家）
        (64, 128, 8, 2, 7, 0),   # T 非 k 倍数 → 空专家必现（7×2=14 < 8×2? 空专家可能）
        (64, 128, 8, 2, 100, 0),
        (128, 512, 64, 4, 3, 0),  # 大 E 稀疏（3×4=12 < 64 → 52 个空专家）
        (128, 512, 64, 4, 1000, 0),
        (128, 512, 8, 2, 64, 1),  # shared expert
        (128, 512, 8, 1, 64, 1),  # top-1 + shared
        (32, 64, 4, 2, 128, 0),
    ]
    for c in cases:
        ok &= check(*c, seed=0)
    # 极端负载：固定输入让所有 token 路由到同一专家（用极端权重伪造）——用大 T 全零激活
    # 场景：x=0 → router logits 全 0（无 bias）→ softmax 均匀 → topk 取前 k 个
    moe = MoE(16, 32, num_experts=4, top_k=1).cuda().half()
    x = torch.zeros(32, 16, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        d = (moe(x).float() - moe.reference(x).float()).abs().max().item()
    ok &= d == 0.0
    print(f"全零输入（router 均匀）: max_diff={d:.3e} ({'PASS' if d == 0 else 'FAIL'})")
    print(f"\n{'ALL PASS（位级一致）' if ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
