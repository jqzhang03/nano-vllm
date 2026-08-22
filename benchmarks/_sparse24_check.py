"""2:4 稀疏 GEMM 正确性检查（standalone，vs 剪枝稠密参考）+ 微基准。

参考：F.linear(x, w_pruned)（剪枝后的稠密权重）——内核只是重新组织存储
（v + idx 打包），数学等价，误差应 ≈ 0（仅 bf16 MMA 累加顺序噪声）。
"""
import time

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import sparse24_gemm

torch.manual_seed(0)
DEV = "cuda"


def prune_2_4(w: torch.Tensor) -> torch.Tensor:
    w4 = w.view(*w.shape[:-1], w.shape[-1] // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    return (w4 * mask).view(w.shape)


def pack_2_4(w: torch.Tensor):
    """与 WeightQuantMixin.quantize_sparse24 相同的打包（v + idx_packed）。"""
    N, K = w.shape
    w4 = w.view(N, K // 4, 4)
    keep = w4.abs().argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(w4, dtype=torch.bool).scatter_(-1, keep, True)
    v = torch.gather(w4 * mask, -1, keep).view(N, K // 2).contiguous()
    idx = keep.to(torch.uint8).view(N, K // 2)
    idx_packed = (idx[:, 0::2] | (idx[:, 1::2] << 2)).contiguous()
    return v, idx_packed


def run_case(M, K, N, tol=0.01):
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    wp = prune_2_4(w)
    v, idx = pack_2_4(w)
    ref = F.linear(x, wp)
    out = sparse24_gemm(x, v, idx)
    err = (out.float() - ref.float()).abs()
    rel = err / (ref.float().abs() + 1e-3)
    print(f"M={M:5d} K={K:5d} N={N:6d}  max abs err={err.max().item():.6f}  "
          f"mean={err.mean().item():.7f}  max rel={rel.max().item():.5f}")
    assert err.max().item() < tol, "sparse24 gemm diverges"


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("=== 正确性（vs 剪枝稠密参考） ===")
for M, K, N in [(8, 1024, 1024), (256, 1024, 8192), (1, 1024, 4096),
                (33, 1024, 2048), (7, 4096, 1024), (64, 1024, 151936)]:
    run_case(M, K, N)

print("\n=== 性能（vs 稠密 bf16 GEMM，权重字节 0.625×） ===")
for M, K, N, name in [(8, 1024, 4096, "gate_up M=8"), (256, 1024, 4096, "gate_up M=256"),
                      (8, 1024, 151936, "lm_head M=8"), (256, 1024, 151936, "lm_head M=256"),
                      (8, 1024, 3072, "qkv M=8"), (256, 1024, 3072, "qkv M=256"),
                      (8, 4096, 1024, "down M=8"), (256, 4096, 1024, "down M=256")]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    v, idx = pack_2_4(w)
    t_dense = bench(lambda: F.linear(x, w))
    t_sp = bench(lambda: sparse24_gemm(x, v, idx))
    print(f"[{name}] dense {t_dense:.4f}ms  2:4 {t_sp:.4f}ms  speedup {t_dense / t_sp:.2f}x")

print("\nSPARSE24 GEMM OK")
