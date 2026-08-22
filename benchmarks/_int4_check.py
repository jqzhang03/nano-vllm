"""INT4 反量化 GEMM 正确性检查（standalone，vs 反量化 fp16 参考）+ 微基准。

参考：w_hat = (q - 8) * scale（对称 int4，offset 码偏移 +8），F.linear(x, w_hat)
——与内核数学等价（内核 = 寄存器内拆半字节 → (码-8)*组scale → bf16 dot）。
"""
import time

import torch
import torch.nn.functional as F

from nanovllm.layers.linear import int4_gemm

torch.manual_seed(0)
DEV = "cuda"


def quantize_ref(w: torch.Tensor, group: int = 128):
    """与 WeightQuantMixin.quantize_int4 相同的量化（返回反量化权重 + scale）。"""
    N, K = w.shape
    w_g = w.view(N, K // group, group)
    scale = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(w_g / scale), -7, 7)
    return (q * scale).view(N, K), scale.squeeze(-1)


def pack_2(w: torch.Tensor, scale: torch.Tensor, group: int = 128):
    """与 quantize_int4 相同的打包：[N, K//2] int8，字节低/高半字节 = k=2j/2j+1。"""
    N, K = w.shape
    q4 = (torch.clamp(torch.round(w / scale[:, torch.arange(K) // group]), -7, 7)
          .to(torch.int8) + 8).to(torch.uint8)
    packed = (q4[:, 0::2] | (q4[:, 1::2] << 4)).to(torch.int8)
    return packed.contiguous()


def run_case(M, K, N, group=128, tol=0.05):
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_hat, scale = quantize_ref(w.float(), group)
    ref = F.linear(x, w_hat.to(torch.bfloat16))
    packed = pack_2(w.float(), scale, group)
    out = int4_gemm(x, packed, scale.to(torch.bfloat16))
    err = (out.float() - ref.float()).abs()
    rel = err / (ref.float().abs() + 1e-3)
    print(f"M={M:5d} K={K:5d} N={N:6d}  max abs err={err.max().item():.5f}  "
          f"mean={err.mean().item():.6f}  max rel={rel.max().item():.4f}")
    assert err.max().item() < tol, "int4 gemm diverges"


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


print("=== 正确性（vs 反量化参考） ===")
for M, K, N in [(8, 1024, 1024), (256, 1024, 8192), (1, 1024, 4096),
                (33, 1024, 2048), (7, 4096, 1024), (64, 1024, 151936)]:
    run_case(M, K, N)

print("\n=== 性能（vs 稠密 bf16 GEMM） ===")
for M, K, N, name in [(8, 1024, 4096, "gate_up M=8"), (256, 1024, 4096, "gate_up M=256"),
                      (8, 1024, 151936, "lm_head M=8"), (256, 1024, 151936, "lm_head M=256"),
                      (8, 1024, 3072, "qkv M=8"), (256, 1024, 3072, "qkv M=256"),
                      (8, 4096, 1024, "down M=8"), (256, 4096, 1024, "down M=256")]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_hat, scale = quantize_ref(w.float())
    packed = pack_2(w.float(), scale)
    t_dense = bench(lambda: F.linear(x, w))
    t_int4 = bench(lambda: int4_gemm(x, packed, scale.to(torch.bfloat16)))
    print(f"[{name}] dense {t_dense:.4f}ms  int4 {t_int4:.4f}ms  speedup {t_dense / t_int4:.2f}x")

print("\n=== 双路径一致性（int4 内核 vs w_deq 稠密反量化，同一份 q/scale） ===")
for M, K, N in [(8, 1024, 8192), (256, 1024, 4096), (8, 1024, 1024), (64, 4096, 1024)]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_hat, scale = quantize_ref(w.float())
    packed = pack_2(w.float(), scale)
    # w_deq = dequant(q, scale)（与 WeightQuantMixin.quantize_int4 相同的解包）
    lo = (packed & 0x0F) - 8
    hi = ((packed >> 4).to(torch.int8) & 0x0F) - 8
    w_deq = torch.zeros(N, K, device=DEV, dtype=torch.float32)
    w_deq[:, 0::2] = lo.float() * scale[:, torch.arange(K // 2) // 64]
    w_deq[:, 1::2] = hi.float() * scale[:, torch.arange(K // 2) // 64]
    y_kernel = int4_gemm(x, packed, scale.to(torch.bfloat16))
    y_dense = F.linear(x, w_deq.to(torch.bfloat16))
    d = (y_kernel.float() - y_dense.float()).abs().max().item()
    print(f"M={M:4d} K={K:5d} N={N:6d}  int4 vs dense max|Δ|={d:.6f}")
    assert d < 0.05, "int4 kernel vs w_deq dense path diverged"

print("\nINT4 GEMM OK")
