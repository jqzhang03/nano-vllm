"""W8A8 int8 GEMM correctness check (standalone, vs fp16 matmul).

Tests both per-channel (num_groups=1) and per-group (128) weight scales.
Reference: dequantize both operands in fp16 and run F.linear — mathematically
identical to the int8 kernel (per-token x-scale, per-group w-scale).
"""
import torch
import torch.nn.functional as F

from nanovllm.layers.linear import w8a8_gemm

torch.manual_seed(0)
DEV = "cuda"


def run_case(M, K, N, group):
    x = torch.randn(M, K, device=DEV) * 0.5
    w = torch.randn(N, K, device=DEV) * 0.02

    a_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    a_int8 = torch.clamp(torch.round(x / a_scale), -127, 127).to(torch.int8)

    if group > 1:
        w_g = w.view(N, K // group, group)
        b_scale = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0  # [N, K//g, 1]
        b_int8 = torch.clamp(torch.round(w_g / b_scale), -127, 127).to(torch.int8)  # [N, K//g, g]
        b_scale2 = b_scale.squeeze(-1)  # [N, K//g]
    else:  # per-channel: one scale per output row, keep [N, 1] (num_groups=1)
        b_scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0   # [N, 1]
        b_int8 = torch.clamp(torch.round(w / b_scale), -127, 127).to(torch.int8)
        b_scale2 = b_scale

    # fp16 reference on the dequantized operands
    x_hat = (a_int8.float() * a_scale)  # [M, K]
    w_hat = (b_int8.float() * b_scale).view(N, K)  # [N, K]
    ref = F.linear(x_hat, w_hat)

    out = w8a8_gemm(a_int8, b_int8.view(N, K), a_scale.squeeze(-1), b_scale2)
    err = (out.float() - ref.float()).abs()
    rel = err / (ref.float().abs() + 1e-3)
    print(f"M={M:5d} K={K:5d} N={N:5d} group={group:4d}  max abs err={err.max().item():.5f}  "
          f"mean abs err={err.mean().item():.6f}  max rel={rel.max().item():.4f}")
    assert err.max().item() < 0.05, "W8A8 gemm diverges"


for group in (128, 1):
    for M, K, N in [(256, 1024, 1024), (1, 1024, 3072), (4096, 1024, 256),
                    (33, 128, 97), (64, 2048, 512)]:
        run_case(M, K, N, group)

print("W8A8 GEMM OK (per-channel + per-group)")
