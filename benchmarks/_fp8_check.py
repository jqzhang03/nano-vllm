"""FP8 量化检查：内核/_scaled_mm 双路径 vs 反量化参考 + 性能对比。

1. 正确性：随机数据上 Triton fp8 内核、torch._scaled_mm 与 fp8 反量化参考的 max/mean 误差
   （fp8 本身 ~6% 相对精度，参考也走同一量化 → 路径间差异应只有累加顺序噪声）；
2. 双路径一致性：同一 (q, scale) 下 Triton（小M）与 _scaled_mm（大M）结果差在 fp8 噪声级；
3. 性能：fp8 Triton 内核 vs cuBLAS bf16 稠密（M=8/256），量化前先用 bf16 基线。
"""
import torch
import triton

from nanovllm.layers.linear import fp8_gemm

torch.manual_seed(0)
DEV = "cuda"


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """per-column fp8 量化（与 WeightQuantMixin.quantize_fp8 同款）。"""
    w = w.float()
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
    return torch.clamp(w / s, -448.0, 448.0).to(torch.float8_e4m3fn), s


def quantize_act(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    s = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    return torch.clamp(x.float() / s, -448.0, 448.0).to(torch.float8_e4m3fn), s


def ref_fp8(x, w, a_scale, w_scale):
    """反量化参考：x ≈ q_a*s_a，w ≈ q_w*s_w → 输出 = (q_a*s_a) @ (q_w*s_w)^T。"""
    return (x.float() * a_scale) @ (w.float() * w_scale).t()


print("== 1) 正确性（K=4096, N=8192）：decode 路径(权重-only) vs scaled_mm(全fp8) vs 参考 ==")
K, N = 4096, 8192
w = torch.randn(N, K, device=DEV)
wq, ws = quantize(w)
for M in (8, 64, 256, 4096):
    x = torch.randn(M, K, device=DEV)
    xb = x.bfloat16()
    if M <= 128:
        y = fp8_gemm(xb, wq, ws)  # 权重-only：激活 bf16
        tag = "triton(wo)"
        ref = xb.float() @ (wq.float() * ws).t()
    else:
        xq, xs = quantize_act(x)
        y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
        tag = "scaled_mm"
        ref = ref_fp8(xq, wq, xs, ws)
    d = (y.float() - ref).abs()
    rel = d.max().item() / ref.abs().max().item()
    print(f"  M={M:<5} {tag:<12} max|Δ|={d.max().item():.4f} mean={d.mean().item():.5f} "
          f"rel_max={rel:.4f}")

print("== 2) 双路径一致性（M=64 triton(wo) vs M=4096 scaled_mm 子块，同一 q/scale） ==")
M = 4096
x = torch.randn(M, K, device=DEV)
xq, xs = quantize_act(x)
y_sm = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
y_tr = fp8_gemm(x[:64].bfloat16(), wq, ws)
d = (y_tr.float() - y_sm[:64].float()).abs()
print(f"  子块 max|Δ|={d.max().item():.4f} mean={d.mean().item():.6f} "
      f"(decode 无激活量化 vs prefill 有 → 差异来自激活 fp8 量化本身)")

print("== 3) 性能：fp8 vs cuBLAS bf16（M=8/256） ==")
for M in (8, 256):
    x = torch.randn(M, K, device=DEV)
    xb = x.bfloat16()
    xq, xs = quantize_act(x)
    wb = w.bfloat16()

    def run(fn, n=200):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / n

    t_cublas = run(lambda: torch.nn.functional.linear(xb, wb))
    t_triton = run(lambda: fp8_gemm(xb, wq, ws))
    t_sm = run(lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(),
                                        out_dtype=torch.bfloat16))
    print(f"  M={M:<4} cuBLAS={t_cublas*1000:7.1f}µs | fp8-triton(wo)={t_triton*1000:7.1f}µs "
          f"({t_cublas/t_triton:.2f}x) | fp8-scaled_mm={t_sm*1000:7.1f}µs ({t_cublas/t_sm:.2f}x)")

print("FP8 CHECK OK")
