"""Qwen3-0.6B 真实层形状下 fp8 vs cuBLAS bf16 的逐层微基准（M=8 与 M=256）。"""
import torch

from nanovllm.layers.linear import fp8_gemm

torch.manual_seed(0)
DEV = "cuda"
SHAPES = [  # (名字, N, K)  Qwen3-0.6B
    ("gate_up", 8192, 1024),
    ("qkv", 4096, 1024),
    ("o_proj", 1024, 2048),
    ("down", 1024, 3072),
    ("lm_head", 151936, 1024),
]


def quantize(w):
    w = w.float()
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
    return torch.clamp(w / s, -448.0, 448.0).to(torch.float8_e4m3fn), s


def quantize_act(x):
    s = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    return torch.clamp(x.float() / s, -448.0, 448.0).to(torch.float8_e4m3fn), s


def bench(fn, n):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(n):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / n


for M in (8, 256):
    print(f"== M={M} ==")
    for name, N, K in SHAPES:
        w = torch.randn(N, K, device=DEV)
        wq, ws = quantize(w)
        x = torch.randn(M, K, device=DEV)
        xq, xs = quantize_act(x)
        wb, xb = w.bfloat16(), x.bfloat16()
        n = 200
        t_cu = bench(lambda: torch.nn.functional.linear(xb, wb), n)
        t_tr = bench(lambda: fp8_gemm(xq, xs, wq, ws), n)
        t_sm = bench(lambda: torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws.t(),
                                              out_dtype=torch.bfloat16), n)
        print(f"  {name:<9} N={N:<6} cuBLAS={t_cu*1000:8.1f}µs | triton={t_tr*1000:8.1f}µs "
              f"({t_cu/t_tr:4.2f}x) | scaled_mm={t_sm*1000:8.1f}µs ({t_cu/t_sm:4.2f}x)")
