"""INT4 内核调试：逐步验证 tl.interleave 语义、半字节拆解、完整内核 vs fp32 精确参考。"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from nanovllm.layers.linear import gemm_int4_kernel, int4_gemm

DEV = "cuda"
torch.manual_seed(0)


@triton.jit
def interleave_kernel(a_ptr, b_ptr, c_ptr, N: tl.constexpr):
    offs = tl.arange(0, N // 2)
    a = tl.load(a_ptr + offs).to(tl.bfloat16)
    b = tl.load(b_ptr + offs).to(tl.bfloat16)
    c = tl.interleave(a, b)
    tl.store(c_ptr + tl.arange(0, N), c)


@triton.jit
def nibble_kernel(b_ptr, lo_ptr, hi_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    b8 = tl.load(b_ptr + offs)
    lo = (b8 & 0x0F) - 8
    hi = ((b8 >> 4) & 0x0F) - 8
    tl.store(lo_ptr + offs, lo.to(tl.int32))
    tl.store(hi_ptr + offs, hi.to(tl.int32))


print("=== 1) tl.interleave 语义 ===")
N = 8
a = torch.arange(N // 2, device=DEV)
b = torch.arange(N // 2, device=DEV) + 100
c = torch.empty(N, device=DEV, dtype=torch.bfloat16)
interleave_kernel[(1,)](a, b, c, N=N)
print("interleave(a,b):", c.tolist(), " 期望 [0,100,1,101,2,102,3,103]")

print("\n=== 2) 半字节拆解（offset 编码：码-8=真值） ===")
data = torch.tensor([0x12, 0x34, 0xAB, 0xF0], dtype=torch.uint8, device=DEV).to(torch.int8)
lo = torch.empty(4, device=DEV, dtype=torch.int32)
hi = torch.empty(4, device=DEV, dtype=torch.int32)
nibble_kernel[(1,)](data, lo, hi, N=4)
print("lo:", lo.tolist(), " 期望 [-6,-4,3,-8]")
print("hi:", hi.tolist(), " 期望 [-7,-5,2,7]")

print("\n=== 3) 完整内核 vs fp32 精确参考（无bf16舍入） ===")
for M, K, N in [(8, 128, 64), (8, 256, 64), (8, 128, 128), (8, 256, 128),
                (8, 1024, 64), (8, 1024, 1024), (256, 1024, 4096)]:
    group = 128
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_g = w.float().view(N, K // group, group)
    scale = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(w_g / scale), -7, 7)
    w_hat = (q * scale).view(N, K)  # fp32 反量化（精确）
    ref = F.linear(x.float(), w_hat)  # fp32 matmul
    q4 = (q.to(torch.int8) + 8).to(torch.uint8).view(N, K)
    packed = (q4[0::2, :] | (q4[1::2, :] << 4)).to(torch.int8).t().contiguous()
    out = int4_gemm(x, packed, scale.squeeze(-1).to(torch.bfloat16))
    err = (out.float() - ref).abs()
    print(f"M={M} K={K} N={N}: max err={err.max().item():.5f} mean={err.mean().item():.6f}")
    if err.max().item() > 0.1:
        am, an = torch.unravel_index(err.argmax(), err.shape)
        print(f"  argmax at (m={am.item()}, n={an.item()}): out={out[am, an].item():.4f} "
              f"ref={ref[am, an].item():.4f}")
        nb = err.view(M, N // 64, 64).amax(dim=(0, 2))
        print("  per-N-block max err:", [f"{v:.3f}" for v in nb.tolist()])
        if K > 128:
            # 每 K 块贡献：把 ref 拆成 K 块各自点积后相加 vs out——直接看前两 K 块单独的输出
            pass
