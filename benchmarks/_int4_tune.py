"""INT4 内核变体调优：对比 interleave 版 vs 2-dot 拆分版（沿K打包），
扫 tile/warps/stages；参照 = cuBLAS 稠密 + 自写 Triton 稠密（上限参照）。"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

DEV = "cuda"
torch.manual_seed(0)


@triton.jit
def gemm_bf16_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=a_mask & b_mask)


@triton.jit
def gemm_int4_v2_kernel(
    a_ptr, b_ptr, c_ptr, b_scale_ptr,
    M, N, K, num_groups,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """2-dot 拆分：B [N, K//2] int8（字节低半=偶数k、高半=奇数k），
    a_e/a_o 步长2加载 → 两个 (BM, BK/2)×(BK/2, BN) dot，无 interleave。"""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k2 = tl.arange(0, BLOCK_K // 2)
    a_e_ptrs = a_ptr + offs_m[:, None] * stride_am + (2 * offs_k2)[None, :] * stride_ak
    a_o_ptrs = a_e_ptrs + stride_ak
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn + offs_k2[:, None] * stride_bk
    a_mask = offs_m[:, None] < M
    b_mask = offs_n[None, :] < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_e = tl.load(a_e_ptrs, mask=a_mask, other=0.0)
        a_o = tl.load(a_o_ptrs, mask=a_mask, other=0.0)
        b8 = tl.load(b_ptrs, mask=b_mask, other=0)
        group = (k // BLOCK_K) % num_groups
        s = tl.load(b_scale_ptr + offs_n * num_groups + group, mask=offs_n < N, other=1.0)
        lo = ((b8 & 0x0F) - 8).to(tl.float32) * s[None, :]
        hi = (((b8 >> 4) & 0x0F) - 8).to(tl.float32) * s[None, :]
        acc += tl.dot(a_e, lo.to(tl.bfloat16), out_dtype=tl.float32)
        acc += tl.dot(a_o, hi.to(tl.bfloat16), out_dtype=tl.float32)
        a_e_ptrs += BLOCK_K * stride_ak
        a_o_ptrs += BLOCK_K * stride_ak
        b_ptrs += (BLOCK_K // 2) * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=a_mask & b_mask)


def quantize(w, group=128):
    N, K = w.shape
    w_g = w.float().view(N, K // group, group)
    scale = w_g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(w_g / scale), -7, 7)
    return (q * scale).view(N, K), scale.squeeze(-1)


def pack_k(w, scale, group=128):
    """沿K打包：字节 j = 偶数k(低) | 奇数k(高)，[N, K//2] int8。"""
    N, K = w.shape
    q4 = (torch.clamp(torch.round(w / scale[:, torch.arange(K) // group]), -7, 7)
          .to(torch.int8) + 8).to(torch.uint8)
    return (q4[:, 0::2] | (q4[:, 1::2] << 4)).to(torch.int8).contiguous()


def run_int4_v2(a, packed, scale, bm, bn, bk, warps, stages):
    M, K = a.shape
    N = scale.shape[0]
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_int4_v2_kernel[grid](
        a, packed, out, scale,
        M, N, K, scale.shape[1],
        a.stride(0), a.stride(1), packed.stride(0), packed.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps, num_stages=stages,
    )
    return out


def run_dense_triton(a, w, bm, bn, bk, warps):
    M, K = a.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    gemm_bf16_kernel[grid](
        a, w, out, M, N, K,
        a.stride(0), a.stride(1), w.stride(1), w.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=8,
        num_warps=warps,
    )
    return out


def bench(fn, iters=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# 正确性（v2）
for M, K, N in [(8, 1024, 4096), (256, 1024, 151936)]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_hat, scale = quantize(w)
    packed = pack_k(w, scale)
    ref = F.linear(x, w_hat.to(torch.bfloat16))
    out = run_int4_v2(x, packed, scale.to(torch.bfloat16), 64, 128, 128, 4, 2)
    err = (out.float() - ref.float()).abs().max().item()
    print(f"v2 correctness M={M} K={K} N={N}: max err={err:.5f}")
    assert err < 0.05

print("\n=== 性能（us） ===")
cases = [(8, 1024, 4096, "gate_up M=8"), (256, 1024, 4096, "gate_up M=256"),
         (8, 1024, 151936, "lm_head M=8"), (256, 1024, 151936, "lm_head M=256"),
         (8, 1024, 3072, "qkv M=8"), (8, 4096, 1024, "down M=8")]
for M, K, N, name in cases:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    w_hat, scale = quantize(w)
    packed = pack_k(w, scale)
    s16 = scale.to(torch.bfloat16)
    t_cublas = bench(lambda: F.linear(x, w))
    print(f"[{name}] cuBLAS {t_cublas * 1e3:.1f}us")
    for bm, bn, bk, warps, stages in [(16, 128, 128, 4, 2), (16, 128, 128, 4, 3),
                                      (32, 128, 128, 4, 2), (64, 128, 128, 4, 2),
                                      (64, 128, 128, 8, 2), (64, 64, 128, 4, 2),
                                      (16, 256, 128, 8, 2), (64, 256, 128, 8, 3)]:
        try:
            t = bench(lambda: run_int4_v2(x, packed, s16, bm, bn, bk, warps, stages))
            print(f"    v2 bm={bm} bn={bn} bk={bk} w={warps} s={stages}: {t * 1e3:7.1f}us  "
                  f"{t_cublas / t:.2f}x")
        except Exception as e:
            print(f"    v2 bm={bm} bn={bn} w={warps} s={stages}: FAIL {type(e).__name__}: {str(e)[:80]}")

# Triton 稠密上限参照（gate_up M=8 与 M=256）
for M, K, N, name in [(8, 1024, 4096, "dense-triton gate_up M=8"), (256, 1024, 4096, "dense-triton gate_up M=256")]:
    x = (torch.randn(M, K, device=DEV) * 0.5).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.bfloat16)
    for bm, bn, bk, warps in [(64, 64, 128, 4), (64, 128, 64, 8), (128, 128, 64, 8)]:
        t = bench(lambda: run_dense_triton(x, w, bm, bn, bk, warps))
        print(f"[{name} bm={bm} bn={bn} bk={bk} w={warps}] {t * 1e3:7.1f}us")
