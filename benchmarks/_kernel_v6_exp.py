"""FP8 decode kernel variants experiment.

Variant B: direct fp8 loads + native cvt (no LUT gather), unmasked loads,
           score-masked. BLOCK_T x num_warps sweep.
Variant C: same as B + scale folding (k_scale into softmax_scale, v_scale
           folded into p before acc).

Usage: python benchmarks/_kernel_v6_exp.py [bs] [seqlen] [--all]
"""
import sys

import torch
import triton
import triton.language as tl

from nanovllm.layers.attention import paged_decode_attention_fp8  # v4 (LUT) ref

torch.manual_seed(0)
DEV = "cuda"
DTYPE = torch.bfloat16
NUM_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE = 256
SCALE = HEAD_DIM ** -0.5


@triton.jit
def fp8_direct_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, cache_seqlens_ptr, o_ptr,
    k_scale, v_scale, softmax_scale,
    max_blocks, num_heads, kv_heads,
    head_dim: tl.constexpr, num_groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """Direct fp8 loads, unmasked; scale folding: qk * (softmax*k_scale), p *= v_scale."""
    pid = tl.program_id(0)
    seq_id = pid // kv_heads
    kv_head = pid % kv_heads
    seqlen = tl.load(cache_seqlens_ptr + seq_id)
    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, num_groups)
    q = tl.load(q_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
                + offs_g[:, None] * head_dim + offs_d[None, :]).to(tl.float32)  # [G, D]

    acc = tl.zeros([num_groups, head_dim], dtype=tl.float32)
    m = tl.full([num_groups, 1], float("-inf"), dtype=tl.float32)
    l = tl.zeros([num_groups, 1], dtype=tl.float32)
    qk_scale = softmax_scale * k_scale

    num_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_stride = BLOCK_SIZE * kv_heads * head_dim
    for b in range(num_blocks):
        block_id = tl.load(block_table_ptr + seq_id * max_blocks + b)
        base = block_id * block_stride + kv_head * head_dim
        for t in range(0, BLOCK_SIZE, BLOCK_T):
            offs_t = t + tl.arange(0, BLOCK_T)
            tok_mask = (b * BLOCK_SIZE + offs_t) < seqlen
            k_ptrs = k_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            v_ptrs = v_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            k = tl.load(k_ptrs).to(tl.float32)      # [T, D] fp32 (unmasked)
            v = tl.load(v_ptrs).to(tl.float32)
            s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * qk_scale  # [G, T]
            s = tl.where(tok_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1)[:, None])
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)
            l = l * alpha + tl.sum(p, axis=1)[:, None]
            acc = acc * alpha + tl.sum(p[:, :, None] * (v * v_scale)[None, :, :], axis=1)
            m = m_new
    o = acc / l
    tl.store(o_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
             + offs_g[:, None] * head_dim + offs_d[None, :], o.to(q_ptr.dtype.element_ty))


def fp8_direct(q, k_cache, v_cache, block_table, cache_seqlens,
               k_scale, v_scale, softmax_scale, block_t, num_warps):
    bs, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    max_blocks = block_table.shape[1]
    num_groups = num_heads // kv_heads
    o = torch.empty_like(q)
    grid = (bs * kv_heads,)
    fp8_direct_kernel[grid](
        q, k_cache, v_cache, block_table, cache_seqlens, o,
        k_scale, v_scale, softmax_scale,
        max_blocks, num_heads, kv_heads,
        head_dim=head_dim, num_groups=num_groups,
        BLOCK_SIZE=k_cache.shape[1], BLOCK_T=block_t,
        num_warps=num_warps,
    )
    return o


@triton.jit
def fp8_mma_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, cache_seqlens_ptr, o_ptr,
    k_scale, v_scale, softmax_scale,
    max_blocks, num_heads, kv_heads,
    head_dim: tl.constexpr, num_groups: tl.constexpr, QPAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, BLOCK_T: tl.constexpr,
):
    """MMA version with QPAD: GQA group (G=2) padded to QPAD=16 for tl.dot's N>=16.

    s  [T,QPAD] = dot(k16 [T,D], q16 [D,QPAD]); garbage columns masked -inf.
    acc[D,QPAD] = dot(v_t [D,T], p16 [T,QPAD]).
    q loaded masked (columns >= num_groups are 0), stored masked back.
    """
    pid = tl.program_id(0)
    seq_id = pid // kv_heads
    kv_head = pid % kv_heads
    seqlen = tl.load(cache_seqlens_ptr + seq_id)
    offs_d = tl.arange(0, head_dim)
    offs_g = tl.arange(0, QPAD)
    g_valid = offs_g[None, :] < num_groups
    q_base = q_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
    q = tl.load(q_base + offs_d[:, None] + offs_g[None, :] * head_dim,
                mask=g_valid, other=0.0).to(tl.float32)                   # [D, QPAD]
    q16 = q.to(tl.float16)

    acc = tl.zeros([head_dim, QPAD], dtype=tl.float32)                    # [D, QPAD]
    m = tl.full([1, QPAD], float("-inf"), dtype=tl.float32)
    l = tl.zeros([1, QPAD], dtype=tl.float32)

    num_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_stride = BLOCK_SIZE * kv_heads * head_dim
    for b in range(num_blocks):
        block_id = tl.load(block_table_ptr + seq_id * max_blocks + b)
        base = block_id * block_stride + kv_head * head_dim
        for t in range(0, BLOCK_SIZE, BLOCK_T):
            offs_t = t + tl.arange(0, BLOCK_T)
            tok_mask = (b * BLOCK_SIZE + offs_t) < seqlen
            k_ptrs = k_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            k16 = (tl.load(k_ptrs).to(tl.float32) * k_scale).to(tl.float16)  # [T, D]
            s = tl.dot(k16, q16, out_dtype=tl.float32) * softmax_scale       # [T, QPAD]
            s = tl.where(tok_mask[:, None] & g_valid, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=0)[None, :])
            alpha = tl.exp(m - m_new)
            p = tl.exp(s - m_new)
            l = l * alpha + tl.sum(p, axis=0)[None, :]
            v_ptrs = v_cache_ptr + base + offs_t[:, None] * (kv_heads * head_dim) + offs_d[None, :]
            v_t = (tl.load(v_ptrs).to(tl.float32) * v_scale).to(tl.float16)  # [T, D]
            acc = acc * alpha + tl.dot(tl.trans(v_t), p.to(tl.float16), out_dtype=tl.float32)
            m = m_new
    o = acc / l                                                            # [D, QPAD]
    tl.store(o_ptr + seq_id * num_heads * head_dim + kv_head * num_groups * head_dim
             + offs_d[:, None] + offs_g[None, :] * head_dim,
             o.to(q_ptr.dtype.element_ty), mask=g_valid)


def fp8_mma(q, k_cache, v_cache, block_table, cache_seqlens,
            k_scale, v_scale, softmax_scale, block_t, num_warps):
    bs, num_heads, head_dim = q.shape
    kv_heads = k_cache.shape[2]
    max_blocks = block_table.shape[1]
    num_groups = num_heads // kv_heads
    qpad = max(16, num_groups)
    o = torch.empty_like(q)
    grid = (bs * kv_heads,)
    fp8_mma_kernel[grid](
        q, k_cache, v_cache, block_table, cache_seqlens, o,
        k_scale, v_scale, softmax_scale,
        max_blocks, num_heads, kv_heads,
        head_dim=head_dim, num_groups=num_groups, QPAD=qpad,
        BLOCK_SIZE=k_cache.shape[1], BLOCK_T=block_t,
        num_warps=num_warps,
    )
    return o


def build(bs, seqlen):
    num_blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    k_fp16 = torch.randn(bs * num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
    v_fp16 = torch.randn(bs * num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
    k_scale = k_fp16.abs().max() / 448.0 * 1.1
    v_scale = v_fp16.abs().max() / 448.0 * 1.1
    k_cache = (k_fp16.float() / k_scale).to(torch.float8_e4m3fn)
    v_cache = (v_fp16.float() / v_scale).to(torch.float8_e4m3fn)
    q = torch.randn(bs, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
    seqlens = torch.full((bs,), seqlen, dtype=torch.int32, device=DEV)
    block_tables = torch.arange(bs * num_blocks, dtype=torch.int32, device=DEV).reshape(bs, num_blocks)
    return q, k_cache, v_cache, block_tables, seqlens, k_scale.item(), v_scale.item()


def timeit(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    all_cfg = "--all" in sys.argv
    cfgs = [(128, 1024), (256, 1024), (128, 160)] if all_cfg else \
           [(int(sys.argv[1]), int(sys.argv[2]))]
    for bs, seqlen in cfgs:
        q, k_cache, v_cache, blk_tbl, sl, ks, vs = build(bs, seqlen)
        o_ref = paged_decode_attention_fp8(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE)
        t_ref = timeit(lambda: paged_decode_attention_fp8(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE))
        print(f"== bs={bs} seqlen={seqlen} | ref(LUT,BT32,w1): {t_ref:.3f} ms")
        best = (t_ref, "ref", 32, 1)
        for bt in (32, 64, 128):
            for w in (1, 2, 4):
                if bt == 32 and w != 1:
                    continue
                try:
                    o = fp8_direct(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE, bt, w)
                    torch.cuda.synchronize()
                    err = (o - o_ref).abs().max().item()
                    t = timeit(lambda: fp8_direct(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE, bt, w))
                    print(f"  direct BT={bt:3d} w={w}: {t:.3f} ms ({t/t_ref:.2f}x) max_err={err:.2e}")
                    if err < 1e-3 and t < best[0]:
                        best = (t, "direct", bt, w)
                except Exception as e:
                    print(f"  direct BT={bt:3d} w={w}: FAILED {type(e).__name__}: {str(e)[:100]}")
        for bt in (32, 64):
            for w in (1, 2, 4):
                try:
                    o = fp8_mma(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE, bt, w)
                    torch.cuda.synchronize()
                    err = (o - o_ref).abs().max().item()
                    t = timeit(lambda: fp8_mma(q, k_cache, v_cache, blk_tbl, sl, ks, vs, SCALE, bt, w))
                    print(f"  mma    BT={bt:3d} w={w}: {t:.3f} ms ({t/t_ref:.2f}x) max_err={err:.2e}")
                    if err < 1e-3 and t < best[0]:
                        best = (t, "mma", bt, w)
                except Exception as e:
                    print(f"  mma    BT={bt:3d} w={w}: FAILED {type(e).__name__}")
                    print(str(e))
        print(f"  => best: {best[1]} BT={best[2]} w={best[3]} {best[0]:.3f} ms ({best[0]/t_ref:.2f}x of ref)")


if __name__ == "__main__":
    main()
