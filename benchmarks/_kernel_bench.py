"""Standalone FP8 decode-attention kernel benchmark vs flash-attn (CUDA events).

Usage: python benchmarks/_kernel_bench.py [bs] [seqlen]
Measures per-step kernel time for:
  A) flash_attn_with_kvcache on fp16 cache (reference)
  B) our paged_decode_attention_fp8 kernel on fp8 cache
Both read the same KV content; times include the per-layer quantize step in B.
"""
import sys
import time

import torch
from flash_attn import flash_attn_with_kvcache

from nanovllm.layers.attention import paged_decode_attention_fp8

torch.manual_seed(0)
DEV = "cuda"
DTYPE = torch.bfloat16
NUM_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE = 256
SCALE = HEAD_DIM ** -0.5

BS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 160
NUM_BLOCKS = (SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE
ITERS = 50

# one block of fp8 KV per seq (random), same data for both paths
k_fp16 = torch.randn(BS * NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
v_fp16 = torch.randn(BS * NUM_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
k_scale = k_fp16.abs().max() / 448.0 * 1.1
v_scale = v_fp16.abs().max() / 448.0 * 1.1
k_cache = (k_fp16.float() / k_scale).to(torch.float8_e4m3fn)
v_cache = (v_fp16.float() / v_scale).to(torch.float8_e4m3fn)
k16 = (k_cache.to(DTYPE).float() * k_scale).to(DTYPE)
v16 = (v_cache.to(DTYPE).float() * v_scale).to(DTYPE)

q = torch.randn(BS, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
seqlens = torch.full((BS,), SEQ, dtype=torch.int32, device=DEV)
block_tables = torch.arange(BS * NUM_BLOCKS, dtype=torch.int32, device=DEV).reshape(BS, NUM_BLOCKS)

def timeit(fn, iters=ITERS):
    for _ in range(3):  # warmup
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

t_fa = timeit(lambda: flash_attn_with_kvcache(q.unsqueeze(1), k16, v16, cache_seqlens=seqlens,
                                              block_table=block_tables, softmax_scale=SCALE, causal=True))
t_fp8 = timeit(lambda: paged_decode_attention_fp8(q, k_cache, v_cache, block_tables, seqlens,
                                                  k_scale.item(), v_scale.item(), SCALE))
# fp8 kernel + write-side quantization (k.float()*inv -> fp8 cast, per layer)
t_fp8q = timeit(lambda: paged_decode_attention_fp8(q, (k16.float() * (1 / k_scale)).to(torch.float8_e4m3fn),
                                                   (v16.float() * (1 / v_scale)).to(torch.float8_e4m3fn),
                                                   block_tables, seqlens, k_scale.item(), v_scale.item(), SCALE))

print(f"bs={BS} seqlen={SEQ} blocks={NUM_BLOCKS}")
print(f"flash-attn fp16 : {t_fa:.3f} ms/step")
print(f"our fp8 kernel  : {t_fp8:.3f} ms/step  ({t_fp8 / t_fa:.2f}x)")
print(f"fp8 + quantize  : {t_fp8q:.3f} ms/step  ({t_fp8q / t_fa:.2f}x)")
