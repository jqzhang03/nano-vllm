"""FP8 paged decode attention kernel correctness check (standalone).

Builds a random decode scenario: bs sequences, each with a random KV history in
fp8 blocks, and compares the Triton kernel output against flash_attn_with_kvcache
on the dequantized fp16 cache (tolerance accounts for fp8 quantization error).
"""
import torch

from nanovllm.layers.attention import paged_decode_attention_fp8

torch.manual_seed(0)
DEV = "cuda"
DTYPE = torch.bfloat16
NUM_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
BLOCK_SIZE = 256
MAX_BLOCKS = 8
BS = 32
SCALE = HEAD_DIM ** -0.5

seqlens = torch.randint(1, MAX_BLOCKS * BLOCK_SIZE, (BS,), device=DEV, dtype=torch.int32)
max_blocks = ((seqlens.max() + BLOCK_SIZE - 1) // BLOCK_SIZE).item()
num_blocks = ((seqlens + BLOCK_SIZE - 1) // BLOCK_SIZE)

# random KV history in fp16, quantized to fp8 with a per-layer scale
k_hist = torch.randn(BS * max_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
v_hist = torch.randn(BS * max_blocks, BLOCK_SIZE, KV_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1
k_scale = k_hist.abs().max() / 448.0 * 1.1
v_scale = v_hist.abs().max() / 448.0 * 1.1
k_cache = (k_hist.float() / k_scale).to(torch.float8_e4m3fn)
v_cache = (v_hist.float() / v_scale).to(torch.float8_e4m3fn)

q = torch.randn(BS, NUM_HEADS, HEAD_DIM, device=DEV, dtype=DTYPE) * 0.1

block_tables = torch.zeros(BS, max_blocks, dtype=torch.int32, device=DEV)
base = 0
for i, nb in enumerate(num_blocks):
    block_tables[i, :nb] = torch.arange(base, base + nb, dtype=torch.int32, device=DEV)
    base += nb

out = paged_decode_attention_fp8(q, k_cache, v_cache, block_tables, seqlens, k_scale.item(), v_scale.item(), SCALE)

# reference: dequantize cache, run flash_attn_with_kvcache on fp16 cache
k16 = (k_cache.to(DTYPE).float() * k_scale).to(DTYPE)
v16 = (v_cache.to(DTYPE).float() * v_scale).to(DTYPE)
from flash_attn import flash_attn_with_kvcache
ref = flash_attn_with_kvcache(q.unsqueeze(1), k16, v16, cache_seqlens=seqlens,
                              block_table=block_tables, softmax_scale=SCALE, causal=True)

err = (out.unsqueeze(1).float() - ref.float()).abs()
rel = err / (ref.float().abs() + 1e-3)
print(f"max abs err  : {err.max().item():.6f}")
print(f"max rel err  : {rel.max().item():.6f}")
print(f"mean abs err : {err.mean().item():.6f}")
print(f"ref abs mean : {ref.float().abs().mean().item():.6f}")
# fp8 quantization noise floor for these magnitudes ~1/448 of scale ratio
noise = (k_hist.float() - (k_cache.float() * k_scale)).abs().mean()
print(f"fp8 kv noise (k): {noise.item():.6f}")
assert err.max().item() < 0.05, "kernel output diverges from flash-attn reference"
print("FP8 decode kernel OK")
