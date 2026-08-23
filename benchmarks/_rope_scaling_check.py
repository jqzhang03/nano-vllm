"""llama3 rope_scaling 单元检查：本实现 vs HF transformers 的 LlamaRotaryEmbedding。

CPU-only。对给定 config 的 rope_scaling（llama3 变体）比较逐频率 inv_freq 与 cos/sin 缓存。
"""
import os
import sys

import torch
from transformers import AutoConfig

from nanovllm.layers.rotary_embedding import _scaled_inv_freq_llama3

path = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Llama-3.1-8B")
cfg = AutoConfig.from_pretrained(path)
rs = getattr(cfg, "rope_scaling", None)
print("rope_scaling:", rs)
if not rs or rs.get("rope_type") != "llama3":
    print("SKIP: no llama3 rope_scaling in config")
    sys.exit(0)

# 我们的 inv_freq（与 build_cache 同路径）
dim = cfg.head_dim or cfg.hidden_size // cfg.num_attention_heads
base = getattr(cfg, "rope_theta", 500000.0)
inv_freq_ours = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float) / dim))
inv_freq_ours = _scaled_inv_freq_llama3(
    inv_freq_ours, rs["factor"], rs["high_freq_factor"],
    rs["low_freq_factor"], rs["original_max_position_embeddings"])

# HF 参考
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
hf_rope = LlamaRotaryEmbedding(cfg)
inv_freq_hf = hf_rope.inv_freq.detach().float()
print(f"inv_freq: ours {tuple(inv_freq_ours.shape)} vs hf {tuple(inv_freq_hf.shape)}")
d = (inv_freq_ours - inv_freq_hf).abs()
print(f"inv_freq max diff: {d.max().item():.3e}  (前6个 ours={inv_freq_ours[:6].tolist()})")
assert d.max().item() < 1e-6, "inv_freq mismatch vs HF!"

# 缓存级对比（前 64 个位置）：参考 = 用 HF 的 inv_freq 按同公式构造
from nanovllm.layers.rotary_embedding import RotaryEmbedding
t = torch.arange(64, dtype=torch.float)
freqs_hf = torch.einsum("i,j -> ij", t, inv_freq_hf)
cos_ref, sin_ref = freqs_hf.cos(), freqs_hf.sin()
rope = RotaryEmbedding(dim, dim, cfg.max_position_embeddings, base, dict(rs))
cache = rope.cos_sin_cache[torch.arange(64)].squeeze(1).float()  # [64, 2*dim]
cos_ours, sin_ours = cache.chunk(2, dim=-1)
print(f"cos max diff: {(cos_ours - cos_ref).abs().max().item():.3e} | "
      f"sin max diff: {(sin_ours - sin_ref).abs().max().item():.3e}")
assert (cos_ours - cos_ref).abs().max().item() < 1e-5
assert (sin_ours - sin_ref).abs().max().item() < 1e-5
print("ROPE_SCALING OK")
