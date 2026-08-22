"""定位 fp8 varlen 内核与 flash varlen 的 causal 掩码语义差异。

构造可区分的 K/V（key 位置 t 的 K 行 = one-hot 在 t），单seq verify形状
（len=50 key、Q=5 query），对比三种输出的注意力权重分布：
  A. flash varlen（causal + block_table）
  B. 新 fp8 varlen 内核
  C. 精确参考1：query r attend keys 0..len-1+r（绝对位置——正确verify语义）
  D. 精确参考2：query r attend keys 0..r（行号语义）
"""
import os
import torch

from flash_attn import flash_attn_varlen_func
from nanovllm.layers.attention import paged_varlen_attention_fp8

torch.manual_seed(0)
H, D, KV = 16, 128, 8
BLOCK = 256
softmax_scale = 1.0 / (D ** 0.5)
len_k, Q = 50, 5

# K/V：K 行 = one-hot 编码位置（注意力权重即被attend的位置分布）；V 行 = 位置值
NUM_BLOCKS = 8
k_cache = torch.zeros(NUM_BLOCKS, BLOCK, KV, D, device="cuda")
v_cache = torch.zeros(NUM_BLOCKS, BLOCK, KV, D, device="cuda")
for t in range(len_k):
    k_cache[0, t, :, t % D] = 448.0 * 0.5   # 量化后 ≈ 224 → 反量化回 ~1（scale取1/448*2）
    v_cache[0, t, :, 0] = float(t)           # V 第一维 = 位置
k_scale = v_scale = 1.0 / 448.0 * 2.0       # 反量化后 K 行 = e_t（one-hot），V = t

q = torch.randn(Q, H, D, device="cuda").to(torch.float16)
cu_q = torch.tensor([0, Q], dtype=torch.int32, device="cuda")
key_lens = torch.tensor([len_k], dtype=torch.int32, device="cuda")
bt = torch.tensor([[0]], dtype=torch.int32, device="cuda")

# A: flash varlen（causal + paged）
cu_k = torch.tensor([0, len_k], dtype=torch.int32, device="cuda")
kd = k_cache.to(torch.float16) * k_scale
vd = v_cache.to(torch.float16) * v_scale
o_flash = flash_attn_varlen_func(q, kd, vd, max_seqlen_q=Q, cu_seqlens_q=cu_q,
                                 max_seqlen_k=len_k, cu_seqlens_k=cu_k,
                                 softmax_scale=softmax_scale, causal=True, block_table=bt)

# B: 新内核
o_new = paged_varlen_attention_fp8(q, k_cache, v_cache, cu_q, key_lens, bt,
                                   k_scale, v_scale, softmax_scale)

# C/D: 精确参考（绝对位置 / 行号）
def exact_ref(mode):
    out = torch.zeros(Q, H, D, device="cuda", dtype=torch.float32)
    for r in range(Q):
        if mode == "abs":
            keys = range(len_k - Q + r + 1)     # 0..len-1+r
        else:
            keys = range(r + 1)                 # 0..r
        for t in keys:
            w = torch.exp(torch.tensor(float(t), device="cuda") * softmax_scale)
            kv = v_cache[0, t, :, :].float().repeat(H // KV, 1) * v_scale  # [KV,D]→[H,D]
            out[r] += w * kv
        out[r] /= out[r][:, 0].sum().clamp_min(1e-9)
    return out

o_abs = exact_ref("abs")   # 注意：v 第一维是位置 → 输出[:, :, 0] 反映平均位置
o_row = exact_ref("row")

def pos(o):
    """输出[:, :, 0] 即被attend key位置的概率加权均值。"""
    return o[:, :, 0].mean(dim=1)

print("== 每个query attend的平均key位置（正确verify语义应≈len-Q+r） ==")
for r in range(Q):
    print(f"  query{r}: flash={pos(o_flash)[r]:6.2f}  内核={pos(o_new)[r]:6.2f}  "
          f"绝对参考={pos(o_abs)[r]:6.2f}  行号参考={pos(o_row)[r]:6.2f}  "
          f"(期望≈{len_k - Q + r})")
