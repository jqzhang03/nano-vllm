"""FP8 varlen（verify）内核检查：paged_varlen_attention_fp8 vs flash varlen 反量化参考。

构造 verify 形状（多seq、Q=γ+1∈[1,5]、变长key、随机分页），对比输出 logits。
"""
import os
import torch
import triton

from flash_attn import flash_attn_varlen_func
from nanovllm.layers.attention import paged_varlen_attention_fp8

torch.manual_seed(0)
H, D, KV = 16, 128, 8        # Qwen3-0.6B 形状
G = H // KV
BLOCK = 256
NUM_BLOCKS = 64
softmax_scale = 1.0 / (D ** 0.5)

# 随机缓存（fp8量化 + scale）
k_cache = (torch.randn(NUM_BLOCKS, BLOCK, KV, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)
v_cache = (torch.randn(NUM_BLOCKS, BLOCK, KV, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)
k_scale = v_scale = 0.5 / 448.0 * 2.0  # 与引擎同量级

def make_case(n_seqs, qlens, key_lens):
    """构造一批verify行：qlens[i]=Q，key_lens[i]=len+γ。"""
    cu_q = [0]
    qs = []
    bts = []
    for i in range(n_seqs):
        cu_q.append(cu_q[-1] + qlens[i])
        qs.append(torch.randn(qlens[i], H, D))
        nb = (key_lens[i] + BLOCK - 1) // BLOCK
        bts.append([(i * 3 + b) % NUM_BLOCKS for b in range(nb)])
    q = torch.cat(qs, dim=0).to(torch.float16).cuda()
    cu_q = torch.tensor(cu_q, dtype=torch.int32).cuda()
    max_blocks = max(len(b) for b in bts)
    bt = torch.tensor([b + [-1] * (max_blocks - len(b)) for b in bts], dtype=torch.int32).cuda()
    key_lens_t = torch.tensor(key_lens, dtype=torch.int32).cuda()
    return q, cu_q, key_lens_t, bt, n_seqs

def reference(q, cu_q, key_lens, bt, n_seqs):
    """参考：反量化缓存 + flash varlen（paged）。"""
    kd = k_cache.to(torch.float16) * k_scale
    vd = v_cache.to(torch.float16) * v_scale
    cu_k = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), key_lens]), dim=0).to(torch.int32)
    # 需要全局key偏移：验证语义下key从0开始（每seq独立）
    return flash_attn_varlen_func(
        q, kd, vd,
        max_seqlen_q=5, cu_seqlens_q=cu_q,
        max_seqlen_k=max(key_lens.tolist()), cu_seqlens_k=cu_k,
        softmax_scale=softmax_scale, causal=True, block_table=bt)

def main():
    print("== FP8 varlen 内核 vs flash varlen 参考 ==")
    ok = True
    cases = [
        (4, [1, 2, 3, 5], [64, 130, 300, 520]),          # 混合Q、跨块
        (8, [5] * 8, [256, 257, 300, 511, 512, 513, 700, 800]),
        (2, [4, 5], [28, 28]),                            # 短key（单块内）
        (16, [2, 3, 4, 5] * 4, [120] * 16),
    ]
    for ci, (n, qlens, key_lens) in enumerate(cases):
        q, cu_q, kl, bt, n_seqs = make_case(n, qlens, key_lens)
        o_ref = reference(q, cu_q, kl, bt, n_seqs)
        o_new = paged_varlen_attention_fp8(q, k_cache, v_cache, cu_q, kl, bt,
                                           k_scale, v_scale, softmax_scale)
        diff = (o_new.float() - o_ref.float()).abs()
        top1_new = o_new.reshape(-1, D).argmax(-1)
        top1_ref = o_ref.reshape(-1, D).argmax(-1)
        agree = (top1_new == top1_ref).float().mean().item()
        # 随机数据的logits近并列：argmax翻转是预期（max|Δ|才是数值一致性的判据）
        status = "OK " if diff.max().item() < 0.01 else "FAIL"
        ok &= status == "OK "
        print(f"  [{status}] case{ci}: n={n} max|Δ|={diff.max().item():.4f} "
              f"mean={diff.mean().item():.6f} top1一致={agree:.3f}")
    print("PASS: 内核与参考一致（误差=量化噪声）。" if ok else "FAIL!")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
