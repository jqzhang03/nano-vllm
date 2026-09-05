"""fp8 KV + 滚动环 端到端（阶段 2b 扩展 M1，GPU）：mistral toy。

对照链（跑通≠写对）：
1) ring(fp8) vs masked(fp8)：同一 fp8 内核、同一行值序列，环只换页地址 →
   逐 decode 步 logits 应**逐位相等**（引擎 B 与 A 同种子同采样 → 轨迹一致）；
2) ring(fp8) 采样步 vs 手工 bf16 窗口参考（模块级，与引擎无关）：fp8 KV
   量化噪声带内的 top-1 一致性（允许少量翻转）。

用法: python benchmarks/_ring_fp8_e2e.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from _ring_e2e import build_toy, masked_ref_logits

TOY = os.path.expanduser("~/mistral_toy")
DUMP_A = "/tmp/_ring_fp8_a.pt"
W = 512
B = 256


def run_engine(rolling: bool, seed: int):
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(seed)
    llm = LLM(TOY, max_model_len=1500, quantization="none", kv_swap=False,
              rolling_cache=rolling, kv_cache_dtype="fp8_e4m3",
              max_num_seqs=8, max_num_batched_tokens=4096)
    out = llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1100),
                       use_tqdm=False, collect_logits=True)
    dec = [lg for kind, lg in llm.collected_logits
           if kind == "decode" and lg is not None]
    llm.exit()
    return dec, [o["token_ids"] for o in out]


def main():
    global PROMPTS
    PROMPTS = build_toy()   # 确定性重建（seed 11，幂等）
    plen = [len(p) for p in PROMPTS]

    dec_a, comp_a = run_engine(True, 20240201)
    torch.save({"decode": dec_a, "completions": comp_a, "plen": plen}, DUMP_A)
    dec_b, comp_b = run_engine(False, 20240201)
    assert len(dec_a) == len(dec_b), f"decode 步数不一致 {len(dec_a)} vs {len(dec_b)}"
    assert comp_a == comp_b, "轨迹分歧——环(fp8)与掩码(fp8)采样结果不同"
    neq = 0
    maxd = 0.0
    for s, (la, lb) in enumerate(zip(dec_a, dec_b)):
        if not torch.equal(la, lb):
            neq += 1
            maxd = max(maxd, (la - lb).abs().max().item())
    print(f"fp8 decode steps {len(dec_a)}: 环 vs 掩码 逐位不等步 {neq}，max abs diff {maxd:.2e}")
    assert neq == 0, "fp8 环与 fp8 掩码应逐位一致"

    # 与手工 bf16 窗口参考对照（量化噪声带）
    from transformers import AutoConfig
    from nanovllm.models.mistral import MistralForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(TOY)
    ref = MistralForCausalLM(cfg).eval()
    load_model(ref, TOY)
    ref = ref.to(torch.bfloat16).cuda()
    if cfg.tie_word_embeddings:
        ref.lm_head.weight.data = ref.model.embed_tokens.weight.data
    bad = total = 0
    for si in range(len(plen)):
        so_far = PROMPTS[si] + comp_a[si]
        last_ok = min(len(comp_a[si]) - 1, len(dec_a) - 1)
        steps = set(range(0, last_ok + 1, 71))
        for n_w in (W - 1, W, W + B - 1, W + B, 2 * W, 2 * W + B):
            s = n_w - plen[si] - 1
            if 0 <= s <= last_ok:
                steps.add(s)
        for s in sorted(steps):
            lg = masked_ref_logits(ref, so_far, plen[si] + s + 1)
            e = dec_a[s][si].float()
            agree = int(e.argmax() == lg.argmax())
            total += 1
            bad += 0 if agree else 1
            dd = (e - lg).abs()
            print(f"seq{si} step{s:4d} L={plen[si]+s+1:4d}: max {dd.max().item():.3f} "
                  f"mean {dd.mean().item():.4f} top1 {'Y' if agree else 'N'}")
    print(f"fp8 环 vs bf16 手工参考: checked {total}, top1 mismatches {bad}")
    assert bad <= max(2, total // 40), "fp8 环对照失败过多（量化噪声带）"
    print("FP8 RING E2E OK")


if __name__ == "__main__":
    main()
