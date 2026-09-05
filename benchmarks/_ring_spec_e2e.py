"""投机(ngram)+滚动环 端到端（阶段 2b 扩展 M3，GPU）：mistral toy W=512。

对照：ring(rolling_cache=True) vs 掩码(rolling_cache=False)，同种子同温度，
同 ngram 参数。两引擎轨迹在 logits 位级一致时相同（草稿来自同一 token
历史）；logits 微小差可能在采样处分叉 → 只统计"共同前缀"内步骤：
- spec/mixed 步的 verify logits 行（环装配 vs 掩码分页）逐行 top-1；
- decode 步 logits 逐行 top-1；mean diff 量级报告。
另附 collect_metrics 的投机统计（α 等）对照。

用法: python benchmarks/_ring_spec_e2e.py [fp8|bf16] [ring]
（无参 = 跑全 4 组合；对子内对照）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/mistral_toy")
VOCAB = 2048
W = 512
DUMP = "/tmp/_ring_spec.pt"


def build_prompts():
    from _ring_e2e import build_toy
    build_toy()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)

    def _ids(text):
        return [t % VOCAB for t in tok.encode(text)]

    rep = "The quick brown fox jumps over the lazy dog. " * 4
    return [_ids(rep[:200]), _ids("Once upon a time, in a land far far away, "
                                  "there was a little fox. " * 3)]


def run_one(rolling: bool, fp8: bool, seed: int, prompts):
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(seed)
    llm = LLM(TOY, max_model_len=1500, quantization="none", kv_swap=False,
              rolling_cache=rolling,
              kv_cache_dtype="fp8_e4m3" if fp8 else "auto",
              speculative="ngram", max_draft_len=4,
              ngram_window=4, ngram_min_window=1,
              max_num_seqs=8, max_num_batched_tokens=4096)
    out = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=900),
                       use_tqdm=False, collect_logits=True)
    logs = [(k, lg) for k, lg in llm.collected_logits if lg is not None]
    comps = [o["token_ids"] for o in out]
    met = llm.collect_metrics()["step_stats"]
    llm.exit()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return logs, comps, met


def cpl(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def compare(fp8: bool):
    prompts = build_prompts() if fp8 else None  # 同一进程内共用
    prompts = prompts or _PROMPTS
    ring = run_one(True, fp8, 4242, prompts)
    mask = run_one(False, fp8, 4242, prompts)
    logs_r, comps_r, met_r = ring
    logs_m, comps_m, met_m = mask
    n = min(len(logs_r), len(logs_m))
    cpls = [cpl(comps_r[i], comps_m[i]) for i in range(len(comps_r))]
    agree_tot = agree_ok = flip = 0
    maxd = 0.0
    mean_sum = 0.0
    for s in range(n):
        kr, lr = logs_r[s]
        km, lm = logs_m[s]
        if kr != km:
            break  # 调度分歧（轨迹已分叉）
        a = lr.float()
        b = lm.float()
        if a.shape != b.shape:
            break
        # verify 步每 seq 多行：按完成共同前缀裁剪到可比行
        for i in range(a.shape[0]):
            step_ctx = min(cpls)  # 简化：全 seq 共同前缀内才可比
            if s > step_ctx:
                continue
            agree_tot += 1
            ok = int(a[i].argmax() == b[i].argmax())
            agree_ok += ok
            flip += 0 if ok else 1
            dd = (a[i] - b[i]).abs()
            maxd = max(maxd, dd.max().item())
            mean_sum += dd.mean().item()
    mean_d = mean_sum / max(agree_tot, 1)
    ar, dr = met_r["spec_accepted_drafts"], met_r["spec_draft_tokens"]
    am, dm = met_m["spec_accepted_drafts"], met_m["spec_draft_tokens"]
    print(f"[{'fp8' if fp8 else 'bf16'}] 可比行 {agree_tot}: top-1 一致 "
          f"{100*agree_ok/max(agree_tot,1):.2f}% (翻转 {flip}) | max {maxd:.4f} "
          f"mean {mean_d:.5f}")
    print(f"  α: ring {ar}/{dr}={ar/max(dr,1):.3f} | masked {am}/{dm}={am/max(dm,1):.3f}")
    assert flip <= max(2, agree_tot // 40), f"{'fp8' if fp8 else 'bf16'} 环spec 对照翻转过多"
    print(f"RING-SPEC ({'fp8' if fp8 else 'bf16'}) E2E OK")


if __name__ == "__main__":
    _PROMPTS = None
    args = [a for a in sys.argv[1:] if a in ("fp8", "bf16", "ring", "mask")]
    want = set(args) or {"bf16", "fp8"}
    _PROMPTS = build_prompts()
    if "bf16" in want:
        compare(False)
    if "fp8" in want:
        compare(True)
