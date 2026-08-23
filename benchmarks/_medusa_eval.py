"""Medusa头快速评估：同源分布（训练prompt）下 none/ngram/medusa 对比。

用法（WSL，GPU）：
    python benchmarks/_medusa_eval.py [--bs 12] [--out-len 96]
"""
from __future__ import annotations

import argparse
import os
import time

from nanovllm import LLM, SamplingParams
from benchmarks.medusa_train import REAL_PROMPTS  # 复用训练prompt（同源分布）

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="模型目录（默认 Qwen3-0.6B）")
    p.add_argument("--bs", type=int, default=12)
    p.add_argument("--out-len", type=int, default=96)
    p.add_argument("--medusa-path", default="results/medusa_heads.pt")
    args = p.parse_args()
    args.model = os.path.expanduser(args.model)  # bash argv 不展开 ~ → 手动展开

    llm = LLM(args.model, gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    tokenizer = llm.tokenizer
    llm.exit()
    torch.cuda.empty_cache()
    eval_prompts = [tokenizer.encode(pp) for pp in REAL_PROMPTS[:args.bs]]
    sps = [SamplingParams(temperature=0.6, max_tokens=args.out_len, ignore_eos=True)] * len(eval_prompts)
    for mode, mpath in (("none", ""), ("ngram", ""), ("medusa", args.medusa_path)):
        evallm = LLM(args.model, speculative=mode, medusa_path=mpath, gpu_memory_utilization=0.9)
        evallm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
        t0 = time.perf_counter()
        evallm.generate(eval_prompts, sps, use_tqdm=False)
        wall = time.perf_counter() - t0
        stats = evallm.collect_metrics()["step_stats"]
        tok = stats["decode_tokens"]
        line = f"   {mode:<6}: {tok} token / {wall:.2f}s = {tok / wall:.0f} tok/s"
        if mode != "none":
            d = stats["spec_draft_tokens"]
            acc = stats["spec_accepted_drafts"]
            line += f" | α = {acc / d if d else float('nan'):.3f}"
        print(line, flush=True)
        evallm.exit()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import torch
    main()
