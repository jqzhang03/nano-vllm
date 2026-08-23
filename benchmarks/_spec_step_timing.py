"""Spec步级计时 v2：定位引擎 spec 步（87µs/tok）与探针 verify 前向（47µs/tok）之间
1.85× 差距的来源。把 run() 拆成 prepare / run_model(前向) / sampler 三阶段，
并跑同一 workload 的 baseline（spec off）做同口径对比。

用法（WSL，GPU）：
    python benchmarks/_spec_step_timing.py [--bs 256] [--out-len 96]
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def run_timed(llm, prompts, sps, spec_mode: bool):
    for pr, sp in zip(prompts, sps):
        llm.add_request(pr, sp)
    t_prep, t_fwd, t_samp, t_other = [], [], [], []
    rows_per_step = []
    while not llm.is_finished():
        seqs, kind = llm.scheduler.schedule()
        for old_id, new_id in llm.scheduler.cow_pairs:
            llm.model_runner.call("cow_block", old_id, new_id)
        runner = llm.model_runner
        t0 = time.perf_counter()
        if kind == "mixed":
            input_ids, positions = runner.prepare_mixed(seqs)
        elif kind == "prefill":
            input_ids, positions = runner.prepare_prefill(seqs)
        elif kind == "spec":
            input_ids, positions = runner.prepare_spec(seqs)
        else:
            input_ids, positions = runner.prepare_decode(seqs)
        temperatures = runner.prepare_sample(seqs)
        t1 = time.perf_counter()
        logits = runner.run_model(input_ids, positions, kind)
        t2 = time.perf_counter()
        token_ids = runner.sampler(logits, temperatures).tolist()
        t3 = time.perf_counter()
        t_prep.append(t1 - t0); t_fwd.append(t2 - t1); t_samp.append(t3 - t2)
        if any(s.draft_tokens is not None for s in seqs):
            rows_per_step.append(sum(1 for s in seqs if s.draft_tokens is not None))
        if any(s.draft_tokens is not None for s in seqs):
            token_lists, n_dec, _, _, _, _ = llm._verify(seqs, token_ids)
            llm.scheduler.postprocess_spec(seqs, token_lists)
        else:
            llm.scheduler.postprocess(seqs, token_ids)
    return t_prep, t_fwd, t_samp, rows_per_step


def main():
    global MODEL
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help="模型目录（默认 Qwen3-0.6B）")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--out-len", type=int, default=96)
    args = p.parse_args()
    MODEL = os.path.expanduser(args.model)  # bash argv 不展开 ~ → 手动展开

    text = "Repeat exactly, continuing the same digit forever: " + " ".join(["5"] * 40)
    prompts = [text] * args.bs
    sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.out_len)] * args.bs

    for spec_mode in (False, True):
        llm = LLM(MODEL, speculative="ngram" if spec_mode else "none",
                  max_model_len=4096, gpu_memory_utilization=0.9)
        llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
        # 先跑一遍真实workload（编译缓存热），再计时跑
        run_timed(llm, prompts, sps, spec_mode)
        t_prep, t_fwd, t_samp, rows = run_timed(llm, prompts, sps, spec_mode)
        llm.exit()
        torch.cuda.empty_cache()
        mode = "spec" if spec_mode else "baseline"
        tot_tok = args.bs * args.out_len - args.bs
        run_ms = sum(t_prep) + sum(t_fwd) + sum(t_samp)
        avg_rows = statistics.fmean(rows) if rows else 0.0
        print(f"\n=== {mode}（{len(t_fwd)} 步, 平均spec行/步={avg_rows:.0f}） ===")
        for name, arr in (("prepare", t_prep), ("run_model", t_fwd), ("sampler", t_samp)):
            print(f"  {name:<10} avg={statistics.fmean(arr) * 1000:6.2f}ms p50={statistics.median(arr) * 1000:6.2f}ms "
                  f"sum={sum(arr) * 1000:7.1f}ms")
        print(f"  总run时间={run_ms:.0f}ms → {tot_tok / run_ms * 1000:.0f} tok/s（纯前向口径）")


if __name__ == "__main__":
    main()
