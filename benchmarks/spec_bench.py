"""n-gram投机解码基准：同一workload下 spec on/off 对比 + 接受率统计。

三种workload风格（诚实光谱，同一引擎同一seed）：
  free     随机token段落（最差情况：几乎无重复n-gram）
  json     JSON记录续写（结构化：中度重复）
  repeat   重复指令（最好情况：模型输出高重复度 → α高）

报告：吞吐（decode_tokens/wall）、TTFT/TPOT p50、接受率α、平均草稿长γ、
tokens/步、verify步数、抢占数。输出CSV + JSON。

用法（WSL，GPU）：
    python benchmarks/spec_bench.py [--styles free,json,repeat] [--num-seqs 64] [--fp8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import torch

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
EOS_IDS = {151643, 151645, 151646, 151647}  # Qwen3 special tokens（仅用于构造重复文本时避开）


def build_prompts(style: str, tokenizer, num_seqs: int, min_len: int, max_len: int, seed: int):
    """返回 prompts 文本列表。"""
    rng = random.Random(seed)
    if style == "free":
        # 随机token段落（不经过tokenizer；与bench.py的synthetic workload同款）
        return [None] * num_seqs, lambda: [rng.sample(range(0, 10000), rng.randint(min_len, max_len))
                                           for _ in range(num_seqs)]
    if style == "repeat":
        # echo任务：模型被强提示复读同一数字 → 草稿=续写（最好情况，α→高）
        text = "Repeat exactly, continuing the same digit forever: 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5"
        return [text] * num_seqs, None
    # json：结构化记录续写
    records = []
    for i in range(20):
        records.append(f'{{"id": {i}, "name": "user_{i}", "score": {i / 20:.2f}, '
                       f'"active": {i % 2 == 0}, "tags": ["a", "b", "c_{i % 3}"]}}')
    text = "Complete the JSON array with more records of the same format:\n[" + ",\n".join(records) + "]"
    return [text] * num_seqs, None


def run_case(style: str, speculative: str, prompts, tokenize, tokenizer, args):
    if tokenize is not None:
        prompts_tok = tokenize()
    else:
        prompts_tok = [tokenizer.encode(t) for t in prompts]
    llm = LLM(MODEL, speculative=speculative, kv_cache_dtype="fp8_e4m3" if args.fp8 else "auto",
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    sps = [SamplingParams(temperature=0.6, max_tokens=args.max_output_len, ignore_eos=True)] * len(prompts_tok)
    t0 = time.perf_counter()
    llm.generate(prompts_tok, sps, use_tqdm=False)
    wall = time.perf_counter() - t0
    m = llm.collect_metrics()
    stats = m["step_stats"]
    ttft = _summarize([r["t_first_token"] - r["t_submitted"] for r in m["per_request"]])
    tpot = _summarize([(r["t_completed"] - r["t_first_token"]) / (r["completion_tokens"] - 1)
                       for r in m["per_request"] if r["completion_tokens"] > 1])
    row = {
        "style": style, "speculative": speculative,
        "wall_s": round(wall, 3),
        "throughput_tok_per_s": round(stats["decode_tokens"] / wall, 1),
        "ttft_p50_ms": ttft, "tpot_p50_ms": tpot,
        "preemptions": m.get("num_preemptions", 0),
        "prefill_steps": stats["prefill_steps"], "decode_steps": stats["decode_steps"],
    }
    if speculative == "ngram":
        d, acc = stats["spec_draft_tokens"], stats["spec_accepted_drafts"]
        rows_spec = stats["spec_rows"]
        row.update({
            "alpha": round(acc / d, 3) if d else None,
            "avg_draft_len": round(d / rows_spec, 2) if rows_spec else None,
            "spec_steps": stats["spec_steps"],
            "spec_verify_tokens": stats["spec_verify_tokens"],
        })
    llm.exit()
    torch.cuda.empty_cache()  # 同进程多引擎必须显式归还显存（caching allocator不自动还驱动）
    return row


def _summarize(vals):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(int(len(vals) * 0.5), len(vals) - 1)] * 1000, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--styles", default="free,json,repeat")
    p.add_argument("--num-seqs", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=256)
    p.add_argument("--max-output-len", type=int, default=96)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--fp8", action="store_true", help="fp8 KV cache（与fp16对比）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/spec_bench.csv")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    rows = []
    for style in [s.strip() for s in args.styles.split(",")]:
        prompts, tokenize = build_prompts(style, tokenizer, args.num_seqs,
                                          args.max_input_len, args.max_input_len, args.seed)
        print(f"\n=== style={style} (fp8={args.fp8}) ===")
        base = run_case(style, "none", prompts, tokenize, tokenizer, args)
        spec = run_case(style, "ngram", prompts, tokenize, tokenizer, args)
        rows += [base, spec]
        print(f"  baseline: {base['throughput_tok_per_s']:7.1f} tok/s | "
              f"TTFT {base['ttft_p50_ms']}ms | TPOT {base['tpot_p50_ms']}ms")
        speedup = spec["throughput_tok_per_s"] / base["throughput_tok_per_s"]
        print(f"  spec    : {spec['throughput_tok_per_s']:7.1f} tok/s ({speedup:+.2f}x) | "
              f"TTFT {spec['ttft_p50_ms']}ms | TPOT {spec['tpot_p50_ms']}ms | "
              f"α={spec.get('alpha')} | avg γ={spec.get('avg_draft_len')}")

    with open(args.output, "w", newline="") as f:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r.keys()))
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(args.output.replace(".csv", ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": MODEL, "num_seqs": args.num_seqs, "fp8": args.fp8},
                   "rows": rows}, f, indent=2)
    print(f"\nCSV -> {args.output}, JSON -> {args.output.replace('.csv', '.json')}")


if __name__ == "__main__":
    main()
