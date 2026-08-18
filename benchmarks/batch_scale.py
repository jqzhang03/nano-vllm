"""Batch scaling experiment: throughput / TTFT / TPOT vs num_seqs.

Constructs ONE engine (one model load + one CUDA-graph capture), then runs the
same workload shape with increasing batch sizes; wall time and per-request
metrics come from the SAME generate() call. Saves CSV + PNG under results/.

Run (WSL):
    python benchmarks/batch_scale.py --num-seqs-list 16,32,64,128,256,384,512
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from nanovllm import LLM, SamplingParams
from bench import build_workload, summarize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    p.add_argument("--num-seqs-list", default="16,32,64,128,256,384,512",
                   help="comma-separated batch sizes to sweep")
    p.add_argument("--min-input-len", type=int, default=64)
    p.add_argument("--max-input-len", type=int, default=256)
    p.add_argument("--min-output-len", type=int, default=32)
    p.add_argument("--max-output-len", type=int, default=128)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--warmup-seqs", type=int, default=8)
    p.add_argument("--prompts-file", default=None,
                   help="JSONL of {\"prompt\": ...} lines; overrides synthetic lengths")
    p.add_argument("--shared-prefix-len", type=int, default=0,
                   help="common prefix shared by all prompts (prefix-cache workload)")
    p.add_argument("--output", default="results/batch_scale.csv")
    p.add_argument("--plot", default="results/batch_scale.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    num_seqs_list = [int(x) for x in args.num_seqs_list.split(",")]
    os.makedirs("results", exist_ok=True)

    llm = LLM(args.model, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    llm.generate(["warm up"] * args.warmup_seqs,
                 SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

    rows = []
    for n in num_seqs_list:
        args.num_seqs = n
        args.seed = args.seed + n  # 每个batch用独立seed，避免相邻run的prompt重叠（重叠会引入前缀缓存命中，污染数据）
        prompts, sps = build_workload(args)
        t0 = time.perf_counter()
        llm.generate(prompts, sps, use_tqdm=False)
        wall = time.perf_counter() - t0
        m = llm.collect_metrics()
        stats = m["step_stats"]
        ttft = summarize(r["t_first_token"] - r["t_submitted"] for r in m["per_request"])
        tpot = summarize((r["t_completed"] - r["t_first_token"]) / (r["completion_tokens"] - 1)
                         for r in m["per_request"] if r["completion_tokens"] > 1)
        row = {
            "num_seqs": n,
            "wall_seconds": wall,
            "throughput_tok_per_s": stats["decode_tokens"] / wall,
            "ttft_p50_ms": round(ttft["p50"] * 1000, 1),
            "ttft_p99_ms": round(ttft["p99"] * 1000, 1),
            "tpot_p50_ms": round(tpot["p50"] * 1000, 2),
            "tpot_p99_ms": round(tpot["p99"] * 1000, 2),
            "preemptions": m.get("num_preemptions", 0),
            "prefill_tokens": stats["prefill_tokens"],
            "decode_tokens": stats["decode_tokens"],
        }
        rows.append(row)
        print(f"num_seqs={n:4d}  wall={wall:5.2f}s  throughput={row['throughput_tok_per_s']:6.0f} tok/s  "
              f"ttft_p50={row['ttft_p50_ms']:7.1f}ms  tpot_p50={row['tpot_p50_ms']:6.2f}ms  "
              f"preemptions={row['preemptions']}")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV -> {args.output}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ns = [r["num_seqs"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(ns, [r["throughput_tok_per_s"] for r in rows], "o-", color="#1f77b4", label="throughput (tok/s)")
        ax1.set_xlabel("num_seqs (batch size)")
        ax1.set_ylabel("throughput (output tok/s)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        ax2.plot(ns, [r["tpot_p50_ms"] for r in rows], "s--", color="#d62728", label="TPOT p50 (ms)")
        ax2.plot(ns, [r["ttft_p50_ms"] for r in rows], "^--", color="#2ca02c", label="TTFT p50 (ms)")
        ax2.set_ylabel("latency (ms)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        for r in rows:
            if r["preemptions"] > 0:
                ax1.annotate(f"preemptions={r['preemptions']}", (r["num_seqs"], r["throughput_tok_per_s"]),
                             textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, color="red")
        ax1.set_title(f"batch scaling — {os.path.basename(args.model.rstrip('/'))} "
                      f"({torch.cuda.get_device_name(0)})")
        fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"plot -> {args.plot}")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
