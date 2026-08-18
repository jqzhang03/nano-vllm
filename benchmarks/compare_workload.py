"""Generate an identical workload spec for BOTH engines (nano-vllm and real vLLM).

The synthetic workload mirrors `bench.build_workload` exactly (same RNG stream,
same token-id sampling), but instead of building it inside each runner it is
materialized once to JSON so both sides execute the *same* prompts with the
*same* per-seq max_tokens. Pure stdlib — runs in any env.

Usage:
    python benchmarks/compare_workload.py --num-seqs 256 --tag clean
    python benchmarks/compare_workload.py --num-seqs 128 \
        --min-input-len 1024 --max-input-len 1024 --min-output-len 128 --max-output-len 128 --tag long

Output: results/compare_workload_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import os
import random

TOKEN_SPACE = range(0, 10000)  # same token-id space as bench.py


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="clean")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--min-input-len", type=int, default=128)
    p.add_argument("--max-input-len", type=int, default=1024)
    p.add_argument("--min-output-len", type=int, default=64)
    p.add_argument("--max-output-len", type=int, default=512)
    p.add_argument("--shared-prefix-len", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None, help="JSON path (default: results/compare_workload_<tag>.json)")
    args = p.parse_args()

    rng = random.Random(args.seed)
    prompts = [
        rng.sample(TOKEN_SPACE, rng.randint(args.min_input_len, args.max_input_len))
        for _ in range(args.num_seqs)
    ]
    if args.shared_prefix_len > 0:
        prefix = rng.sample(TOKEN_SPACE, args.shared_prefix_len)
        prompts = [prefix + p for p in prompts]
    max_tokens = [
        rng.randint(args.min_output_len, args.max_output_len) for _ in range(args.num_seqs)
    ]

    spec = {
        "tag": args.tag,
        "seed": args.seed,
        "num_seqs": args.num_seqs,
        "min_input_len": args.min_input_len,
        "max_input_len": args.max_input_len,
        "min_output_len": args.min_output_len,
        "max_output_len": args.max_output_len,
        "shared_prefix_len": args.shared_prefix_len,
        "prompts": prompts,
        "max_tokens": max_tokens,
    }
    out_path = args.output or os.path.join("results", f"compare_workload_{args.tag}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    total_in = sum(len(p) for p in prompts)
    total_out = sum(max_tokens)
    print(f"workload '{args.tag}': {args.num_seqs} seqs | "
          f"input {args.min_input_len}-{args.max_input_len} | "
          f"output {args.min_output_len}-{args.max_output_len} | "
          f"total_in {total_in:,} tok | total_out {total_out:,} tok")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
