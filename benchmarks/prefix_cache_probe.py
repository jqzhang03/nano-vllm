"""Verify the scheduler's prefix-cache hit estimate against the tokens actually reused.

The scheduler's Top-W cache-affinity ranking scores a waiting sequence with
``BlockManager.resolve_prefix_features(seq)`` -> ``cached_tokens``. That is the
same estimate the admission/scheduling policy uses to decide which request is
cheap. The *committed* reuse is applied later by ``BlockManager.allocate`` ->
``Sequence.num_prefix_cached_tokens`` (the number the server reports as
``context.prefix_cache_hit_tokens`` and the scheduler counts in
``prefix_cache_hit_tokens``).

The two can disagree when the cache changes between ranking and allocation
(eviction, deferred free, another sequence publishing the same block) or when a
sequence is preempted and re-prefilled. This probe measures the gap directly:

  * wave 1 publishes a shared prefix (nothing can hit yet);
  * wave 2 re-submits exact repeats, repeats with a new suffix, and a shorter
    shared prefix, so the estimate for every request is non-trivial.

Run inside the WSL env from the repo root::

    python benchmarks/prefix_cache_probe.py --model ~/huggingface/Qwen3-0.6B/
"""
from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--prefix-len", type=int, default=512,
                        help="shared prefix length of wave 1")
    parser.add_argument("--suffix-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


class EstimateRecorder:
    """Record the Top-W cache estimate for every sequence the scheduler ranks."""

    def __init__(self, engine):
        self.scheduler = engine.scheduler
        self.estimates: dict[int, int] = {}
        self._original = self.scheduler._waiting_score
        self.scheduler._waiting_score = self._recording_score

    def _recording_score(self, seq, now, index):
        score = self._original(seq, now, index)
        if seq.seq_id not in self.estimates:
            features, _ = self.scheduler.block_manager.resolve_prefix_features(seq)
            self.estimates[seq.seq_id] = features.cached_tokens
        return score

    def restore(self) -> None:
        self.scheduler._waiting_score = self._original

    def take(self) -> dict[int, int]:
        values = dict(self.estimates)
        self.estimates.clear()
        return values


def summarize_hits(values: list[int]) -> dict:
    ordered = sorted(values)
    unique = sorted(set(values))
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "max": ordered[-1] if ordered else None,
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "p50": ordered[len(ordered) // 2] if ordered else None,
        "distinct": unique[:12],
        "non_zero": sum(value > 0 for value in ordered),
    }


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    engine = LLM(args.model, execution_mode="mixed", max_model_len=args.max_model_len,
                 gpu_memory_utilization=args.gpu_memory_utilization)
    recorder = EstimateRecorder(engine)
    try:
        vocab = max(2, int(getattr(engine.tokenizer, "vocab_size", 32000)))
        engine.generate([[(1234 - i * 7919) % vocab for i in range(64)]],
                        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4),
                        use_tqdm=False)
        engine._req_metrics = []
        engine.scheduler.reset_metrics()
        recorder.take()

        prefix = rng.sample(range(0, 10000), args.prefix_len)
        params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_tokens)

        def run_wave(prompts, label):
            engine._req_metrics = []
            estimates = recorder.take()  # drop warmup estimates
            engine.generate(prompts, [params] * len(prompts), use_tqdm=False)
            estimates.update(recorder.take())
            rows = []
            for prompt, committed in zip(prompts, engine._req_metrics):
                seq_id = committed["seq_id"]
                rows.append({
                    "wave": label,
                    "prompt_tokens": committed["prompt_tokens"],
                    "estimated_cached_tokens": estimates.get(seq_id),
                    "committed_cached_tokens": committed["prefix_cached_tokens"],
                })
            return rows

        # Wave 1: publishes the shared prefix, so every estimate must be 0.
        wave1 = [prefix + rng.sample(range(0, 10000), args.suffix_len) for _ in range(4)]
        wave1_rows = run_wave(wave1, "publish")
        # Wave 2: exact repeats + repeats with fresh suffixes + half-prefix reuse.
        wave2 = list(wave1) + [
            prefix + rng.sample(range(0, 10000), args.suffix_len),
            prefix[:args.prefix_len // 2] + rng.sample(range(0, 10000), args.suffix_len),
        ]
        wave2_rows = run_wave(wave2, "reuse")
        rows = wave1_rows + wave2_rows

        matched = [row for row in rows
                   if row["estimated_cached_tokens"] is not None
                   and row["committed_cached_tokens"] is not None]
        exact = sum(row["estimated_cached_tokens"] == row["committed_cached_tokens"]
                    for row in matched)
        block_size = engine.config.kvcache_block_size
        result = {
            "meta": {
                "model": args.model,
                "date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "block_size": block_size,
            },
            "config": vars(args),
            "requests": rows,
            "comparison": {
                "rows_with_both_numbers": len(matched),
                "rows_without_estimate": sum(
                    row["estimated_cached_tokens"] is None for row in rows),
                "exact_matches": exact,
                "exact_match_rate": exact / len(matched) if matched else None,
                "estimated": summarize_hits(
                    [row["estimated_cached_tokens"] for row in matched]),
                "committed": summarize_hits(
                    [row["committed_cached_tokens"] for row in matched]),
                "absolute_error": summarize_hits(
                    [abs(row["estimated_cached_tokens"]
                         - row["committed_cached_tokens"]) for row in matched]),
                "block_aligned": all(
                    row["committed_cached_tokens"] % block_size == 0 or
                    row["committed_cached_tokens"] == row["prompt_tokens"]
                    for row in matched),
            },
            "scheduler": {
                key: engine.collect_metrics()[key] for key in (
                    "prefix_cache_hit_tokens", "prefix_cache_hit_requests",
                    "num_prefix_feature_parses", "num_prefix_feature_reuses",
                    "prefix_cache_evictions", "prefix_cache_lru_entries",
                    "num_preemptions",
                )
            },
        }
        print(json.dumps(result, indent=2, default=str))
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, default=str)
    finally:
        recorder.restore()
        engine.exit()


if __name__ == "__main__":
    main()
