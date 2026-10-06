"""Compare continuous-arrival scheduling policies on the same request trace.

Examples (run from the repository root in the WSL inference environment)::

    python benchmarks/scheduling_ablation.py --num-seqs 128 --arrival-rate 16 \
        --arrival-mode poisson --shared-prefix-len 512

    python benchmarks/scheduling_ablation.py --variants baseline all_on \
        --num-seqs 256 --arrival-mode burst

Each variant gets a fresh engine and the same prompts, sampling parameters, and
relative arrival times. Results include per-request latency and scheduler counters.
"""
from __future__ import annotations

import argparse
import gc
import json
from math import isfinite
import os
import random
import time
from datetime import datetime, timezone

import torch

try:
    from .bench import add_workload_args, build_workload, env_info, summarize
except ImportError:  # direct script execution: python benchmarks/scheduling_ablation.py
    from bench import add_workload_args, build_workload, env_info, summarize

from nanovllm import LLM, SamplingParams


FEATURES = (
    "latency_aware_scheduling",
    "cache_affinity_admission",
    "prefix_feature_cache",
    "aging_fairness",
    "recompute_aware_preemption",
    "slo_aware_scheduling",
    "tpot_aware_scheduling",
    "multi_step_decode",
    "decode_burst_yield",
)


def variant_table() -> dict[str, dict[str, bool]]:
    all_off = {feature: False for feature in FEATURES}
    all_on = {feature: True for feature in FEATURES}
    variants = {"baseline": all_off, "all_on": all_on}
    names = {
        "latency_aware_scheduling": "without_latency",
        "cache_affinity_admission": "without_cache_affinity",
        "prefix_feature_cache": "without_prefix_feature_cache",
        "aging_fairness": "without_aging",
        "recompute_aware_preemption": "without_recompute_aware_preemption",
        "slo_aware_scheduling": "without_slo_adaptive",
        "tpot_aware_scheduling": "without_tpot_aware",
        "multi_step_decode": "without_multi_step_decode",
        "decode_burst_yield": "without_decode_burst_yield",
    }
    for feature, name in names.items():
        options = dict(all_on)
        options[feature] = False
        variants[name] = options
    return variants


def build_arrival_trace(num_requests: int, mode: str, rate: float, seed: int) -> list[float]:
    if mode == "burst":
        return [0.0] * num_requests
    if rate <= 0:
        raise ValueError("arrival rate must be positive for constant/poisson modes")
    if mode == "constant":
        return [index / rate for index in range(num_requests)]
    rng = random.Random(seed)
    arrivals = [0.0]
    for _ in range(1, num_requests):
        arrivals.append(arrivals[-1] + rng.expovariate(rate))
    return arrivals


def build_ttft_slo_targets(num_requests: int, min_ms: float, max_ms: float,
                           seed: int) -> list[float]:
    if (not isfinite(min_ms) or not isfinite(max_ms)
            or min_ms <= 0 or max_ms < min_ms):
        raise ValueError("TTFT target range must satisfy 0 < min <= max")
    if min_ms == max_ms:
        return [min_ms] * num_requests
    rng = random.Random(seed)
    return [rng.uniform(min_ms, max_ms) for _ in range(num_requests)]


def scheduler_list(engine: LLM):
    return ((engine.prefill_scheduler, engine.decode_scheduler) if engine._pd
            else (engine.scheduler,))


def _step_stats_empty():
    return {
        "prefill_steps": 0,
        "decode_steps": 0,
        "prefill_tokens": 0,
        "decode_tokens": 0,
        "prefill_time": 0.0,
        "decode_time": 0.0,
        "decode_iterations": 0,
        "multi_step_decode_steps": 0,
        "multi_step_decode_tokens": 0,
        "decode_bursts": 0,
        "decode_burst_rounds": 0,
        "decode_burst_yields": 0,
        "decode_burst_skipped_slots": 0,
        "decode_burst_pressure_yields": 0,
    }


def run_arrival_trace(args, prompts, sampling_params, arrival_times,
                      ttft_slo_targets_ms: list[float],
                      policy: dict[str, bool]) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine = LLM(
        args.model,
        execution_mode="mixed",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        streaming_load=args.streaming_load,
        kv_swap=not args.no_swap_kv,
        kv_swap_space_gb=args.kv_swap_space_gb,
        admission_window=args.admission_window,
        aging_timeout_ms=args.aging_timeout_ms,
        prefill_reserve_tokens=args.prefill_reserve_tokens,
        default_ttft_slo_ms=500.0,
        default_tpot_slo_ms=args.slo_tpot_ms,
        max_decode_steps=args.max_decode_steps,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        queue_depth_for_full_prefill=args.queue_depth_for_full_prefill,
        preempt_prefill_tokens_per_second=args.preempt_prefill_tps,
        preempt_kv_transfer_gbps=args.preempt_kv_transfer_gbps,
        **policy,
    )
    try:
        # Compile kernels and warm CUDA without inserting workload prefixes into the cache.
        warmup_count = min(args.warmup_seqs, len(prompts))
        if warmup_count:
            vocab = max(2, int(getattr(engine.tokenizer, "vocab_size", 32000)))
            warmup_prompts = []
            for index, prompt in enumerate(prompts[:warmup_count]):
                base = (vocab - 1 - index * 997) % vocab
                warmup_prompts.append([(base - offset * 7919) % vocab
                                       for offset in range(len(prompt))])
            engine.generate(
                warmup_prompts,
                SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8),
                use_tqdm=False,
            )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        engine._req_metrics = []
        engine._step_stats = engine._empty_step_stats()
        for scheduler in scheduler_list(engine):
            scheduler.reset_metrics()
            # Keep the online estimator's starting point identical across variants;
            # warmup timings are excluded from this workload comparison.
            scheduler._prefill_seconds_per_token = 1.0 / args.preempt_prefill_tps
            scheduler._swap_seconds_per_byte = 1.0 / (
                args.preempt_kv_transfer_gbps * 1e9)

        next_request = 0
        completed = 0
        step_stats = _step_stats_empty()
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.set_decode_burst_yield_callback(
            lambda: (next_request < len(arrival_times)
                     and arrival_times[next_request] <= time.perf_counter() - start))

        while completed < len(prompts):
            now = time.perf_counter()
            elapsed = now - start
            while (next_request < len(prompts)
                   and arrival_times[next_request] <= elapsed):
                engine.add_request(
                    prompts[next_request], sampling_params[next_request],
                    submitted_at=start + arrival_times[next_request],
                    ttft_slo_ms=ttft_slo_targets_ms[next_request],
                    tpot_slo_ms=args.slo_tpot_ms,
                )
                next_request += 1

            if engine.is_finished():
                if next_request == len(prompts):
                    break
                delay = max(0.0, arrival_times[next_request] - (time.perf_counter() - start))
                time.sleep(min(delay, 0.01))
                continue

            step_started = time.perf_counter()
            finished, kind, n_prefill, n_decode = engine.step()
            duration = time.perf_counter() - step_started
            step_stats["decode_iterations"] += engine._last_step_decode_iterations
            step_stats["multi_step_decode_tokens"] += (
                engine._last_step_multistep_tokens)
            step_stats["decode_burst_pressure_yields"] += int(
                engine._last_step_decode_burst_yielded)
            if engine._last_step_decode_iterations > 1:
                step_stats["multi_step_decode_steps"] += 1
            for burst_key in ("decode_bursts", "decode_burst_rounds",
                              "decode_burst_yields", "decode_burst_skipped_slots"):
                step_stats[burst_key] += engine._step_stats.get(burst_key, 0)
            if kind == "prefill":
                step_stats["prefill_steps"] += 1
                step_stats["prefill_tokens"] += n_prefill
                step_stats["prefill_time"] += duration
            elif kind in ("decode", "spec"):
                step_stats["decode_steps"] += 1
                step_stats["decode_tokens"] += n_decode
                step_stats["decode_time"] += duration
            else:
                step_stats["prefill_steps"] += int(n_prefill > 0)
                step_stats["decode_steps"] += int(n_decode > 0)
                step_stats["prefill_tokens"] += n_prefill
                step_stats["decode_tokens"] += n_decode
                total = n_prefill + n_decode
                if total:
                    step_stats["prefill_time"] += duration * n_prefill / total
                    step_stats["decode_time"] += duration * n_decode / total
            completed += len(finished)

        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        metrics = engine.collect_metrics()
        metrics["step_stats"] = step_stats
        summary = summarize_requests(metrics, wall, args)
        return {
            "wall_seconds": wall,
            "summary": summary,
            "scheduler_metrics": {
                key: metrics[key] for key in (
                    "num_preemptions", "num_swaps", "num_recompute_preemptions",
                    "recompute_tokens", "estimated_recompute_seconds",
                    "estimated_swap_seconds", "num_affinity_probes",
                    "prefix_cache_hit_tokens", "prefix_cache_hit_requests",
                    "num_prefix_feature_parses", "num_prefix_feature_reuses",
                    "prefix_feature_invalidations", "prefix_cache_evictions",
                    "prefix_cache_lru_entries", "prefix_cache_generation",
                    "kv_generation_mutations", "deferred_free_refs_queued",
                    "deferred_free_refs_committed", "deferred_free_flushes",
                    "deferred_free_peak_blocks", "deferred_free_peak_refs",
                    "deferred_free_pending_blocks", "deferred_free_pending_refs",
                    "num_aging_promotions", "slo_prefill_steps",
                    "adaptive_prefill_tokens_avg", "adaptive_prefill_tokens_max",
                    "adaptive_prefill_rows_avg", "tpot_prefill_steps",
                    "tpot_priority_steps", "tpot_adaptive_prefill_tokens_avg",
                )
            },
            "per_request": metrics["per_request"],
            "policy": policy,
        }
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def summarize_requests(metrics: dict, wall: float, args) -> dict:
    rows = metrics["per_request"]
    ttft = [row["t_first_token"] - row["t_submitted"] for row in rows
            if row["t_first_token"] is not None and row["t_submitted"] is not None]
    queue_wait = [row["t_prefill_started"] - row["t_submitted"] for row in rows
                  if row.get("t_prefill_started") is not None
                  and row["t_submitted"] is not None]
    tpot = [(row["t_completed"] - row["t_first_token"])
            / (row["completion_tokens"] - 1) for row in rows
            if row["t_completed"] is not None and row["t_first_token"] is not None
            and row["completion_tokens"] > 1]
    e2e = [row["t_completed"] - row["t_submitted"] for row in rows
           if row["t_completed"] is not None and row["t_submitted"] is not None]
    total_output = sum(row["completion_tokens"] for row in rows)
    stats = metrics["step_stats"]
    ttft_slo_count = sum(value * 1000 <= args.slo_ttft_ms for value in ttft)
    tpot_slo_count = sum(value * 1000 <= args.slo_tpot_ms for value in tpot)
    request_slo_rows = [row for row in rows if row.get("ttft_slo_met") is not None]
    request_tpot_slo_rows = [row for row in rows if row.get("tpot_slo_met") is not None]
    prefill_tps = (stats["prefill_tokens"] / stats["prefill_time"]
                   if stats["prefill_time"] > 0 else None)
    decode_tps = (stats["decode_tokens"] / stats["decode_time"]
                  if stats["decode_time"] > 0 else None)
    feature_parses = metrics.get("num_prefix_feature_parses", 0)
    feature_reuses = metrics.get("num_prefix_feature_reuses", 0)
    feature_resolutions = feature_parses + feature_reuses
    feature_invalidations = metrics.get("prefix_feature_invalidations", 0)
    return {
        "requests_completed": len(rows),
        "output_tokens": total_output,
        "throughput_output_tok_per_s": total_output / wall if wall else 0.0,
        "ttft": summarize(ttft),
        "queue_wait": summarize(queue_wait),
        "tpot": summarize(tpot),
        "e2e": summarize(e2e),
        "prefill_throughput_tok_per_s": prefill_tps,
        "decode_throughput_tok_per_s": decode_tps,
        "ttft_slo_percent": 100.0 * ttft_slo_count / max(1, len(rows)),
        "request_ttft_slo_percent": (
            100.0 * sum(row["ttft_slo_met"] for row in request_slo_rows)
            / max(1, len(request_slo_rows))),
        "request_tpot_slo_percent": (
            100.0 * sum(row["tpot_slo_met"] for row in request_tpot_slo_rows)
            / len(request_tpot_slo_rows) if request_tpot_slo_rows else None),
        "tpot_slo_percent": (
            100.0 * tpot_slo_count / len(tpot) if tpot else None),
        "prefill_tokens": stats["prefill_tokens"],
        "decode_tokens": stats["decode_tokens"],
        "prefill_steps": stats["prefill_steps"],
        "decode_steps": stats["decode_steps"],
        "decode_iterations": stats["decode_iterations"],
        "multi_step_decode_steps": stats["multi_step_decode_steps"],
        "multi_step_decode_tokens": stats["multi_step_decode_tokens"],
        "decode_bursts": stats["decode_bursts"],
        "decode_burst_rounds": stats["decode_burst_rounds"],
        "decode_burst_rounds_per_burst": (
            stats["decode_burst_rounds"] / stats["decode_bursts"]
            if stats["decode_bursts"] else None),
        "decode_burst_yields": stats["decode_burst_yields"],
        "decode_burst_yield_rate": (
            stats["decode_burst_yields"] / stats["decode_bursts"]
            if stats["decode_bursts"] else None),
        # burst 期间"新请求到达"触发的提前收尾（服务端信号 / trace 回调）
        "decode_burst_pressure_yields": stats["decode_burst_pressure_yields"],
        "decode_burst_pressure_yield_rate": (
            stats["decode_burst_pressure_yields"] / stats["decode_bursts"]
            if stats["decode_bursts"] else None),
        "decode_burst_skipped_slots": stats["decode_burst_skipped_slots"],
        "max_queue_wait_seconds": max(queue_wait, default=0.0),
        "lifecycle": {
            "prefix_feature_parses": feature_parses,
            "prefix_feature_reuses": feature_reuses,
            "prefix_feature_resolutions": feature_resolutions,
            "prefix_feature_reuse_rate": (
                feature_reuses / feature_resolutions if feature_resolutions else None),
            "prefix_feature_invalidations": feature_invalidations,
            "prefix_feature_invalidation_rate": (
                feature_invalidations / feature_parses if feature_parses else None),
            "prefix_cache_evictions": metrics.get("prefix_cache_evictions", 0),
            "prefix_cache_lru_entries": metrics.get("prefix_cache_lru_entries", 0),
            "prefix_cache_generation": metrics.get("prefix_cache_generation", 0),
            "kv_generation_mutations": metrics.get("kv_generation_mutations", 0),
            "deferred_free_refs_queued": metrics.get("deferred_free_refs_queued", 0),
            "deferred_free_refs_committed": metrics.get(
                "deferred_free_refs_committed", 0),
            "deferred_free_flushes": metrics.get("deferred_free_flushes", 0),
            "deferred_free_peak_blocks": metrics.get("deferred_free_peak_blocks", 0),
            "deferred_free_peak_refs": metrics.get("deferred_free_peak_refs", 0),
            "deferred_free_pending_blocks": metrics.get(
                "deferred_free_pending_blocks", 0),
            "deferred_free_pending_refs": metrics.get(
                "deferred_free_pending_refs", 0),
        },
    }


def relative_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return 100.0 * (value - baseline) / baseline


def comparison_delta(summary: dict, baseline: dict) -> dict[str, float | None]:
    """Return relative throughput/latency deltas and absolute SLO point deltas."""
    result = {
        "throughput_output_tok_per_s": relative_change(
            summary["throughput_output_tok_per_s"],
            baseline["throughput_output_tok_per_s"]),
        "prefill_throughput_tok_per_s": relative_change(
            summary["prefill_throughput_tok_per_s"],
            baseline["prefill_throughput_tok_per_s"]),
        "decode_throughput_tok_per_s": relative_change(
            summary["decode_throughput_tok_per_s"],
            baseline["decode_throughput_tok_per_s"]),
        "max_queue_wait_seconds": relative_change(
            summary["max_queue_wait_seconds"], baseline["max_queue_wait_seconds"]),
    }
    for metric in ("queue_wait", "ttft", "tpot", "e2e"):
        for percentile_name in ("p50", "p99"):
            result[f"{metric}_{percentile_name}"] = relative_change(
                summary[metric][percentile_name], baseline[metric][percentile_name])
    result["ttft_slo_percent_points"] = (
        summary["ttft_slo_percent"] - baseline["ttft_slo_percent"])
    result["tpot_slo_percent_points"] = (
        summary["tpot_slo_percent"] - baseline["tpot_slo_percent"]
        if summary["tpot_slo_percent"] is not None
        and baseline["tpot_slo_percent"] is not None else None)
    result["request_ttft_slo_percent_points"] = (
        summary["request_ttft_slo_percent"] - baseline["request_ttft_slo_percent"])
    result["request_tpot_slo_percent_points"] = (
        summary["request_tpot_slo_percent"] - baseline["request_tpot_slo_percent"]
        if summary["request_tpot_slo_percent"] is not None
        and baseline["request_tpot_slo_percent"] is not None else None)
    result["decode_burst_pressure_yields"] = (
        summary["decode_burst_pressure_yields"]
        - baseline["decode_burst_pressure_yields"])
    lifecycle = {}
    for metric in (
        "prefix_feature_parses", "prefix_feature_reuses",
        "prefix_feature_invalidations", "prefix_cache_evictions",
        "kv_generation_mutations", "deferred_free_refs_queued",
        "deferred_free_refs_committed", "deferred_free_flushes",
        "deferred_free_peak_blocks", "deferred_free_peak_refs",
    ):
        lifecycle[metric] = relative_change(
            summary["lifecycle"][metric], baseline["lifecycle"][metric])
    for metric in ("prefix_feature_reuse_rate", "prefix_feature_invalidation_rate"):
        lifecycle[metric] = relative_change(
            summary["lifecycle"][metric], baseline["lifecycle"][metric])
    result["lifecycle"] = lifecycle
    return result


def parse_args():
    variants = variant_table()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(parser)
    parser.add_argument("--arrival-mode", choices=("poisson", "constant", "burst"),
                        default="poisson")
    parser.add_argument("--arrival-rate", type=float, default=16.0,
                        help="offered request rate for poisson/constant arrivals")
    parser.add_argument("--variants", nargs="+", choices=tuple(variants),
                        default=list(variants), help="policy variants to run")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8_e4m3"), default="auto")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--streaming-load", action="store_true")
    parser.add_argument("--no-swap-kv", action="store_true")
    parser.add_argument("--kv-swap-space-gb", type=float, default=2.0)
    parser.add_argument("--admission-window", type=int, default=16)
    parser.add_argument("--aging-timeout-ms", type=float, default=2000.0)
    parser.add_argument("--prefill-reserve-tokens", type=int, default=256)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=4096)
    parser.add_argument("--queue-depth-for-full-prefill", type=int, default=16)
    parser.add_argument("--preempt-prefill-tps", type=float, default=10000.0)
    parser.add_argument("--preempt-kv-transfer-gbps", type=float, default=12.0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-seqs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--min-request-ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--max-request-ttft-slo-ms", type=float, default=500.0)
    parser.add_argument("--slo-tpot-ms", type=float, default=10.0)
    parser.add_argument("--max-decode-steps", type=int, default=4,
                        help="maximum pure-decode forwards per engine step")
    parser.add_argument("--no-decode-burst-yield", action="store_true",
                        help="force decode_burst_yield off in every variant "
                             "(burst runs its full round budget past newly arrived prefills)")
    parser.add_argument("--no-decode-burst-yield-on-arrival", action="store_true",
                        help="force decode_burst_yield_on_arrival off in every variant "
                             "(requests that become due during a burst no longer end it)")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True) \
        if args.prompts_file else None
    prompts, sampling_params = build_workload(args, tokenizer)
    arrivals = build_arrival_trace(len(prompts), args.arrival_mode,
                                   args.arrival_rate, args.seed + 1)
    ttft_slo_targets_ms = build_ttft_slo_targets(
        len(prompts), args.min_request_ttft_slo_ms,
        args.max_request_ttft_slo_ms, args.seed + 2)
    variants = variant_table()
    if args.no_decode_burst_yield:
        for options in variants.values():
            options["decode_burst_yield"] = False
    if args.no_decode_burst_yield_on_arrival:
        for options in variants.values():
            options["decode_burst_yield_on_arrival"] = False
    runs = {}

    for variant_name in args.variants:
        print(f"running {variant_name} ({len(prompts)} requests, {args.arrival_mode}, "
              f"{args.arrival_rate:g} req/s) ...", flush=True)
        result = run_arrival_trace(
            args, prompts, sampling_params, arrivals, ttft_slo_targets_ms,
            variants[variant_name])
        runs[variant_name] = result
        summary = result["summary"]
        request_tpot = summary["request_tpot_slo_percent"]
        request_tpot_text = (f"{request_tpot:.1f}%" if request_tpot is not None
                             else "n/a")
        tpot_p50 = summary["tpot"]["p50"]
        tpot_p50_text = (f"{tpot_p50 * 1000:.1f} ms"
                         if tpot_p50 is not None else "n/a")
        print(f"  throughput {summary['throughput_output_tok_per_s']:.1f} tok/s | "
              f"TTFT p50/p99 {summary['ttft']['p50'] * 1000:.1f}/"
              f"{summary['ttft']['p99'] * 1000:.1f} ms | "
              f"request SLO {summary['request_ttft_slo_percent']:.1f}% | "
              f"request TPOT target {request_tpot_text} | "
              f"TPOT p50 {tpot_p50_text} | "
              f"E2E p99 {summary['e2e']['p99'] * 1000:.1f} ms | "
              f"burst pressure yields {summary['decode_burst_pressure_yields']}",
              flush=True)

    baseline = runs.get("baseline")
    if baseline:
        baseline_summary = baseline["summary"]
        for result in runs.values():
            result["delta_vs_baseline"] = comparison_delta(
                result["summary"], baseline_summary)
    all_on = runs.get("all_on")
    if all_on:
        for result in runs.values():
            result["delta_vs_all_on"] = comparison_delta(
                result["summary"], all_on["summary"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or os.path.join(
        "results", f"scheduling_ablation_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    payload = {
        "meta": {"model": args.model, "date": timestamp, **env_info()},
        "benchmark_config": {
            "execution_mode": "mixed",
            "tensor_parallel_size": args.tp,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "kv_cache_dtype": args.kv_cache_dtype,
            "quantization": args.quantization,
            "kv_swap": not args.no_swap_kv,
            "kv_swap_space_gb": args.kv_swap_space_gb,
            "admission_window": args.admission_window,
            "aging_timeout_ms": args.aging_timeout_ms,
            "prefill_reserve_tokens": args.prefill_reserve_tokens,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
            "queue_depth_for_full_prefill": args.queue_depth_for_full_prefill,
            "preempt_prefill_tokens_per_second": args.preempt_prefill_tps,
            "preempt_kv_transfer_gbps": args.preempt_kv_transfer_gbps,
            "slo_ttft_ms": args.slo_ttft_ms,
            "request_ttft_slo_min_ms": args.min_request_ttft_slo_ms,
            "request_ttft_slo_max_ms": args.max_request_ttft_slo_ms,
            "slo_tpot_ms": args.slo_tpot_ms,
            "max_decode_steps": args.max_decode_steps,
            "decode_burst_yield_on_arrival_default": True,
            "warmup_seqs": args.warmup_seqs,
        },
        "variants": {name: dict(options) for name, options in variants.items()},
        "workload": {
            "num_requests": len(prompts),
            "arrival_mode": args.arrival_mode,
            "arrival_rate_requests_per_second": args.arrival_rate,
            "arrival_times_seconds": arrivals,
            "ttft_slo_targets_ms": ttft_slo_targets_ms,
            "tpot_slo_targets_ms": [args.slo_tpot_ms] * len(prompts),
            "seed": args.seed,
            "shared_prefix_len": args.shared_prefix_len,
            "min_input_len": args.min_input_len,
            "max_input_len": args.max_input_len,
            "min_output_len": args.min_output_len,
            "max_output_len": args.max_output_len,
        },
        "runs": runs,
    }
    with open(output_path, "w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2, default=str)

    print("\nAblation comparison (latency deltas are relative; lower is better):")
    print("variant                 output tok/s / Δ  TTFT p50  TTFT p99  queue p99  TPOT p50  E2E p99 req-SLO")
    for name, result in runs.items():
        summary = result["summary"]
        delta = result.get("delta_vs_baseline", {})

        def fmt_delta(key, unit="%"):
            value = delta.get(key)
            if value is None:
                return "baseline" if name == "baseline" else "n/a"
            return f"{value:+.1f}{unit}"

        throughput = (f"{summary['throughput_output_tok_per_s']:.1f} / "
                      f"{fmt_delta('throughput_output_tok_per_s')}")
        print(f"{name:<23} {throughput:>17} "
              f"{fmt_delta('ttft_p50'):>9} {fmt_delta('ttft_p99'):>9} "
              f"{fmt_delta('queue_wait_p99'):>9} {fmt_delta('tpot_p50'):>9} "
              f"{fmt_delta('e2e_p99'):>9} "
              f"{fmt_delta('request_ttft_slo_percent_points', 'pp'):>7}")

    print("\nPrefix/KV lifecycle work by policy:")
    print("variant                 feature parse/reuse    stale   evict  gen-mut  deferred refs queued/committed peak blocks/refs")
    for name, result in runs.items():
        life = result["summary"]["lifecycle"]
        reuse_rate = life["prefix_feature_reuse_rate"]
        reuse_text = "n/a" if reuse_rate is None else f"{reuse_rate * 100:.1f}%"
        print(f"{name:<23} {life['prefix_feature_parses']:>6}/"
              f"{life['prefix_feature_reuses']:<6} ({reuse_text:>5}) "
              f"{life['prefix_feature_invalidations']:>6} "
              f"{life['prefix_cache_evictions']:>6} "
              f"{life['kv_generation_mutations']:>8} "
              f"{life['deferred_free_refs_queued']:>7}/"
              f"{life['deferred_free_refs_committed']:<7} "
              f"{life['deferred_free_peak_blocks']:>4}/"
              f"{life['deferred_free_peak_refs']:<4}")

    print("\nDecode burst behaviour by policy (rounds = model forwards per burst):")
    print("variant                 bursts  rounds  rounds/burst  queued-yield  arrival-yield  arrival%  skipped slots")
    for name, result in runs.items():
        summary = result["summary"]
        rounds_per_burst = summary["decode_burst_rounds_per_burst"]
        arrival_rate = summary["decode_burst_pressure_yield_rate"]
        rounds_text = ("n/a" if rounds_per_burst is None
                       else f"{rounds_per_burst:.2f}")
        arrival_text = "n/a" if arrival_rate is None else f"{arrival_rate * 100:.1f}%"
        print(f"{name:<23} {summary['decode_bursts']:>6} "
              f"{summary['decode_burst_rounds']:>7} {rounds_text:>13} "
              f"{summary['decode_burst_yields']:>13} "
              f"{summary['decode_burst_pressure_yields']:>14} {arrival_text:>9} "
              f"{summary['decode_burst_skipped_slots']:>14}")

    if all_on:
        print("\nEffect of removing one policy (relative to all_on; lower latency is better):")
        for name, result in runs.items():
            if not name.startswith("without_"):
                continue
            delta = result.get("delta_vs_all_on", {})
            throughput_delta = delta.get("throughput_output_tok_per_s")
            ttft_delta = delta.get("ttft_p99")
            slo_delta = delta.get("request_ttft_slo_percent_points")
            burst_yield_delta = delta.get("decode_burst_pressure_yields")
            fmt = lambda value, unit="%": "n/a" if value is None else f"{value:+.1f}{unit}"
            print(f"  {name:<38} throughput {fmt(throughput_delta)} | "
                  f"TTFT p99 {fmt(ttft_delta)} | request SLO {fmt(slo_delta, ' pp')} | "
                  f"burst-yield count Δ {fmt(burst_yield_delta, ' yields')}")
    print(f"\nfull per-request results -> {output_path}")


if __name__ == "__main__":
    main()
