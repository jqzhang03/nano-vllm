"""Throughput / latency benchmark for nano-vllm.

Measures wall-clock throughput, per-request TTFT / TPOT / E2E latency with
p50/p99 percentiles, SLO attainment, and an optional side-by-side comparison
against the real vLLM. Three workload modes:

  * synthetic (default): random token ids, input/output lengths sampled
    uniformly from configurable ranges;
  * shared prefix (--shared-prefix-len N): every prompt starts with the same
    N tokens, exercising the prefix cache;
  * real prompts (--prompts-file): JSONL file with one {"prompt": "..."} per
    line, tokenized with the model's tokenizer.

Run from the WSL conda env (see INTERVIEW.md §10):

    python benchmarks/bench.py --num-seqs 256
    python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512
    python benchmarks/bench.py --num-seqs 256 --compare-vllm
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import torch

from nanovllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# workload construction (shared with profile.py)
# ---------------------------------------------------------------------------

def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-seqs", type=int, default=256,
                        help="request count; 0 consumes every row in --prompts-file")
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--shared-prefix-len", type=int, default=0,
                        help="common prefix shared by all prompts (prefix-cache workload)")
    parser.add_argument("--prompts-file", default=None,
                        help="JSONL prompt/request trace; supports prompt, max_tokens, "
                             "arrival_time_s, request_id, prefix_group, SLO targets, "
                             "and prompt_token_hash. Legacy {\"prompt\": ...} is supported")


@dataclass(slots=True)
class WorkloadData:
    prompts: list[list[int]]
    sampling_params: list[SamplingParams]
    requests: list[dict[str, Any]]
    source: str
    has_arrival_trace: bool


def prompt_token_hash(token_ids: list[int]) -> str:
    """Return a stable hash for one concrete tokenizer output."""
    encoded = json.dumps(token_ids, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def input_length_bucket(token_count: int) -> str:
    if token_count <= 256:
        return "1-256"
    if token_count <= 512:
        return "257-512"
    if token_count <= 1024:
        return "513-1024"
    if token_count <= 2048:
        return "1025-2048"
    return ">2048"


def _finite_positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        condition = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {condition}")
    return result


def _read_jsonl_prompts(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            if "prompt" not in row or not isinstance(row["prompt"], str):
                raise ValueError(f"JSONL row {line_number} needs a string prompt")
            rows.append(row)
    if not rows:
        raise ValueError(f"prompt file is empty: {path}")
    return rows


def build_workload_data(args, tokenizer=None) -> WorkloadData:
    """Build tokenized requests and retain optional trace metadata.

    Request-trace files are consumed in order and must match ``--num-seqs``;
    use zero to consume all rows. Legacy prompt-only files retain seeded
    sampling and synthetic output-length behavior.
    """
    rng = random.Random(args.seed)
    source = "synthetic"
    requests: list[dict[str, Any]] = []
    has_arrival_trace = False
    if args.prompts_file:
        if tokenizer is None:
            raise ValueError("--prompts-file requires a tokenizer")
        rows = _read_jsonl_prompts(args.prompts_file)
        trace_fields = {
            "request_id", "max_tokens", "arrival_time_s", "prefix_group",
            "ttft_slo_ms", "tpot_slo_ms", "prompt_token_hash",
            "prompt_prefix_token_ids", "content_class",
        }
        has_request_trace = any(any(field in row for field in trace_fields)
                                for row in rows)
        if has_request_trace:
            if args.num_seqs < 0:
                raise ValueError("--num-seqs must be non-negative")
            if args.num_seqs not in (0, len(rows)):
                raise ValueError(
                    f"request trace has {len(rows)} rows but --num-seqs is "
                    f"{args.num_seqs}; pass the exact count or 0 to use all rows")
            source = "request_trace"
        else:
            if args.num_seqs < 0:
                raise ValueError("--num-seqs must be non-negative")
            if args.num_seqs and len(rows) > args.num_seqs:
                rows = rng.sample(rows, args.num_seqs)
            source = "prompt_jsonl"

        arrival_presence = ["arrival_time_s" in row for row in rows]
        if any(arrival_presence) and not all(arrival_presence):
            raise ValueError("arrival_time_s must be present on every row or none")
        has_arrival_trace = bool(rows) and all(arrival_presence)
        if has_arrival_trace:
            previous_arrival = -1.0
            for index, row in enumerate(rows):
                arrival = _finite_positive(row["arrival_time_s"],
                                           f"row {index} arrival_time_s",
                                           allow_zero=True)
                if arrival < previous_arrival:
                    raise ValueError("arrival_time_s values must be non-decreasing")
                previous_arrival = arrival

        prompts = []
        sampling_params = []
        seen_request_ids = set()
        for index, row in enumerate(rows):
            body_tokens = list(tokenizer.encode(row["prompt"]))
            prefix_value = row.get("prompt_prefix_token_ids", [])
            if not isinstance(prefix_value, list) or any(
                    not isinstance(token_id, int) or token_id < 0
                    for token_id in prefix_value):
                raise ValueError(
                    f"row {index} prompt_prefix_token_ids must be non-negative integers")
            if prefix_value and args.shared_prefix_len:
                raise ValueError(
                    "do not combine prompt_prefix_token_ids with --shared-prefix-len")
            prefix_tokens = list(prefix_value)
            prompt_tokens = prefix_tokens + body_tokens
            expected_hash = row.get("prompt_token_hash")
            actual_hash = prompt_token_hash(prompt_tokens)
            if expected_hash is not None and expected_hash != actual_hash:
                raise ValueError(
                    f"row {index} prompt_token_hash mismatch; check tokenizer/model version")
            if not prompt_tokens:
                raise ValueError(f"row {index} tokenizes to an empty prompt")
            max_tokens_value = row.get("max_tokens")
            if max_tokens_value is None:
                max_tokens_value = rng.randint(args.min_output_len,
                                               args.max_output_len)
            if (not isinstance(max_tokens_value, int)
                    or isinstance(max_tokens_value, bool) or max_tokens_value < 1):
                raise ValueError(f"row {index} max_tokens must be a positive integer")
            request_id = str(row.get("request_id", f"request-{index:06d}"))
            if not request_id or request_id in seen_request_ids:
                raise ValueError(f"request_id is empty or duplicated: {request_id!r}")
            seen_request_ids.add(request_id)
            ttft_target = row.get("ttft_slo_ms")
            tpot_target = row.get("tpot_slo_ms")
            if ttft_target is not None:
                ttft_target = _finite_positive(
                    ttft_target, f"row {index} ttft_slo_ms")
            if tpot_target is not None:
                tpot_target = _finite_positive(
                    tpot_target, f"row {index} tpot_slo_ms")
            if ttft_target is not None:
                ttft_presence = row.get("ttft_slo_ms") is not None
            prompts.append(prompt_tokens)
            sampling_params.append(SamplingParams(
                temperature=float(row.get("temperature", 0.6)),
                ignore_eos=bool(row.get("ignore_eos", True)),
                max_tokens=max_tokens_value,
            ))
            requests.append({
                "request_id": request_id,
                "prompt_token_hash": actual_hash,
                "prompt_tokens": len(prompt_tokens),
                "user_prompt_tokens": len(body_tokens),
                "prefix_tokens": len(prefix_tokens),
                "input_length_bucket": input_length_bucket(len(prompt_tokens)),
                "prefix_group": row.get("prefix_group"),
                "prefix_kind": row.get("prefix_kind"),
                "content_class": row.get("content_class"),
                "arrival_time_s": (float(row["arrival_time_s"])
                                   if has_arrival_trace else None),
                "arrival_mode": row.get("arrival_mode"),
                "arrival_rate_requests_per_second": row.get(
                    "arrival_rate_requests_per_second"),
                "max_tokens": max_tokens_value,
                "ttft_slo_ms": ttft_target,
                "tpot_slo_ms": tpot_target,
            })
    else:
        if args.num_seqs < 1:
            raise ValueError("--num-seqs must be positive for a synthetic workload")
        prompts = [rng.sample(range(0, 10000), rng.randint(args.min_input_len, args.max_input_len))
                   for _ in range(args.num_seqs)]
        sampling_params = [SamplingParams(
            temperature=0.6, ignore_eos=True,
            max_tokens=rng.randint(args.min_output_len, args.max_output_len))
            for _ in prompts]
        requests = [{
            "request_id": f"request-{index:06d}",
            "prompt_token_hash": prompt_token_hash(prompt),
            "prompt_tokens": len(prompt),
            "user_prompt_tokens": len(prompt),
            "prefix_tokens": 0,
            "input_length_bucket": input_length_bucket(len(prompt)),
            "prefix_group": None,
            "prefix_kind": None,
            "content_class": None,
            "arrival_time_s": None,
            "arrival_mode": None,
            "arrival_rate_requests_per_second": None,
            "max_tokens": params.max_tokens,
            "ttft_slo_ms": None,
            "tpot_slo_ms": None,
        } for index, (prompt, params) in enumerate(zip(prompts, sampling_params))]

    if args.shared_prefix_len > 0:
        if args.prompts_file and any(row["prefix_tokens"] for row in requests):
            raise ValueError(
                "do not combine trace prefixes with --shared-prefix-len")
        prefix = rng.sample(range(0, 10000), args.shared_prefix_len)
        prompts = [prefix + prompt for prompt in prompts]
        for index, (prompt, request) in enumerate(zip(prompts, requests)):
            request["prefix_tokens"] = len(prefix)
            request["prompt_tokens"] = len(prompt)
            request["prompt_token_hash"] = prompt_token_hash(prompt)
            request["input_length_bucket"] = input_length_bucket(len(prompt))
            request["prefix_group"] = "synthetic_shared_prefix"
            request["prefix_kind"] = "synthetic_token_ids"

    return WorkloadData(prompts, sampling_params, requests, source,
                        has_arrival_trace)


def build_workload(args, tokenizer=None):
    """Backward-compatible (prompts, sampling_params) workload helper."""
    workload = build_workload_data(args, tokenizer)
    return workload.prompts, workload.sampling_params


def scheduler_options(args, overrides=None):
    """Build the scheduler ablation options shared by bench entry points."""
    options = {
        "latency_aware_scheduling": not getattr(
            args, "no_latency_aware_scheduling", False),
        "cache_affinity_admission": not getattr(
            args, "no_cache_affinity_admission", False),
        "prefix_feature_cache": not getattr(
            args, "no_prefix_feature_cache", False),
        "aging_fairness": not getattr(args, "no_aging_fairness", False),
        "recompute_aware_preemption": not getattr(
            args, "no_recompute_aware_preemption", False),
        "admission_window": getattr(args, "admission_window", 16),
        "aging_timeout_ms": getattr(args, "aging_timeout_ms", 2000.0),
        "prefill_reserve_tokens": getattr(args, "prefill_reserve_tokens", 256),
        "slo_aware_scheduling": not getattr(args, "no_slo_aware_scheduling", False),
        "default_ttft_slo_ms": getattr(args, "default_ttft_slo_ms", 500.0),
        "tpot_aware_scheduling": not getattr(args, "no_tpot_aware_scheduling", False),
        "default_tpot_slo_ms": getattr(args, "default_tpot_slo_ms", None),
        "tpot_decode_ms_fallback": getattr(args, "tpot_decode_ms_fallback", 20.0),
        "tpot_prefill_throttle_margin": getattr(
            args, "tpot_prefill_throttle_margin", 0.5),
        "tpot_throttle_max_waiting": getattr(args, "tpot_throttle_max_waiting", 16),
        "multi_step_decode": not getattr(args, "no_multi_step_decode", False),
        "max_decode_steps": getattr(args, "max_decode_steps", 4),
        "decode_burst_yield": not getattr(args, "no_decode_burst_yield", False),
        "decode_burst_yield_on_arrival": not getattr(
            args, "no_decode_burst_yield_on_arrival", False),
        "max_prefill_chunk_tokens": getattr(args, "max_prefill_chunk_tokens", 4096),
        "queue_depth_for_full_prefill": getattr(
            args, "queue_depth_for_full_prefill", 16),
        "preempt_prefill_tokens_per_second": getattr(
            args, "preempt_prefill_tps", 10000.0),
        "preempt_kv_transfer_gbps": getattr(
            args, "preempt_kv_transfer_gbps", 12.0),
    }
    if overrides:
        options.update(overrides)
    return options


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------

def percentile(sorted_vals, q):
    return sorted_vals[min(int(len(sorted_vals) * q), len(sorted_vals) - 1)]


def summarize(values):
    """values: iterable of float | None. Returns avg/p50/p99/min/max/count."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"avg": None, "p50": None, "p99": None, "min": None, "max": None, "count": 0}
    return {
        "avg": statistics.fmean(vals),
        "p50": percentile(vals, 0.50),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "min": vals[0],
        "max": vals[-1],
        "count": len(vals),
    }


def _fmt_seconds(v):
    return "n/a" if v is None else f"{v * 1000:.1f}ms"


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def run_nanovllm(args, prompts, sampling_params):
    llm = LLM(args.model, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              kv_cache_dtype=getattr(args, "kv_cache_dtype", "auto"),
              kv_calibration_path=getattr(args, "kv_calibration_path", ""),
              kv_fp8_scale_margin=getattr(args, "kv_fp8_scale_margin", 1.1),
              quantization=getattr(args, "quantization", "none"),
              nvfp4_backend=getattr(args, "nvfp4_backend", "software"),
              awq_scales_path=getattr(args, "awq_scales_path", ""),
              quantize_lm_head=getattr(args, "quantize_lm_head", False),
              speculative=getattr(args, "speculative", "none"),
              streaming_load=getattr(args, "streaming_load", False),
              int4_dense_path=not getattr(args, "no_int4_dense_path", False),
              int4_group_size=getattr(args, "int4_group_size", 128),
              kv_swap=not getattr(args, "no_swap_kv", False),
              kv_swap_space_gb=getattr(args, "kv_swap_space_gb", 2.0),
              mixed_cudagraph=not getattr(args, "no_mixed_cudagraph", False),
              mixed_cudagraph_max_graphs=getattr(args, "mixed_cudagraph_max_graphs", 4),
              mixed_cudagraph_max_tokens=getattr(args, "mixed_cudagraph_max_tokens", 4096),
              prefix_cache_max_free_blocks=getattr(args, "prefix_cache_max_free_blocks", 0),
              **scheduler_options(args))
    # 预热：触发torch.compile/triton的JIT编译、分配KV Cache、捕获CUDA Graph。
    # 用真实 workload prompts（而非 3-token 的 "warm up"）——fp8 的融合量化/硬件MMA
    # 内核按 K 形状一次性 JIT 编译（~100-400ms），必须落在预热里而不是计时区间
    llm.generate(prompts[: min(args.warmup_seqs, len(prompts))],
                 SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    batches = []
    for i in range(args.repeat_batches):
        t0 = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        wall = time.perf_counter() - t0
        batches.append({"batch": i, "wall": wall, **llm.collect_metrics()})
    cfg = llm.model_runner.config
    kv_info = {
        "num_kvcache_blocks": cfg.num_kvcache_blocks,
        "kvcache_block_size": cfg.kvcache_block_size,
        "capacity_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size,
    }
    return batches, kv_info


def run_vllm(args, prompts, sampling_params):
    """Run the same workload on the real vLLM (best effort). Returns None if unavailable."""
    try:
        from vllm import LLM as VLLM
        from vllm import SamplingParams as VSP
    except ImportError:
        return None
    try:
        llm = VLLM(args.model, enforce_eager=args.enforce_eager, tensor_parallel_size=args.tp,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
        llm.generate(["warm up"] * args.warmup_seqs, VSP(temperature=0.6, max_tokens=8))
        vprompts = [dict(prompt_token_ids=p) for p in prompts]
        vsps = [VSP(temperature=0.6, max_tokens=sp.max_tokens, ignore_eos=True) for sp in sampling_params]
        t0 = time.perf_counter()
        outputs = llm.generate(vprompts, vsps)
        wall = time.perf_counter() - t0
        rows = []
        for out in outputs:
            comp = len(out.outputs[0].token_ids)
            m = getattr(out, "metrics", None)  # vLLM >= 0.6: RequestOutput.metrics
            if m is not None and getattr(m, "first_token_time", None) is not None:
                rows.append({
                    "ttft": m.first_token_time - m.arrival_time,
                    "e2e": m.finished_time - m.arrival_time,
                    "completion_tokens": comp,
                })
        return {"wall": wall, "total_completion_tokens": sum(len(o.outputs[0].token_ids) for o in outputs),
                "per_request": rows}
    except Exception as e:  # vLLM版本/配置差异时给出提示而非中断
        print(f"[warn] vLLM comparison failed: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def make_summary(metrics):
    rows = metrics["per_request"]
    ttfts = [r["t_first_token"] - r["t_submitted"] for r in rows
             if r["t_submitted"] is not None and r["t_first_token"] is not None]
    tpots = [(r["t_completed"] - r["t_first_token"]) / (r["completion_tokens"] - 1) for r in rows
             if r["t_completed"] is not None and r["t_first_token"] is not None and r["completion_tokens"] > 1]
    e2es = [r["t_completed"] - r["t_submitted"] for r in rows
            if r["t_completed"] is not None and r["t_submitted"] is not None]
    queue_waits = [r["t_prefill_started"] - r["t_submitted"] for r in rows
                   if r.get("t_prefill_started") is not None and r["t_submitted"] is not None]
    stats = metrics["step_stats"]
    prefill_tps = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else None
    decode_tps = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else None
    request_slo_rows = [row for row in rows if row.get("ttft_slo_met") is not None]
    request_tpot_slo_rows = [row for row in rows if row.get("tpot_slo_met") is not None]
    feature_parses = metrics.get("num_prefix_feature_parses", 0)
    feature_reuses = metrics.get("num_prefix_feature_reuses", 0)
    feature_resolutions = feature_parses + feature_reuses
    feature_invalidations = metrics.get("prefix_feature_invalidations", 0)
    return {
        "ttft": summarize(ttfts),
        "tpot": summarize(tpots),
        "e2e": summarize(e2es),
        "queue_wait": summarize(queue_waits),
        "prefill_throughput_tok_per_s": prefill_tps,
        "decode_throughput_tok_per_s": decode_tps,
        "prefill_steps": stats["prefill_steps"],
        "decode_steps": stats["decode_steps"],
        "decode_iterations": stats.get("decode_iterations", stats["decode_steps"]),
        "multi_step_decode_steps": stats.get("multi_step_decode_steps", 0),
        "multi_step_decode_tokens": stats.get("multi_step_decode_tokens", 0),
        "decode_burst_pressure_yields": stats.get("decode_burst_pressure_yields", 0),
        "prefill_tokens": stats["prefill_tokens"],
        "decode_tokens": stats["decode_tokens"],
        "num_preemptions": metrics.get("num_preemptions", 0),
        "num_swaps": metrics.get("num_swaps", 0),
        "num_recompute_preemptions": metrics.get("num_recompute_preemptions", 0),
        "recompute_tokens": metrics.get("recompute_tokens", 0),
        "estimated_recompute_seconds": metrics.get("estimated_recompute_seconds", 0.0),
        "estimated_swap_seconds": metrics.get("estimated_swap_seconds", 0.0),
        "prefix_cache_hit_tokens": metrics.get("prefix_cache_hit_tokens", 0),
        "prefix_cache_hit_requests": metrics.get("prefix_cache_hit_requests", 0),
        "num_prefix_feature_parses": feature_parses,
        "num_prefix_feature_reuses": feature_reuses,
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
        "deferred_free_refs_committed": metrics.get("deferred_free_refs_committed", 0),
        "deferred_free_flushes": metrics.get("deferred_free_flushes", 0),
        "deferred_free_peak_blocks": metrics.get("deferred_free_peak_blocks", 0),
        "deferred_free_peak_refs": metrics.get("deferred_free_peak_refs", 0),
        "deferred_free_pending_blocks": metrics.get("deferred_free_pending_blocks", 0),
        "deferred_free_pending_refs": metrics.get("deferred_free_pending_refs", 0),
        "mixed_cudagraph": metrics.get("mixed_cudagraph", {}),
        "num_affinity_probes": metrics.get("num_affinity_probes", 0),
        "num_aging_promotions": metrics.get("num_aging_promotions", 0),
        "request_ttft_slo_percent": (
            100.0 * sum(row["ttft_slo_met"] for row in request_slo_rows)
            / max(1, len(request_slo_rows))),
        "request_tpot_slo_percent": (
            100.0 * sum(row["tpot_slo_met"] for row in request_tpot_slo_rows)
            / len(request_tpot_slo_rows) if request_tpot_slo_rows else None),
        "slo_prefill_steps": metrics.get("slo_prefill_steps", 0),
        "tpot_prefill_steps": metrics.get("tpot_prefill_steps", 0),
        "tpot_priority_steps": metrics.get("tpot_priority_steps", 0),
        "tpot_adaptive_prefill_tokens_avg": metrics.get(
            "tpot_adaptive_prefill_tokens_avg", 0.0),
        "adaptive_prefill_tokens_avg": metrics.get("adaptive_prefill_tokens_avg", 0.0),
        "adaptive_prefill_tokens_max": metrics.get("adaptive_prefill_tokens_max", 0),
        "adaptive_prefill_rows_avg": metrics.get("adaptive_prefill_rows_avg", 0.0),
    }


def print_report(args, wall, metrics, kv_info, vllm_res=None, out_path=None):
    s = make_summary(metrics)
    request_tpot_target_configured = any(
        row.get("tpot_slo_ms") is not None for row in metrics["per_request"])
    total_out = sum(r["completion_tokens"] for r in metrics["per_request"])
    total_in = sum(r["prompt_tokens"] for r in metrics["per_request"])
    throughput = total_out / wall

    def line(name, val):
        print(f"  {name:<22} {val}")

    print("=" * 72)
    print(f"nano-vllm benchmark — {os.path.basename(args.model.rstrip('/'))} "
          f"({torch.cuda.get_device_name(0)})")
    print(f"workload: {args.num_seqs} seqs | input {args.min_input_len}-{args.max_input_len} tok | "
          f"output {args.min_output_len}-{args.max_output_len} tok | "
          f"shared-prefix {args.shared_prefix_len} | eager={args.enforce_eager} | tp={args.tp}")
    opts = scheduler_options(args)
    line("scheduler", (f"latency={opts['latency_aware_scheduling']} "
                       f"affinity={opts['cache_affinity_admission']} "
                       f"feature-cache={opts['prefix_feature_cache']} "
                       f"aging={opts['aging_fairness']} "
                       f"recompute-aware={opts['recompute_aware_preemption']} "
                       f"SLO-aware={opts['slo_aware_scheduling']}"))
    print("-" * 72)
    if kv_info:
        line("KV cache", (f"{kv_info['num_kvcache_blocks']} blocks x {kv_info['kvcache_block_size']} tok"
                          f" = {kv_info['capacity_tokens']:,} tok capacity"))
    line("wall time", f"{wall:.2f}s")
    line("throughput (output)", f"{throughput:.1f} tok/s")
    line("prefill", (f"{s['prefill_tokens']} tok in {s['prefill_steps']} steps"
                     + (f" ({s['prefill_throughput_tok_per_s']:.0f} tok/s)" if s["prefill_throughput_tok_per_s"] else "")))
    line("decode", (f"{s['decode_tokens']} tok in {s['decode_steps']} steps"
                    + (f" ({s['decode_throughput_tok_per_s']:.0f} tok/s)" if s["decode_throughput_tok_per_s"] else "")))
    line("decode forwards", (f"{s['decode_iterations']} total; "
                              f"{s['multi_step_decode_steps']} multi-step engine steps; "
                              f"{s['multi_step_decode_tokens']} extra tokens; "
                              f"{s['decode_burst_pressure_yields']} pressure yields"))
    line("preemptions", f"{s['num_preemptions']} (KV cache 不足导致的抢占；0 表示容量充足)")
    line("kv swaps", f"{s['num_swaps']} (KV 换出到 CPU 的次数；swap 免重新prefill)")
    line("recompute preempts", f"{s['num_recompute_preemptions']} ({s['recompute_tokens']} estimated tokens)")
    line("prefix cache", f"{s['prefix_cache_hit_tokens']} hit tokens / "
                         f"{s['prefix_cache_hit_requests']} of {s['num_affinity_probes']} admissions")
    reuse_rate = (f"{s['prefix_feature_reuse_rate'] * 100:.1f}% reuse" if
                  s["prefix_feature_reuse_rate"] is not None else "n/a reuse")
    invalidation_rate = (f"{s['prefix_feature_invalidation_rate'] * 100:.1f}% stale reparses" if
                         s["prefix_feature_invalidation_rate"] is not None
                         else "n/a stale reparses")
    line("prefix feature work", (f"{s['num_prefix_feature_parses']} parses / "
                                  f"{s['num_prefix_feature_reuses']} cached reads; "
                                  f"{reuse_rate}; {invalidation_rate}"))
    line("prefix cache lifecycle", (f"{s['prefix_cache_evictions']} evictions; "
                                     f"{s['prefix_cache_lru_entries']} indexed; "
                                     f"{s['kv_generation_mutations']} generation mutations"))
    line("deferred free lifecycle", (f"{s['deferred_free_refs_committed']}/"
                                      f"{s['deferred_free_refs_queued']} refs committed in "
                                      f"{s['deferred_free_flushes']} flushes; peak "
                                      f"{s['deferred_free_peak_blocks']} blocks/"
                                      f"{s['deferred_free_peak_refs']} refs; currently "
                                      f"{s['deferred_free_pending_blocks']} blocks/"
                                      f"{s['deferred_free_pending_refs']} refs"))
    graph = s["mixed_cudagraph"]
    line("mixed CUDA Graph", (f"enabled={graph.get('enabled', False)} "
                              f"captures={graph.get('captures', 0)} "
                              f"replays={graph.get('replays', 0)} "
                              f"eager-fallbacks={graph.get('eager_fallbacks', 0)} "
                              f"shapes={graph.get('cached_shapes', 0)}"))
    line("aging promotions", s["num_aging_promotions"])
    line("request TTFT SLO", f"{s['request_ttft_slo_percent']:.1f}% met")
    line("request TPOT SLO", (
        f"{s['request_tpot_slo_percent']:.1f}% met"
        if s["request_tpot_slo_percent"] is not None else
        "no eligible request" if request_tpot_target_configured else "not configured"))
    line("adaptive prefill", (f"avg {s['adaptive_prefill_tokens_avg']:.0f} / "
                              f"max {s['adaptive_prefill_tokens_max']} tokens; "
                              f"{s['adaptive_prefill_rows_avg']:.1f} rows/step"))
    line("queue wait", f"p50 {_fmt_seconds(s['queue_wait']['p50'])} | "
                      f"p99 {_fmt_seconds(s['queue_wait']['p99'])}")
    line("TTFT", f"avg {_fmt_seconds(s['ttft']['avg'])} | p50 {_fmt_seconds(s['ttft']['p50'])} | "
                 f"p99 {_fmt_seconds(s['ttft']['p99'])} (n={s['ttft']['count']})")
    line("TPOT", f"avg {_fmt_seconds(s['tpot']['avg'])} | p50 {_fmt_seconds(s['tpot']['p50'])} | "
                 f"p99 {_fmt_seconds(s['tpot']['p99'])} (n={s['tpot']['count']})")
    line("E2E", f"avg {_fmt_seconds(s['e2e']['avg'])} | p50 {_fmt_seconds(s['e2e']['p50'])} | "
                f"p99 {_fmt_seconds(s['e2e']['p99'])} (n={s['e2e']['count']})")
    tpot_eligible = [
        r for r in metrics["per_request"]
        if r["t_completed"] is not None and r["t_first_token"] is not None
        and r["completion_tokens"] > 1
    ]
    tpot_met = sum(
        (r["t_completed"] - r["t_first_token"])
        / (r["completion_tokens"] - 1) * 1000 <= args.slo_tpot_ms
        for r in tpot_eligible
    )
    tpot_slo_text = (
        f"{100 * tpot_met / len(tpot_eligible):.1f}% (n={len(tpot_eligible)})"
        if tpot_eligible else "n/a (no request emitted at least 2 tokens)"
    )
    ttft_met = sum(
        r["t_first_token"] is not None and r["t_submitted"] is not None
        and (r["t_first_token"] - r["t_submitted"]) * 1000 <= args.slo_ttft_ms
        for r in metrics["per_request"]
    )
    line("SLO", f"TTFT<={args.slo_ttft_ms:.0f}ms: "
                f"{100 * ttft_met / max(1, len(metrics['per_request'])):.1f}% | "
                f"TPOT<={args.slo_tpot_ms:.0f}ms: {tpot_slo_text}")
    if vllm_res is not None:
        print("-" * 72)
        print("vLLM reference:")
        line("throughput (output)", f"{vllm_res['total_completion_tokens'] / vllm_res['wall']:.1f} tok/s")
        if vllm_res["per_request"]:
            ttft = summarize(r["ttft"] for r in vllm_res["per_request"])
            line("TTFT", f"p50 {_fmt_seconds(ttft['p50'])} | p99 {_fmt_seconds(ttft['p99'])} (n={ttft['count']})")
    print("-" * 72)
    print(f"results -> {out_path}")
    print("=" * 72)


def env_info():
    info = {
        "python": __import__("platform").python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1) if torch.cuda.is_available() else None,
    }
    for module_name, key in (("triton", "triton"),
                             ("transformers", "transformers")):
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", None)
        except ImportError:
            info[key] = None
    try:
        import flash_attn
        info["flash_attn"] = flash_attn.__version__
    except ImportError:
        info["flash_attn"] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True,
            text=True, cwd=os.path.dirname(os.path.dirname(__file__)),
        ).stdout
        info["git_revision"] = revision
        info["git_worktree_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        info["git_revision"] = None
        info["git_worktree_dirty"] = None
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,clocks.current.sm,"
             "clocks.current.graphics,clocks.current.memory,power.draw,memory.used,"
             "memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout.splitlines()
        if query:
            values = next(csv.reader([query[0]]))
            keys = ("name", "temperature_c", "sm_clock_mhz", "graphics_clock_mhz",
                    "memory_clock_mhz", "power_w", "memory_used_mib",
                    "memory_total_mib", "utilization_percent")
            snapshot = {}
            for key, value in zip(keys, values):
                value = value.strip()
                try:
                    snapshot[key] = float(value)
                    if snapshot[key].is_integer():
                        snapshot[key] = int(snapshot[key])
                except ValueError:
                    snapshot[key] = value or None
            info["gpu_snapshot_at_report"] = snapshot
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            StopIteration, csv.Error):
        info["gpu_snapshot_at_report"] = None
    return info


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(p)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--no-mixed-cudagraph", action="store_true",
                   help="ablation: run ordinary mixed prefill/decode batches eagerly")
    p.add_argument("--mixed-cudagraph-max-graphs", type=int, default=4,
                   help="maximum lazily captured mixed batch shapes to retain")
    p.add_argument("--mixed-cudagraph-max-tokens", type=int, default=4096,
                   help="use eager execution for larger mixed batches to bound capture memory")
    p.add_argument("--prefix-cache-max-free-blocks", type=int, default=0,
                   help="idle prefix cache cap; 0 lets the allocator reclaim by LRU")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="KV缓存dtype: auto(模型dtype) 或 fp8_e4m3(FP8量化，容量翻倍)")
    p.add_argument("--kv-calibration-path", default="",
                   help="FP8 KV校准token IDs JSON（kv_fp8_calibrate.py产出）")
    p.add_argument("--kv-fp8-scale-margin", type=float, default=1.1,
                   help="FP8 KV校准最大值的scale安全因子；小于1会增加饱和裁剪")
    p.add_argument("--quantization", default="none",
                   help="权重量化: none | w8a8(int8, Triton GEMM) | int4(per-group int4, Triton GEMM) | "
                        "nvfp4(E2M1 + E4M3 scales; software or native cuBLASLt MMA) | "
                        "awq(int4+激活感知缩放) | sparse24(2:4结构化剪枝, Triton稀疏GEMM) | "
                        "fp8(e4m3全量化: per-column权重+per-token激活, Triton内核+硬件FP8 MMA)")
    p.add_argument("--nvfp4-backend", choices=("software", "cublaslt"), default="software",
                   help="NVFP4 GEMM backend; cublaslt dynamically quantizes activations and "
                        "requires Blackwell native FP4 MMA")
    p.add_argument("--awq-scales-path", default="",
                   help="AWQ缩放文件（benchmarks/awq_calibrate.py产出）；空=随机token内联校准")
    p.add_argument("--quantize-lm-head", action="store_true",
                   help="同时量化LM head（默认不量化，见INTERVIEW.md §10.3.6）")
    p.add_argument("--no-int4-dense-path", action="store_true",
                   help="int4 关闭双路径模式（纯 int4 显存模式，0.85GB；吞吐回退见 INTERVIEW.md §10.3.6；"
                        "流式加载(7B+)自动强制关闭）")
    p.add_argument("--int4-group-size", type=int, default=128,
                   help="INT4 K 维分组大小；DeepSeek-V2-Lite 的 10944 维需设为 64")
    p.add_argument("--no-swap-kv", action="store_true",
                   help="关闭KV swap抢占（KV块不足时回退重算 recompute 而非换出到CPU；"
                        "swap 支持 TP=1 的 auto 与 fp8_e4m3 KV；TP>1 时回退重算）")
    p.add_argument("--kv-swap-space-gb", type=float, default=2.0,
                   help="KV swap 的 CPU 缓冲空间上限（GB；换出累计超限回落 recompute）")
    p.add_argument("--no-latency-aware-scheduling", action="store_true",
                   help="关闭prefill预算保留与短剩余prefill优先策略")
    p.add_argument("--no-cache-affinity-admission", action="store_true",
                   help="关闭Top-W前缀缓存亲和准入")
    p.add_argument("--no-prefix-feature-cache", action="store_true",
                   help="消融generation标记的懒加载特征缓存，每次Top-W查询都重新解析")
    p.add_argument("--no-aging-fairness", action="store_true",
                   help="关闭等待时长aging公平提升")
    p.add_argument("--no-recompute-aware-preemption", action="store_true",
                   help="关闭成本估算，恢复可swap时优先swap的旧策略")
    p.add_argument("--admission-window", type=int, default=16,
                   help="Top-W admission窗口大小")
    p.add_argument("--aging-timeout-ms", type=float, default=2000.0,
                   help="请求等待达到该毫秒数后触发aging提升")
    p.add_argument("--prefill-reserve-tokens", type=int, default=256,
                   help="混合调度给等待prefill保留的token预算")
    p.add_argument("--no-slo-aware-scheduling", action="store_true",
                   help="关闭TTFT目标感知与自适应prefill分块/批次配额")
    p.add_argument("--default-ttft-slo-ms", type=float, default=500.0,
                   help="未单独指定时每个请求使用的TTFT目标（毫秒）")
    p.add_argument("--default-tpot-slo-ms", type=float, default=None,
                   help="未单独指定时每个请求使用的平均TPOT目标（毫秒）")
    p.add_argument("--no-tpot-aware-scheduling", action="store_true",
                   help="关闭TPOT目标对decode优先级与prefill配额的影响")
    p.add_argument("--tpot-decode-ms-fallback", type=float, default=20.0,
                   help="请求尚无decode样本时估算的每token延迟（毫秒）")
    p.add_argument("--tpot-prefill-throttle-margin", type=float, default=0.5,
                   help="TPOT 压力压缩 prefill 配额的死区比例（0=旧行为：任何超出都饱和）")
    p.add_argument("--tpot-throttle-max-waiting", type=int, default=16,
                   help="等待队列达到该深度后停用 TPOT 对 prefill 的压缩（0=不设门限）")
    p.add_argument("--no-multi-step-decode", action="store_true",
                   help="关闭纯decode多步执行，作为消融对照")
    p.add_argument("--max-decode-steps", type=int, default=4,
                   help="每个engine step最多连续执行的纯decode轮数")
    p.add_argument("--no-decode-burst-yield", action="store_true",
                   help="关闭burst提前让出（消融）：有prefill等待时仍跑满轮数预算")
    p.add_argument("--no-decode-burst-yield-on-arrival", action="store_true",
                   help="关闭「burst期间新请求到达即收尾」（消融）：到达压力不再打断burst")
    p.add_argument("--max-prefill-chunk-tokens", type=int, default=4096,
                   help="自适应调度单步prefill预算上限")
    p.add_argument("--queue-depth-for-full-prefill", type=int, default=16,
                   help="等待队列达到该长度时使用最高prefill配额")
    p.add_argument("--preempt-prefill-tps", type=float, default=10000.0,
                   help="重算成本估算的prefill token/s回退值")
    p.add_argument("--preempt-kv-transfer-gbps", type=float, default=12.0,
                   help="KV swap成本估算的GB/s回退值")
    p.add_argument("--streaming-load", action="store_true",
                   help="按层流式加载+即时量化（meta构造→逐layer物化→量化→释放fp16）；"
                        "7B+ 在16GB卡上的前提。7B 大模型且启用量化时会自动开启，此参数强制开启")
    p.add_argument("--speculative", default="none",
                   help="投机解码: none 或 ngram(n-gram/prompt-lookup草稿, 无模型)")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--warmup-seqs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--slo-ttft-ms", type=float, default=500.0)
    p.add_argument("--slo-tpot-ms", type=float, default=10.0)
    p.add_argument("--repeat-batches", type=int, default=1,
                   help="run the same workload N times back-to-back; batches after the first "
                        "exercise the prefix cache (identical prompts -> prefill tokens ~= 0)")
    p.add_argument("--compare-vllm", action="store_true")
    p.add_argument("--output", default=None, help="JSON results path (default: results/bench_<tag>_<ts>.json)")
    return p.parse_args()


def main():
    args = parse_args()
    args.model = os.path.expanduser(args.model)  # bash argv 不展开 ~ → 手动展开
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True) if args.prompts_file else None
    workload_data = build_workload_data(args, tokenizer)
    prompts, sampling_params = workload_data.prompts, workload_data.sampling_params

    print(f"running nano-vllm ({len(prompts)} seqs, input {args.min_input_len}-{args.max_input_len}, "
          f"output {args.min_output_len}-{args.max_output_len}, shared-prefix {args.shared_prefix_len}, "
          f"eager={args.enforce_eager}) ...")
    batches, kv_info = run_nanovllm(args, prompts, sampling_params)
    metrics = batches[0]

    vllm_res = run_vllm(args, prompts, sampling_params) if args.compare_vllm else None

    tag = (f"n{args.num_seqs}_i{args.min_input_len}-{args.max_input_len}_"
           f"o{args.min_output_len}-{args.max_output_len}"
           + (f"_prefix{args.shared_prefix_len}" if args.shared_prefix_len else "")
           + (f"_nvfp4-{args.nvfp4_backend}" if args.quantization == "nvfp4" else "")
           + ("_eager" if args.enforce_eager else ""))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or os.path.join("results", f"bench_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    request_fingerprint_input = [{
        "request_id": request["request_id"],
        "prompt_token_hash": request["prompt_token_hash"],
        "max_tokens": params.max_tokens,
        "arrival_time_s": request["arrival_time_s"],
        "ttft_slo_ms": request["ttft_slo_ms"],
        "tpot_slo_ms": request["tpot_slo_ms"],
    } for request, params in zip(workload_data.requests, sampling_params)]
    workload_fingerprint = hashlib.sha256(json.dumps(
        {"seed": args.seed, "requests": request_fingerprint_input},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()

    payload = {
        "meta": {"model": args.model, **env_info(), "date": ts, "kv_cache": kv_info},
        "workload": {"num_seqs": len(prompts), "min_input_len": args.min_input_len,
                     "max_input_len": args.max_input_len, "min_output_len": args.min_output_len,
                     "max_output_len": args.max_output_len, "shared_prefix_len": args.shared_prefix_len,
                     "prompts_file": args.prompts_file, "enforce_eager": args.enforce_eager, "tp": args.tp,
                     "repeat_batches": args.repeat_batches,
                     "source": workload_data.source,
                     "arrival_trace_used": workload_data.has_arrival_trace,
                     "fingerprint": workload_fingerprint,
                     "request_ids": [row["request_id"] for row in workload_data.requests],
                     "prompt_tokens": [row["prompt_tokens"] for row in workload_data.requests],
                     "user_prompt_tokens": [row["user_prompt_tokens"] for row in workload_data.requests],
                     "max_output_tokens": [params.max_tokens for params in sampling_params],
                     "arrival_times_seconds": [row["arrival_time_s"]
                                                for row in workload_data.requests],
                     "ttft_slo_targets_ms": [row["ttft_slo_ms"]
                                              for row in workload_data.requests],
                     "tpot_slo_targets_ms": [row["tpot_slo_ms"]
                                              for row in workload_data.requests],
                     "prefix_groups": [row["prefix_group"]
                                       for row in workload_data.requests],
                     "content_classes": [row["content_class"]
                                         for row in workload_data.requests],
                     "prompt_token_hashes": [row["prompt_token_hash"]
                                              for row in workload_data.requests]},
        "scheduler": scheduler_options(args),
        "benchmark_config": {
            "quantization": args.quantization,
            "nvfp4_backend": args.nvfp4_backend,
            "int4_group_size": args.int4_group_size,
            "enforce_eager": args.enforce_eager,
            "tensor_parallel_size": args.tp,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "nanovllm": {"batches": [{k: v for k, v in b.items() if k != "per_request"} for b in batches],
                     "summary": make_summary(metrics), "per_request": metrics["per_request"],
                     "num_preemptions": metrics.get("num_preemptions", 0)},
        "vllm": vllm_res,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    wall = batches[0]["wall"]
    print_report(args, wall, metrics, kv_info, vllm_res, out_path)
    if args.repeat_batches > 1:
        print("per-batch (prefix-cache effect; identical prompts):")
        for b in batches:
            s = b["step_stats"]
            print(f"  batch {b['batch']}: wall {b['wall']:.2f}s | prefill {s['prefill_tokens']} tok "
                  f"({s['prefill_steps']} steps) | decode {s['decode_tokens']} tok ({s['decode_steps']} steps) "
                  f"| preemptions {b.get('num_preemptions', 0)}")


if __name__ == "__main__":
    main()
