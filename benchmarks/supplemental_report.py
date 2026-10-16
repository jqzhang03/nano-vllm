"""Aggregate the 2026-10-15 supplemental experiment outputs.

Run from the repository root after the experiments have written JSON/CSV files.
Aggregation itself needs no model weights or GPU:

    python -m benchmarks.supplemental_report

The script writes review tables under ``results/supplemental/report``, charts
under ``benchmarks/_supplemental_figures``, and refreshes
``benchmarks/_supplemental_report.md``. It does not modify benchmark inputs or
raw results. Charts use PNG when matplotlib is installed and dependency-free
SVG otherwise. Tracked dated follow-ups matching
_supplemental_report_20*.md are appended after the generated 2026-10-15 section
so report regeneration preserves later experiment records.
"""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
import os
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _nested(obj: dict[str, Any], *keys: str,
            default: Any = None) -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _percent_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1.0) * 100.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (json.dumps(value, ensure_ascii=False, sort_keys=True)
                      if isinstance(value, (dict, list)) else value)
                for key, value in row.items()
            })


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _bootstrap_mean_ci(values: list[float], *, seed: int,
                       samples: int = 20_000) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "ci95_low": mean, "ci95_high": mean}
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    low = draws[max(0, int(0.025 * samples))]
    high = draws[min(samples - 1, int(0.975 * samples))]
    return {"n": len(values), "mean": mean, "ci95_low": low,
            "ci95_high": high}


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    try:
        return statistics.correlation(_average_ranks(left), _average_ranks(right))
    except statistics.StatisticsError:
        return None


def _collect_scheduling(data_dir: Path) -> tuple[list[dict[str, Any]],
                                                  list[dict[str, Any]],
                                                  dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    pattern = re.compile(r"scheduling_sharegpt500_r8_seed(\d+)\.json$")
    for path in sorted(data_dir.glob("scheduling_sharegpt500_r8_seed*.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        data = _read_json(path)
        seed = int(match.group(1))
        fingerprint = _nested(data, "workload", "fingerprint")
        for variant in ("baseline", "all_on"):
            run = _nested(data, "runs", variant, default={})
            summary = run.get("summary", {})
            goodput = summary.get("slo_goodput", {})
            ttft = summary.get("ttft", {})
            tpot = summary.get("tpot", {})
            row = {
                "source": path.name,
                "seed": seed,
                "variant": variant,
                "fingerprint": fingerprint,
                "requests_completed": summary.get("requests_completed"),
                "output_tokens": summary.get("output_tokens"),
                "wall_seconds": run.get("wall_seconds"),
                "output_tok_s": summary.get("throughput_output_tok_per_s"),
                "joint_slo_requests": goodput.get("joint_slo_requests"),
                "joint_slo_request_rate_per_s": goodput.get(
                    "joint_slo_request_rate_per_second"),
                "joint_slo_arrival_percent": goodput.get("joint_slo_arrival_percent"),
                "joint_slo_output_tokens_per_s": goodput.get(
                    "joint_slo_output_tokens_per_second"),
                "ttft_p50_ms": _number(ttft.get("p50"), 0.0) * 1000.0,
                "ttft_p95_ms": _number(ttft.get("p95"), 0.0) * 1000.0,
                "ttft_p99_ms": _number(ttft.get("p99"), 0.0) * 1000.0,
                "tpot_p50_ms": _number(tpot.get("p50"), 0.0) * 1000.0,
                "tpot_p95_ms": _number(tpot.get("p95"), 0.0) * 1000.0,
                "tpot_p99_ms": _number(tpot.get("p99"), 0.0) * 1000.0,
                "ttft_slo_percent": summary.get("request_ttft_slo_percent"),
                "tpot_slo_percent": summary.get("request_tpot_slo_percent"),
                "num_preemptions": _nested(run, "scheduler_metrics",
                                             "num_preemptions"),
            }
            rows.append(row)
            by_seed[seed][variant] = row

    paired: list[dict[str, Any]] = []
    fields = (
        "output_tok_s", "joint_slo_output_tokens_per_s",
        "joint_slo_arrival_percent", "ttft_p99_ms", "tpot_p99_ms",
    )
    for seed, variants in sorted(by_seed.items()):
        baseline = variants.get("baseline")
        all_on = variants.get("all_on")
        if not baseline or not all_on:
            continue
        pair: dict[str, Any] = {"seed": seed,
                                "fingerprint": baseline["fingerprint"]}
        for field in fields:
            left = _number(baseline.get(field))
            right = _number(all_on.get(field))
            if left is None or right is None:
                continue
            pair[f"baseline_{field}"] = left
            pair[f"all_on_{field}"] = right
            pair[f"delta_{field}"] = right - left
            pair[f"delta_{field}_percent"] = _percent_delta(right, left)
        paired.append(pair)

    ci: dict[str, Any] = {}
    for field in ("output_tok_s", "joint_slo_output_tokens_per_s",
                  "joint_slo_arrival_percent", "ttft_p99_ms", "tpot_p99_ms"):
        values = [float(pair[f"delta_{field}_percent"])
                  for pair in paired
                  if pair.get(f"delta_{field}_percent") is not None]
        ci[field] = _bootstrap_mean_ci(values, seed=20261015 + len(field))
    return rows, paired, ci


def _collect_scheduling_one_factor(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "scheduling_sharegpt500_r8_seed0_one_factor.json"
    if not path.exists():
        return []
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for variant, run in data.get("runs", {}).items():
        summary = run.get("summary", {})
        goodput = summary.get("slo_goodput", {})
        ttft = summary.get("ttft", {})
        tpot = summary.get("tpot", {})
        rows.append({
            "variant": variant,
            "workload_fingerprint": _nested(data, "workload", "fingerprint"),
            "output_tok_s": summary.get("throughput_output_tok_per_s"),
            "joint_slo_output_tok_s": goodput.get(
                "joint_slo_output_tokens_per_second"),
            "joint_slo_arrival_percent": goodput.get(
                "joint_slo_arrival_percent"),
            "ttft_p99_ms": _number(ttft.get("p99"), 0.0) * 1000.0,
            "tpot_p99_ms": _number(tpot.get("p99"), 0.0) * 1000.0,
        })
    baseline = next((row for row in rows if row["variant"] == "all_on"), None)
    if baseline:
        for row in rows:
            if row is baseline:
                continue
            for field in ("output_tok_s", "joint_slo_output_tok_s",
                          "joint_slo_arrival_percent", "ttft_p99_ms",
                          "tpot_p99_ms"):
                row[f"delta_vs_all_on_{field}_percent"] = _percent_delta(
                    _number(row.get(field)), _number(baseline.get(field)))
    return rows


def _collect_admission(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = re.compile(
        r"admission_sharegpt(?:_smoke)?(\d+)_r(\d+)_prefix(\d+)\.json$")
    for path in sorted(data_dir.glob("admission_sharegpt*.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        data = _read_json(path)
        request_count = int(_nested(data, "workload", "num_requests", default=0))
        offered = _number(_nested(data, "config", "arrival_rate"),
                          float(match.group(2)))
        seed = _nested(data, "config", "seed", default=0)
        prefix_percent = int(match.group(3))
        for variant, run in data.get("runs", {}).items():
            accepted = int(run.get("accepted", 0))
            rejected = int(run.get("rejected", 0))
            goodput = run.get("slo_goodput", {})
            hits = run.get("cache_hits", {})
            ttft = run.get("ttft_ms", {})
            tpot = run.get("tpot_ms", {})
            admission = run.get("admission", {})
            rows.append({
                "source": path.name,
                "requests": request_count,
                "offered_rate_req_s": offered,
                "prefix_percent": prefix_percent,
                "seed": seed,
                "variant": variant,
                "accepted": accepted,
                "rejected": rejected,
                "accept_percent": accepted * 100.0 / max(1, request_count),
                "reject_percent": rejected * 100.0 / max(1, request_count),
                "deferred_total": admission.get(
                    "deferred_total", run.get("deferred_then_accepted", 0)),
                "output_tokens": run.get("output_tokens"),
                "output_tok_s": run.get("output_tokens_per_second"),
                "joint_slo_requests": goodput.get("joint_slo_requests"),
                "joint_slo_request_rate_per_s": goodput.get(
                    "joint_slo_request_rate_per_second"),
                "joint_slo_arrival_percent": goodput.get(
                    "joint_slo_arrival_percent"),
                "joint_slo_output_tok_s": goodput.get(
                    "joint_slo_output_tokens_per_second"),
                "ttft_p50_ms": _nested(ttft, "p50"),
                "ttft_p95_ms": _nested(ttft, "p95"),
                "ttft_p99_ms": _nested(ttft, "p99"),
                "tpot_p50_ms": _nested(tpot, "p50"),
                "tpot_p95_ms": _nested(tpot, "p95"),
                "tpot_p99_ms": _nested(tpot, "p99"),
                "served_ttft_slo_percent": run.get("target_ttft_met_percent"),
                "estimated_prefix_tokens": hits.get("estimated_tokens"),
                "actual_prefix_tokens": hits.get("actual_tokens"),
                "prefix_reuse_requests": hits.get("requests_reusing_prefix"),
                "prefix_estimate_error_tokens": hits.get(
                    "mean_estimate_error_tokens"),
                "fingerprint": _nested(data, "workload", "fingerprint"),
            })
    return rows


def _collect_admission_calibration(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted({path for pattern in (
        "admission_sweep_*.json",
        "admission_refine_*.json",
        "admission_validate_*.json",
    ) for path in data_dir.glob(pattern)})
    for path in paths:
        data = _read_json(path)
        config = data.get("config", {})
        request_count = int(_nested(data, "workload", "num_requests", default=0))
        name = path.name
        if "smoke100" in name:
            phase = "smoke100"
        elif name.startswith("admission_sweep_"):
            phase = "target_work_grid"
        elif name.startswith("admission_refine_"):
            phase = "soft_wait_refinement"
        elif "heldout" in name:
            phase = "heldout_validation"
        else:
            phase = "calibration_validation"
        for variant, run in data.get("runs", {}).items():
            acceptance = run.get("acceptance", {})
            goodput = run.get("slo_goodput", {})
            admission = run.get("admission", {})
            ttft = run.get("ttft_ms", {})
            tpot = run.get("tpot_ms", {})
            accepted = int(run.get("accepted", 0))
            rejected = int(run.get("rejected", 0))
            rows.append({
                "source": name,
                "phase": phase,
                "requests": request_count,
                "offered_rate_req_s": _number(config.get("arrival_rate")),
                "seed": config.get("seed"),
                "variant": variant,
                "target_ttft_ms": _number(
                    config.get("admission_target_ttft_ms")),
                "work_budget_ms": _number(
                    config.get("admission_work_budget_ms")),
                "soft_pressure": _number(
                    config.get("admission_soft_pressure")),
                "max_wait_ms": _number(config.get("max_admission_wait_ms")),
                "slo_ttft_ms": _number(config.get("slo_ttft_ms")),
                "slo_tpot_ms": _number(config.get("slo_tpot_ms")),
                "accepted": accepted,
                "rejected": rejected,
                "accept_percent": _number(
                    acceptance.get("accept_rate_percent"),
                    100.0 * accepted / max(1, request_count)),
                "reject_percent": _number(
                    acceptance.get("reject_rate_percent"),
                    100.0 * rejected / max(1, request_count)),
                "deferred_total": admission.get(
                    "deferred_total", run.get("deferred_then_accepted", 0)),
                "output_tokens": run.get("output_tokens"),
                "output_tok_s": _number(
                    run.get("output_tokens_per_second")),
                "joint_slo_requests": goodput.get("joint_slo_requests"),
                "joint_slo_output_tok_s": _number(
                    goodput.get("joint_slo_output_tokens_per_second")),
                "joint_slo_request_rate_per_s": _number(
                    goodput.get("joint_slo_request_rate_per_second")),
                "joint_slo_arrival_percent": _number(
                    goodput.get("joint_slo_arrival_percent")),
                "ttft_p50_ms": _number(ttft.get("p50")),
                "ttft_p95_ms": _number(ttft.get("p95")),
                "ttft_p99_ms": _number(ttft.get("p99")),
                "tpot_p50_ms": _number(tpot.get("p50")),
                "tpot_p95_ms": _number(tpot.get("p95")),
                "tpot_p99_ms": _number(tpot.get("p99")),
                "served_ttft_slo_percent": _number(
                    run.get("target_ttft_met_percent")),
                "fingerprint": _nested(data, "workload", "fingerprint"),
            })
    return rows


def _pair_admission_validation(
        rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("phase") not in {
                "calibration_validation", "heldout_validation"}:
            continue
        by_source[str(row["source"])][str(row["variant"])] = row

    paired: list[dict[str, Any]] = []
    for source, variants in sorted(by_source.items()):
        off = variants.get("dynamic_off")
        aware = variants.get("dynamic_on_cache_aware")
        if off is None or aware is None:
            continue
        off_goodput = _number(off.get("joint_slo_output_tok_s"))
        aware_goodput = _number(aware.get("joint_slo_output_tok_s"))
        off_reject = _number(off.get("reject_percent"))
        aware_reject = _number(aware.get("reject_percent"))
        off_fingerprint = off.get("fingerprint")
        aware_fingerprint = aware.get("fingerprint")
        paired.append({
            "source": source,
            "phase": aware.get("phase"),
            "seed": aware.get("seed"),
            "requests": aware.get("requests"),
            "offered_rate_req_s": aware.get("offered_rate_req_s"),
            "slo_ttft_ms": aware.get("slo_ttft_ms"),
            "slo_tpot_ms": aware.get("slo_tpot_ms"),
            "target_ttft_ms": aware.get("target_ttft_ms"),
            "work_budget_ms": aware.get("work_budget_ms"),
            "soft_pressure": aware.get("soft_pressure"),
            "max_wait_ms": aware.get("max_wait_ms"),
            "fingerprint_match": bool(
                off_fingerprint and off_fingerprint == aware_fingerprint),
            "admission_off_reject_percent": off_reject,
            "cache_aware_reject_percent": aware_reject,
            "reject_delta_percentage_points": (
                aware_reject - off_reject
                if aware_reject is not None and off_reject is not None
                else None),
            "admission_off_joint_slo_output_tok_s": off_goodput,
            "cache_aware_joint_slo_output_tok_s": aware_goodput,
            "joint_slo_output_delta_tok_s": (
                aware_goodput - off_goodput
                if aware_goodput is not None and off_goodput is not None
                else None),
            "joint_slo_output_delta_percent": _percent_delta(
                aware_goodput, off_goodput),
            "admission_off_ttft_p99_ms": _number(off.get("ttft_p99_ms")),
            "cache_aware_ttft_p99_ms": _number(aware.get("ttft_p99_ms")),
            "ttft_p99_delta_ms": (
                _number(aware.get("ttft_p99_ms"))
                - _number(off.get("ttft_p99_ms"))
                if _number(aware.get("ttft_p99_ms")) is not None
                and _number(off.get("ttft_p99_ms")) is not None else None),
        })
    return paired


def _collect_kv(data_dir: Path) -> list[dict[str, Any]]:
    names = (
        "kv_auto_no_swap_warmed.csv",
        "kv_auto_swap_warmed.csv",
        "kv_fp8_no_swap_warmed.csv",
        "kv_real_prompts_no_swap.csv",
        "kv_auto_swap_4096_highconc.csv",
        "kv_auto_no_swap_4096_highconc.csv",
        "kv_auto_swap_probe_4096_c32_o512.csv",
        "kv_auto_swap_force_4096_c32_o512_rep3.csv",
        "kv_auto_no_swap_force_4096_c32_o512_rep3.csv",
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            continue
        for row in _read_csv(path):
            rows.append({
                "source": name,
                "context_length": int(row["context_length"]),
                "prompt_length_mode": row["prompt_length_mode"],
                "concurrency": int(row["concurrency"]),
                "repeats": int(row["repeats"]),
                "requests": int(row["requests"]),
                "accepted": int(row["accepted"]),
                "acceptance_percent": _number(row.get("acceptance_percent")),
                "ttft_slo_met_percent": _number(row.get("ttft_slo_met_percent")),
                "tpot_slo_met_percent": _number(row.get("tpot_slo_met_percent")),
                "output_tok_s": _number(row.get("output_tokens_per_second")),
                "ttft_p99_ms": _number(row.get("ttft_ms_p99")),
                "tpot_p99_ms": _number(row.get("average_tpot_ms_p99")),
                "preemptions": int(row.get("preemptions") or 0),
                "swaps": int(row.get("swaps") or 0),
                "recompute_preemptions": int(row.get("recompute_preemptions") or 0),
                "kv_capacity_token_slots": int(row.get("kv_capacity_token_slots") or 0),
                "kv_total_blocks": int(row.get("kv_total_blocks") or 0),
                "kv_peak_used_blocks": int(row.get("kv_peak_used_blocks") or 0),
                "kv_peak_pool_utilization": _number(
                    row.get("kv_peak_pool_utilization")),
                "max_full_context_requests": int(
                    row.get("max_full_context_requests") or 0),
                "actual_prompt_tokens_avg": _number(
                    row.get("actual_prompt_tokens_avg")),
                "gpu_peak_allocated_gb": (
                    (_number(row.get("gpu_peak_allocated_bytes"), 0.0) or 0.0)
                    / 1e9),
            })
    return rows


def _collect_fp8_validation(data_dir: Path) -> tuple[list[dict[str, Any]],
                                                     dict[str, Any]]:
    csv_path = data_dir / "kv_fp8_sharegpt_400_100.csv"
    json_path = data_dir / "kv_fp8_sharegpt_400_100.json"
    if not csv_path.exists():
        return [], {}
    numeric_fields = (
        "scale_margin", "aligned_first_token_prompts",
        "first_token_agreement_percent", "max_evaluation_range_utilization",
        "mean_evaluation_range_utilization", "components_over_e4m3_limit",
        "logits_max_abs_error", "logits_mean_abs_error", "logits_rmse",
        "logits_cosine_similarity", "kl_ref_to_fp8", "top1_agreement_percent",
        "top5_overlap_percent",
    )
    rows: list[dict[str, Any]] = []
    for raw in _read_csv(csv_path):
        row: dict[str, Any] = dict(raw)
        for field in numeric_fields:
            row[field] = _number(raw.get(field))
        rows.append(row)
    details: dict[str, Any] = {}
    if json_path.exists():
        data = _read_json(json_path)
        calibration = data.get("calibration", {})
        evaluation = data.get("evaluation", {})
        details = {
            "calibration_prompt_count": calibration.get("prompts"),
            "calibration_token_count": calibration.get("total_tokens"),
            "calibration_prompt_lengths": calibration.get("prompt_lengths"),
            "evaluation_prompt_count": evaluation.get("prompts"),
            "evaluation_token_count": evaluation.get("total_tokens"),
            "evaluation_prompt_lengths": evaluation.get("prompt_lengths"),
            "margins": data.get("config", {}).get("margins"),
            "warnings": data.get("warnings", []),
        }
    return rows, details


def _collect_micro(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = data_dir / "moe_int4_micro_20261015.json"
    if not path.exists():
        return [], {}
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for batch, variants in data.get("runs", {}).items():
        loop_ms = _number(_nested(variants, "loop", "median_ms"))
        for variant, result in variants.items():
            variant_ms = _number(result.get("median_ms"))
            row = {
                "token_batch": int(batch),
                "variant": variant,
                "median_ms": variant_ms,
                "speedup_vs_loop": (
                    loop_ms / variant_ms
                    if loop_ms is not None and loop_ms > 0
                    and variant_ms is not None and variant_ms > 0 else None),
                "exact_match": result.get("exact_match"),
                "relative_error": result.get("relative_error"),
                "max_abs_error": result.get("max_abs_error"),
            }
            for stage, value in result.get("stages", {}).items():
                row[stage if stage.endswith("_ms") else f"{stage}_ms"] = value
            rows.append(row)
    correctness = data.get("correctness", {})
    layer_check = correctness.get("all_moe_layers", {})
    layers = layer_check.get("layers", {})
    per_variant: dict[str, dict[str, int]] = defaultdict(
        lambda: {"checks": 0, "exact_matches": 0, "within_tolerance": 0})
    layer_cases = 0
    for layer in layers.values():
        for batch_checks in layer.get("checks", {}).values():
            for seed_checks in batch_checks.values():
                layer_cases += 1
                for variant, check in seed_checks.items():
                    counts = per_variant[variant]
                    counts["checks"] += 1
                    counts["exact_matches"] += int(
                        check.get("exact_match") is True)
                    counts["within_tolerance"] += int(
                        check.get("within_tolerance") is True)
    correctness_summary = {
        "within_tolerance": correctness.get("within_tolerance"),
        "all_layers_within_tolerance": layer_check.get("within_tolerance"),
        "all_layer_count": layer_check.get("layer_count"),
        "all_layer_checks": layer_cases,
        "all_layer_variant_checks": dict(per_variant),
        "all_layer_token_batches": layer_check.get("token_batches"),
        "all_layer_seeds": layer_check.get("seeds"),
        "profile_token_batches": data.get("benchmark_config", {}).get(
            "profile_token_batches"),
    }
    return rows, correctness_summary


def _collect_online_moe(data_dir: Path) -> tuple[list[dict[str, Any]],
                                                dict[str, Any]]:
    path = data_dir / "moe_int4_realtrace_sweep_20261015.json"
    if not path.exists():
        return [], {}
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for pair in data.get("pairs", []):
        condition = pair.get("condition", {})
        comparison = pair.get("comparison", {})
        split = _nested(pair, "runs", "segment_split", "summary", default={})
        fused = _nested(pair, "runs", "segment_fused", "summary", default={})
        split_slo = split.get("slo_goodput", {})
        fused_slo = fused.get("slo_goodput", {})
        parity = pair.get("output_parity", {})
        if not isinstance(parity, dict):
            parity = {}
        rows.append({
            "source": path.name,
            "seed": condition.get("seed"),
            "repeat_index": pair.get("repeat_index"),
            "tpot_slo_ms": condition.get("tpot_slo_ms"),
            "arrival_rate_req_s": condition.get("arrival_rate_requests_per_second"),
            "num_requests": condition.get("num_requests"),
            "output_tokens_cap": condition.get("output_tokens"),
            "workload_fingerprint": condition.get("workload_fingerprint"),
            "output_parity": parity.get("all_requests_match"),
            "matching_output_requests": parity.get("matching_requests"),
            "total_output_requests": parity.get("total_requests"),
            "output_mismatch_count": len(parity.get("mismatches", [])),
            "split_output_tok_s": split.get("throughput_output_tok_per_s"),
            "fused_output_tok_s": fused.get("throughput_output_tok_per_s"),
            "throughput_delta_percent": comparison.get(
                "output_throughput_delta_percent"),
            "split_joint_slo_requests": split_slo.get("joint_slo_requests"),
            "fused_joint_slo_requests": fused_slo.get("joint_slo_requests"),
            "split_joint_slo_output_tok_s": split_slo.get(
                "joint_slo_output_tokens_per_second"),
            "fused_joint_slo_output_tok_s": fused_slo.get(
                "joint_slo_output_tokens_per_second"),
            "joint_slo_output_delta_tok_s": comparison.get(
                "joint_slo_output_goodput_delta_tok_s"),
            "joint_slo_output_delta_percent": comparison.get(
                "joint_slo_output_goodput_delta_percent"),
            "ttft_p50_delta_percent": comparison.get("ttft_p50_delta_percent"),
            "ttft_p99_delta_percent": comparison.get("ttft_p99_delta_percent"),
            "tpot_p50_delta_percent": comparison.get("tpot_p50_delta_percent"),
            "tpot_p99_delta_percent": comparison.get("tpot_p99_delta_percent"),
        })
    return rows, data.get("summary", {})


def _collect_online_moe_repeats(data_dir: Path) -> tuple[list[dict[str, Any]],
                                                        dict[str, Any]]:
    path = data_dir / (
        "moe_int4_realtrace_repeat_r8_o128_t100_seed5_rep5_20261015.json")
    if not path.exists():
        return [], {}
    data = _read_json(path)
    rows: list[dict[str, Any]] = []
    for pair in data.get("pairs", []):
        condition = pair.get("condition", {})
        comparison = pair.get("comparison", {})
        split = _nested(pair, "runs", "segment_split", "summary", default={})
        fused = _nested(pair, "runs", "segment_fused", "summary", default={})
        split_slo = split.get("slo_goodput", {})
        fused_slo = fused.get("slo_goodput", {})
        parity = pair.get("output_parity", {})
        if not isinstance(parity, dict):
            parity = {}
        rows.append({
            "source": path.name,
            "seed": condition.get("seed"),
            "repeat_index": pair.get("repeat_index"),
            "tpot_slo_ms": condition.get("tpot_slo_ms"),
            "arrival_rate_req_s": condition.get(
                "arrival_rate_requests_per_second"),
            "num_requests": condition.get("num_requests"),
            "output_tokens_cap": condition.get("output_tokens"),
            "workload_fingerprint": condition.get("workload_fingerprint"),
            "output_parity": parity.get("all_requests_match"),
            "matching_output_requests": parity.get("matching_requests"),
            "total_output_requests": parity.get("total_requests"),
            "output_mismatch_count": len(parity.get("mismatches", [])),
            "split_output_tok_s": split.get("throughput_output_tok_per_s"),
            "fused_output_tok_s": fused.get("throughput_output_tok_per_s"),
            "throughput_delta_percent": comparison.get(
                "output_throughput_delta_percent"),
            "split_joint_slo_requests": split_slo.get("joint_slo_requests"),
            "fused_joint_slo_requests": fused_slo.get("joint_slo_requests"),
            "split_joint_slo_output_tok_s": split_slo.get(
                "joint_slo_output_tokens_per_second"),
            "fused_joint_slo_output_tok_s": fused_slo.get(
                "joint_slo_output_tokens_per_second"),
            "joint_slo_output_delta_tok_s": comparison.get(
                "joint_slo_output_goodput_delta_tok_s"),
            "joint_slo_output_delta_percent": comparison.get(
                "joint_slo_output_goodput_delta_percent"),
            "ttft_p50_delta_percent": comparison.get("ttft_p50_delta_percent"),
            "ttft_p99_delta_percent": comparison.get("ttft_p99_delta_percent"),
            "tpot_p50_delta_percent": comparison.get("tpot_p50_delta_percent"),
            "tpot_p99_delta_percent": comparison.get("tpot_p99_delta_percent"),
        })
    throughput = [float(row["throughput_delta_percent"])
                  for row in rows if row.get("throughput_delta_percent") is not None]
    goodput = [float(row["joint_slo_output_delta_tok_s"])
               for row in rows if row.get("joint_slo_output_delta_tok_s") is not None]
    parity_count = sum(row.get("output_parity") is True for row in rows)
    wins = sum(value > 0 for value in throughput)
    losses = sum(value < 0 for value in throughput)
    ties = sum(value == 0 for value in throughput)
    summary = {
        "paired_measurements": len(rows),
        "seed_count": len({row.get("seed") for row in rows}),
        "repetitions_per_condition": data.get("summary", {}).get(
            "repetitions_per_condition"),
        "condition": {
            "arrival_rate_req_s": 8.0,
            "num_requests": 16,
            "output_tokens_cap": 128,
            "tpot_slo_ms": 100.0,
        },
        "output_throughput_delta_percent": {
            **_bootstrap_mean_ci(throughput, seed=20261015),
            "median": statistics.median(throughput) if throughput else None,
            "min": min(throughput) if throughput else None,
            "max": max(throughput) if throughput else None,
            "wins": wins,
            "ties": ties,
            "losses": losses,
        },
        "joint_slo_output_delta_tok_s": {
            **_bootstrap_mean_ci(goodput, seed=20261016),
            "median": statistics.median(goodput) if goodput else None,
            "min": min(goodput) if goodput else None,
            "max": max(goodput) if goodput else None,
        },
        "output_token_parity_pairs": parity_count,
        "output_token_parity_total": len(rows),
    }
    return rows, summary


def _collect_spec(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = (
        "spec_content_batch8_g4_seed012.json",
        "spec_batch_gamma_free_repeat_seed012.json",
    )
    rows: list[dict[str, Any]] = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            continue
        data = _read_json(path)
        for pair in data.get("pairs", []):
            runs = pair.get("runs", {})
            baseline = runs.get("none", {})
            ngram = runs.get("ngram", {})
            baseline_tps = _number(baseline.get("throughput_tok_per_s"))
            ngram_tps = _number(ngram.get("throughput_tok_per_s"))
            content = ngram.get("content_metrics", {})
            comparison = _nested(pair, "comparisons", "ngram", default={})
            for mode, result in (("none", baseline), ("ngram", ngram)):
                result_content = result.get("content_metrics", {})
                rows.append({
                    "source": name,
                    "style": pair.get("style"),
                    "batch_size": pair.get("batch_size"),
                    "max_draft_len": pair.get("max_draft_len"),
                    "seed": pair.get("seed"),
                    "repeat_index": pair.get("repeat_index"),
                    "content_class_counts": pair.get("content_class_counts"),
                    "mode": mode,
                    "throughput_tok_s": result.get("throughput_tok_per_s"),
                    "speedup_vs_none": (
                        ngram_tps / baseline_tps if mode == "ngram"
                        and ngram_tps is not None and baseline_tps else 1.0),
                    "alpha": result.get("alpha"),
                    "avg_draft_len": result.get("avg_draft_len"),
                    "spec_proposed_tokens": result.get("spec_proposed_tokens"),
                    "spec_accepted_tokens": result.get("spec_accepted_tokens"),
                    "ttft_p50_ms": _nested(result, "ttft_ms", "p50"),
                    "ttft_p95_ms": _nested(result, "ttft_ms", "p95"),
                    "tpot_p50_ms": _nested(result, "tpot_ms", "p50"),
                    "tpot_p95_ms": _nested(result, "tpot_ms", "p95"),
                    "distinct4_mean": _nested(result_content, "distinct_4", "mean"),
                    "fourgram_repeat_rate_mean": _nested(
                        result_content, "fourgram_repeat_rate", "mean"),
                    "token_entropy_bits_mean": _nested(
                        result_content, "token_entropy_bits", "mean"),
                    "output_token_match": comparison.get("output_token_match"),
                    "input_fingerprint": pair.get("input_fingerprint"),
                    "output_token_fingerprint": result.get(
                        "output_token_fingerprint"),
                })

    ngram_rows = [row for row in rows if row["mode"] == "ngram"]
    alphas = [float(row["alpha"]) for row in ngram_rows
              if row.get("alpha") is not None]
    repeats = [float(row["fourgram_repeat_rate_mean"]) for row in ngram_rows
               if row.get("alpha") is not None
               and row.get("fourgram_repeat_rate_mean") is not None]
    aligned_alphas = [float(row["alpha"]) for row in ngram_rows
                      if row.get("alpha") is not None
                      and row.get("fourgram_repeat_rate_mean") is not None]
    distinct = [float(row["distinct4_mean"]) for row in ngram_rows
                if row.get("alpha") is not None
                and row.get("distinct4_mean") is not None]
    entropy = [float(row["token_entropy_bits_mean"]) for row in ngram_rows
               if row.get("alpha") is not None
               and row.get("token_entropy_bits_mean") is not None]
    alpha_repeat = [float(row["alpha"]) for row in ngram_rows
                    if row.get("alpha") is not None
                    and row.get("fourgram_repeat_rate_mean") is not None]
    alpha_distinct = [float(row["alpha"]) for row in ngram_rows
                      if row.get("alpha") is not None
                      and row.get("distinct4_mean") is not None]
    alpha_entropy = [float(row["alpha"]) for row in ngram_rows
                     if row.get("alpha") is not None
                     and row.get("token_entropy_bits_mean") is not None]
    correlation = {
        "ngram_runs": len(ngram_rows),
        "alpha_vs_fourgram_repeat_rate_spearman": _spearman(
            alpha_repeat, repeats),
        "alpha_vs_distinct4_spearman": _spearman(alpha_distinct, distinct),
        "alpha_vs_token_entropy_spearman": _spearman(alpha_entropy, entropy),
        "output_token_match_pairs": sum(
            1 for row in ngram_rows if row.get("output_token_match") is True),
        "output_token_mismatch_pairs": sum(
            1 for row in ngram_rows if row.get("output_token_match") is False),
    }
    return rows, correlation


def _format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _span(values: list[float], *, include_zero: bool = False,
          padding: float = 0.06) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    if include_zero:
        low, high = min(low, 0.0), max(high, 0.0)
    width = high - low
    if width == 0:
        width = max(abs(high) * 0.2, 1.0)
    return low - width * padding, high + width * padding


class _SvgFigure:
    """Small SVG chart writer used when matplotlib is unavailable."""

    def __init__(self, title: str, *, width: int = 1200,
                 height: int = 600) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.panels: list[_SvgPanel] = []
        self.elements = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263238}'
            '</style>',
            f'<text x="{width / 2:g}" y="26" text-anchor="middle" '
            f'font-size="17" font-weight="600">{escape(title)}</text>',
        ]

    def panel(self, x: int, y: int, width: int, height: int, *,
              title: str, xlabel: str, ylabel: str,
              xlim: tuple[float, float], ylim: tuple[float, float],
              xticks: list[tuple[float, str]] | None = None,
              rotate_xticks: bool = False) -> _SvgPanel:
        panel = _SvgPanel(self, x, y, width, height, title=title,
                          xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim,
                          xticks=xticks, rotate_xticks=rotate_xticks)
        self.panels.append(panel)
        return panel

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for panel in self.panels:
            panel._draw_legend()
        path.write_text("\n".join((*self.elements, "</svg>")),
                        encoding="utf-8")


class _SvgPanel:
    def __init__(self, figure: _SvgFigure, x: int, y: int, width: int,
                 height: int, *, title: str, xlabel: str, ylabel: str,
                 xlim: tuple[float, float], ylim: tuple[float, float],
                 xticks: list[tuple[float, str]] | None,
                 rotate_xticks: bool) -> None:
        self.figure = figure
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.xlim = xlim if xlim[1] > xlim[0] else (xlim[0], xlim[0] + 1.0)
        self.ylim = ylim if ylim[1] > ylim[0] else (ylim[0], ylim[0] + 1.0)
        self.plot_x = x + 74
        self.plot_y = y + 70
        self.plot_w = max(40, width - 100)
        self.plot_h = max(50, height - 145)
        self.legend_items: list[tuple[str, str, str]] = []
        figure.elements.append(
            f'<text x="{x + width / 2:g}" y="{y + 19}" text-anchor="middle" '
            f'font-size="13" font-weight="600">{escape(title)}</text>')

        for index in range(5):
            fraction = index / 4
            value = self.ylim[0] + (self.ylim[1] - self.ylim[0]) * fraction
            py = self._y(value)
            figure.elements.append(
                f'<line x1="{self.plot_x}" y1="{py:.2f}" '
                f'x2="{self.plot_x + self.plot_w}" y2="{py:.2f}" '
                'stroke="#d7dce1" stroke-width="1"/>')
            figure.elements.append(
                f'<text x="{self.plot_x - 8}" y="{py + 4:.2f}" '
                f'text-anchor="end" font-size="10">{_format_tick(value)}</text>')

        ticks = xticks or [
            (self.xlim[0] + (self.xlim[1] - self.xlim[0]) * index / 4,
             _format_tick(self.xlim[0]
                          + (self.xlim[1] - self.xlim[0]) * index / 4))
            for index in range(5)
        ]
        for value, label in ticks:
            px = self._x(value)
            figure.elements.append(
                f'<line x1="{px:.2f}" y1="{self.plot_y}" '
                f'x2="{px:.2f}" y2="{self.plot_y + self.plot_h}" '
                'stroke="#edf0f2" stroke-width="1"/>')
            transform = (f' transform="rotate(-38 {px:.2f} '
                         f'{self.plot_y + self.plot_h + 18})"'
                         if rotate_xticks else "")
            anchor = "end" if rotate_xticks else "middle"
            figure.elements.append(
                f'<text x="{px:.2f}" y="{self.plot_y + self.plot_h + 18}" '
                f'text-anchor="{anchor}" font-size="9"{transform}>'
                f'{escape(str(label))}</text>')
        figure.elements.extend((
            f'<line x1="{self.plot_x}" y1="{self.plot_y + self.plot_h}" '
            f'x2="{self.plot_x + self.plot_w}" '
            f'y2="{self.plot_y + self.plot_h}" stroke="#64727a"/>',
            f'<line x1="{self.plot_x}" y1="{self.plot_y}" '
            f'x2="{self.plot_x}" y2="{self.plot_y + self.plot_h}" '
            'stroke="#64727a"/>',
            f'<text x="{self.plot_x + self.plot_w / 2:g}" '
            f'y="{y + height - 9}" text-anchor="middle" font-size="11">'
            f'{escape(xlabel)}</text>',
            f'<text x="{x + 16}" y="{self.plot_y + self.plot_h / 2:g}" '
            f'text-anchor="middle" font-size="11" '
            f'transform="rotate(-90 {x + 16} '
            f'{self.plot_y + self.plot_h / 2:g})">{escape(ylabel)}</text>',
        ))

    def _x(self, value: float) -> float:
        return self.plot_x + (value - self.xlim[0]) / (
            self.xlim[1] - self.xlim[0]) * self.plot_w

    def _y(self, value: float) -> float:
        return self.plot_y + self.plot_h - (value - self.ylim[0]) / (
            self.ylim[1] - self.ylim[0]) * self.plot_h

    def add_series(self, points: list[tuple[float, float]], *, color: str,
                   label: str = "", connect: bool = True,
                   marker: str = "circle", dashed: bool = False,
                   tooltips: list[str] | None = None,
                   point_labels: list[str] | None = None) -> None:
        valid = [(x, y, index) for index, (x, y) in enumerate(points)
                 if math.isfinite(x) and math.isfinite(y)]
        if connect and len(valid) > 1:
            path = " ".join(f"{self._x(x):.2f},{self._y(y):.2f}"
                            for x, y, _ in valid)
            dash = ' stroke-dasharray="5 4"' if dashed else ""
            self.figure.elements.append(
                f'<polyline points="{path}" fill="none" stroke="{color}" '
                f'stroke-width="2"{dash}/>')
        for x, y, index in valid:
            px, py = self._x(x), self._y(y)
            tooltip = (escape(tooltips[index]) if tooltips
                       and index < len(tooltips) else "")
            title = f"<title>{tooltip}</title>" if tooltip else ""
            if marker == "square":
                shape = (f'<rect x="{px - 3.5:.2f}" y="{py - 3.5:.2f}" '
                         f'width="7" height="7" fill="{color}">'
                         f'{title}</rect>')
            elif marker == "triangle":
                shape = (f'<path d="M {px:.2f} {py - 4:.2f} '
                         f'L {px + 4:.2f} {py + 3.5:.2f} '
                         f'L {px - 4:.2f} {py + 3.5:.2f} Z" fill="{color}">'
                         f'{title}</path>')
            else:
                shape = (f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.5" '
                         f'fill="{color}">{title}</circle>')
            self.figure.elements.append(shape)
            if point_labels and index < len(point_labels) and point_labels[index]:
                self.figure.elements.append(
                    f'<text x="{px + 5:.2f}" y="{py - 5:.2f}" '
                    f'font-size="8">{escape(point_labels[index])}</text>')
        if label and not any(item[0] == label for item in self.legend_items):
            self.legend_items.append((label, color, marker))

    def add_hline(self, value: float, *, color: str, label: str,
                  dashed: bool = True) -> None:
        py = self._y(value)
        dash = ' stroke-dasharray="4 4"' if dashed else ""
        self.figure.elements.append(
            f'<line x1="{self.plot_x}" y1="{py:.2f}" '
            f'x2="{self.plot_x + self.plot_w}" y2="{py:.2f}" '
            f'stroke="{color}" stroke-width="1.2"{dash}/>')
        if not any(item[0] == label for item in self.legend_items):
            self.legend_items.append((label, color, "line"))

    def _draw_legend(self) -> None:
        base_x = self.plot_x
        base_y = self.y + 43
        current_x = base_x
        current_y = base_y
        max_x = self.x + self.width - 12
        for label, color, marker in self.legend_items:
            estimated = 24 + min(170, len(label) * 6.0)
            if current_x + estimated > max_x:
                current_x = base_x
                current_y += 14
            if marker == "line":
                sample = (f'<line x1="{current_x}" y1="{current_y - 3}" '
                          f'x2="{current_x + 14}" y2="{current_y - 3}" '
                          f'stroke="{color}" stroke-width="2" '
                          'stroke-dasharray="4 3"/>')
            else:
                sample = (f'<circle cx="{current_x + 5}" '
                          f'cy="{current_y - 3}" r="3.5" fill="{color}"/>')
            self.figure.elements.append(sample)
            self.figure.elements.append(
                f'<text x="{current_x + 13}" y="{current_y}" '
                f'font-size="9">{escape(label)}</text>')
            current_x += estimated


def _figure(path: Path, draw, pyplot=None) -> bool:
    try:
        if pyplot is None:
            import matplotlib.pyplot as plt
        else:
            plt = pyplot
    except ImportError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    draw(plt)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close("all")
    return True


def _write_svg_charts(output_dir: Path, *,
                      scheduling_rows: list[dict[str, Any]],
                      admission_rows: list[dict[str, Any]],
                      admission_calibration_rows: list[dict[str, Any]],
                      kv_rows: list[dict[str, Any]],
                      micro_rows: list[dict[str, Any]],
                      online_rows: list[dict[str, Any]],
                      spec_rows: list[dict[str, Any]],
                      spec_correlation: dict[str, Any]) -> None:
    colors = {
        "baseline": "#355f8a", "all_on": "#d17c2f",
        "dynamic_off": "#355f8a", "dynamic_on_no_cache": "#3b9b73",
        "dynamic_on_cache_aware": "#d17c2f",
        "segment_split": "#2d708e", "segment_fused": "#d17c2f",
        "free": "#2d708e", "repeat": "#d17c2f",
        "json": "#5969a8", "structured": "#6aa786",
        "sharegpt": "#bf6b5d", "open_ended": "#8c6bb1",
    }

    paired = [row for row in scheduling_rows
              if row.get("variant") in {"baseline", "all_on"}]
    if paired:
        figure = _SvgFigure(
            "Paired scheduling results · five ShareGPT arrival seeds",
            width=1220, height=590)
        for index, (field, label) in enumerate((
                ("output_tok_s", "Output throughput (tokens/s)"),
                ("joint_slo_output_tokens_per_s",
                 "Joint-SLO output goodput (tokens/s)"))):
            values = [_number(row.get(field)) for row in paired]
            span = _span([value for value in values if value is not None],
                         padding=0.12)
            panel = figure.panel(
                20 + index * 600, 48, 585, 510,
                title=("ShareGPT · 500 requests · 8 req/s" if index == 0
                       else "All arrivals count toward SLO goodput"),
                xlabel="Scheduling variant", ylabel=label,
                xlim=(0, 1), ylim=span,
                xticks=[(0, "baseline"), (1, "all_on")])
            by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
            for row in paired:
                by_seed[int(row["seed"])][str(row["variant"])] = row
            baseline_points: list[tuple[float, float]] = []
            all_on_points: list[tuple[float, float]] = []
            for seed, variants in sorted(by_seed.items()):
                base = _number(variants.get("baseline", {}).get(field))
                tuned = _number(variants.get("all_on", {}).get(field))
                if base is None or tuned is None:
                    continue
                panel.add_series([(0, base), (1, tuned)], color="#b7bec7",
                                 connect=True, marker="circle")
                baseline_points.append((0, base))
                all_on_points.append((1, tuned))
            panel.add_series(baseline_points, color=colors["baseline"],
                             label="baseline", connect=False)
            panel.add_series(all_on_points, color=colors["all_on"],
                             label="all_on", connect=False, marker="square")
        figure.save(output_dir / "scheduling_slo.svg")

    admission = [row for row in admission_rows
                 if int(row.get("requests", 0)) == 500
                 and _number(row.get("joint_slo_output_tok_s")) is not None]
    if admission:
        x_values = [_number(row.get("reject_percent")) for row in admission]
        y_values = [_number(row.get("joint_slo_output_tok_s"))
                    for row in admission]
        figure = _SvgFigure("Admission operating points · 500-request trace",
                            width=1120, height=650)
        panel = figure.panel(
            25, 46, 1070, 560, title="Lower rejection and higher goodput are preferred",
            xlabel="Rejected requests (% of arrivals)",
            ylabel="Joint-SLO output goodput (tokens/s)",
            xlim=_span([v for v in x_values if v is not None], padding=0.10),
            ylim=_span([v for v in y_values if v is not None], padding=0.10))
        for variant, marker in (("dynamic_off", "circle"),
                                ("dynamic_on_no_cache", "square"),
                                ("dynamic_on_cache_aware", "triangle")):
            subset = [row for row in admission if row["variant"] == variant]
            points = [(_number(row.get("reject_percent")) or 0.0,
                       _number(row.get("joint_slo_output_tok_s")) or 0.0)
                      for row in subset]
            labels = [f"{float(row['offered_rate_req_s']):g}r/"
                      f"{row['prefix_percent']}%" for row in subset]
            tooltips = [f"{variant}; rate={row['offered_rate_req_s']} req/s; "
                        f"prefix={row['prefix_percent']}%; "
                        f"TTFT p99={row['ttft_p99_ms']} ms" for row in subset]
            panel.add_series(points, color=colors[variant], label=variant,
                             connect=False, marker=marker, tooltips=tooltips,
                             point_labels=labels)
        figure.save(output_dir / "admission_pareto.svg")

    admission_grid = [row for row in admission_calibration_rows
                      if row.get("phase") == "target_work_grid"
                      and int(row.get("requests", 0)) >= 300
                      and _number(row.get("offered_rate_req_s")) == 96.0
                      and _number(row.get("slo_tpot_ms")) == 20.0]
    aware_grid = [row for row in admission_grid
                  if row.get("variant") == "dynamic_on_cache_aware"
                  and _number(row.get("joint_slo_output_tok_s")) is not None
                  and _number(row.get("reject_percent")) is not None]
    if aware_grid:
        x_values = [float(row["reject_percent"]) for row in aware_grid]
        y_values = [float(row["joint_slo_output_tok_s"]) for row in aware_grid]
        figure = _SvgFigure(
            "Admission parameter search · 96 req/s, 400-request calibration trace",
            width=1120, height=650)
        panel = figure.panel(
            25, 46, 1070, 560,
            title="Cache-aware parameter candidates; lower rejection and higher goodput are preferred",
            xlabel="Rejected requests (% of arrivals)",
            ylabel="Joint-SLO output goodput (tokens/s)",
            xlim=(0.0, max(10.0, min(100.0, max(x_values) + 5.0))),
            ylim=_span(y_values, include_zero=True))
        budget_colors = {2000.0: "#2d708e", 5000.0: "#3b9b73",
                         10000.0: "#d17c2f"}
        for budget in sorted({float(row["work_budget_ms"])
                              for row in aware_grid
                              if row.get("work_budget_ms") is not None}):
            subset = [row for row in aware_grid
                      if float(row["work_budget_ms"]) == budget]
            points = [(float(row["reject_percent"]),
                       float(row["joint_slo_output_tok_s"]))
                      for row in subset]
            labels = [f"T{float(row['target_ttft_ms']):g}"
                      for row in subset]
            tooltips = [
                f"target={row['target_ttft_ms']} ms; work={budget:g} ms; "
                f"soft={row['soft_pressure']}; max_wait={row['max_wait_ms']} ms; "
                f"reject={row['reject_percent']:.1f}%; "
                f"goodput={row['joint_slo_output_tok_s']:.1f} tok/s"
                for row in subset
            ]
            panel.add_series(
                points, color=budget_colors.get(budget, "#5969a8"),
                label=f"work budget {budget:g} ms", connect=False,
                marker="circle", tooltips=tooltips, point_labels=labels)

        blind_grid = [row for row in admission_grid
                      if row.get("variant") == "dynamic_on_no_cache"
                      and _number(row.get("joint_slo_output_tok_s")) is not None
                      and _number(row.get("reject_percent")) is not None]
        if blind_grid:
            panel.add_series(
                [(float(row["reject_percent"]),
                  float(row["joint_slo_output_tok_s"])) for row in blind_grid],
                color="#8c6bb1", label="cache-blind candidates",
                connect=False, marker="triangle")
        off_grid = [row for row in admission_grid
                    if row.get("variant") == "dynamic_off"
                    and _number(row.get("joint_slo_output_tok_s")) is not None
                    and _number(row.get("reject_percent")) is not None]
        if off_grid:
            off_reject = statistics.median(
                float(row["reject_percent"]) for row in off_grid)
            off_goodput = statistics.median(
                float(row["joint_slo_output_tok_s"]) for row in off_grid)
            panel.add_series(
                [(off_reject, off_goodput)], color="#355f8a",
                label="admission off", connect=False, marker="square")
        frontier: list[dict[str, Any]] = []
        best_goodput = -math.inf
        for row in sorted(aware_grid, key=lambda item: float(item["reject_percent"])):
            goodput = float(row["joint_slo_output_tok_s"])
            if goodput >= best_goodput:
                frontier.append(row)
                best_goodput = goodput
        if len(frontier) > 1:
            panel.add_series(
                [(float(row["reject_percent"]),
                  float(row["joint_slo_output_tok_s"])) for row in frontier],
                color="#263238", label="cache-aware Pareto frontier",
                connect=True, marker="square", dashed=True)
        figure.save(output_dir / "admission_calibration.svg")

    admission_refine = [row for row in admission_calibration_rows
                        if row.get("phase") == "soft_wait_refinement"
                        and int(row.get("requests", 0)) >= 300
                        and _number(row.get("offered_rate_req_s")) == 96.0
                        and _number(row.get("slo_tpot_ms")) == 80.0
                        and row.get("variant") == "dynamic_on_cache_aware"
                        and _number(row.get("joint_slo_output_tok_s")) is not None]
    if admission_refine:
        figure = _SvgFigure(
            "Admission soft-pressure / max-wait refinement · TPOT ≤ 80 ms",
            width=1220, height=650)
        reject_values = [float(row["reject_percent"])
                         for row in admission_refine
                         if row.get("reject_percent") is not None]
        goodput_values = [float(row["joint_slo_output_tok_s"])
                          for row in admission_refine]
        panel = figure.panel(
            25, 46, 1170, 560,
            title="One seed per cell; points are screening measurements",
            xlabel="Rejected requests (% of arrivals)",
            ylabel="Joint-SLO output goodput (tokens/s)",
            xlim=(0.0, max(10.0, min(100.0, max(reject_values) + 5.0))),
            ylim=_span(goodput_values, include_zero=True))
        pressure_colors = {0.5: "#5969a8", 0.7: "#2d708e",
                           0.85: "#3b9b73", 1.0: "#d17c2f"}
        candidate_markers = {
            (500.0, 5000.0): "circle",
            (2000.0, 10000.0): "square",
        }
        for candidate, marker in candidate_markers.items():
            target, budget = candidate
            for pressure in (0.5, 0.7, 0.85, 1.0):
                subset = sorted(
                    (row for row in admission_refine
                     if _number(row.get("target_ttft_ms")) == target
                     and _number(row.get("work_budget_ms")) == budget
                     and _number(row.get("soft_pressure")) == pressure),
                    key=lambda row: float(row["max_wait_ms"]))
                if not subset:
                    continue
                points = [(float(row["reject_percent"]),
                           float(row["joint_slo_output_tok_s"]))
                          for row in subset]
                tooltips = [
                    f"target={target:g} ms; work={budget:g} ms; "
                    f"soft={pressure:g}; max_wait={row['max_wait_ms']:g} ms; "
                    f"reject={row['reject_percent']:.1f}%; "
                    f"goodput={row['joint_slo_output_tok_s']:.1f} tok/s"
                    for row in subset
                ]
                panel.add_series(
                    points, color=pressure_colors[pressure],
                    label=f"target/work {target:g}/{budget:g}, soft {pressure:g}",
                    connect=False, marker=marker, tooltips=tooltips)
        frontier: list[dict[str, Any]] = []
        best_goodput = -math.inf
        for row in sorted(admission_refine,
                          key=lambda item: float(item["reject_percent"])):
            goodput = float(row["joint_slo_output_tok_s"])
            if goodput >= best_goodput:
                frontier.append(row)
                best_goodput = goodput
        if len(frontier) > 1:
            panel.add_series(
                [(float(row["reject_percent"]),
                  float(row["joint_slo_output_tok_s"])) for row in frontier],
                color="#263238", label="screening Pareto frontier",
                connect=True, marker="triangle", dashed=True)
        figure.save(output_dir / "admission_soft_wait.svg")

    if kv_rows:
        bf16 = [row for row in kv_rows
                if row["source"] == "kv_auto_no_swap_warmed.csv"
                and row["prompt_length_mode"] == "synthetic_exact_length"]
        fp8 = [row for row in kv_rows
               if row["source"] == "kv_fp8_no_swap_warmed.csv"
               and row["prompt_length_mode"] == "synthetic_exact_length"]
        left_values = [float(row["ttft_p99_ms"]) for row in bf16
                       if row.get("ttft_p99_ms") is not None]
        left_values.extend(float(row["ttft_p99_ms"]) for row in kv_rows
                           if row["source"] in {
                               "kv_auto_swap_4096_highconc.csv",
                               "kv_auto_no_swap_4096_highconc.csv"}
                           and row.get("ttft_p99_ms") is not None)
        right_values = [float(row["ttft_p99_ms"]) for row in fp8
                        if row.get("ttft_p99_ms") is not None]
        figure = _SvgFigure("KV capacity and service boundary",
                            width=1320, height=620)
        left = figure.panel(
            18, 48, 645, 535,
            title="BF16 KV · synthetic exact-length prompts",
            xlabel="Simultaneous requests", ylabel="TTFT p99 (ms)",
            xlim=(1, 64), ylim=(0, max(left_values, default=2000) * 1.08),
            xticks=[(x, str(x)) for x in (1, 2, 4, 8, 16, 32, 48, 64)])
        context_colors = {512: "#2d708e", 1024: "#3b9b73",
                          2048: "#e1a340", 4096: "#a63a50"}
        for context in (512, 1024, 2048, 4096):
            subset = sorted((row for row in bf16
                             if row["context_length"] == context),
                            key=lambda row: row["concurrency"])
            if subset:
                left.add_series(
                    [(float(row["concurrency"]), float(row["ttft_p99_ms"]))
                     for row in subset if row.get("ttft_p99_ms") is not None],
                    color=context_colors[context], label=f"{context} tokens")
        for source, label, marker in (
                ("kv_auto_no_swap_4096_highconc.csv", "4096 no-swap extension",
                 "circle"),
                ("kv_auto_swap_4096_highconc.csv", "4096 swap enabled", "square")):
            subset = sorted((row for row in kv_rows if row["source"] == source),
                            key=lambda row: row["concurrency"])
            if subset:
                left.add_series(
                    [(float(row["concurrency"]), float(row["ttft_p99_ms"]))
                     for row in subset if row.get("ttft_p99_ms") is not None],
                    color=context_colors[4096], label=label, marker=marker,
                    dashed=True)
        left.add_hline(2000, color="#555555", label="2 s TTFT target")

        right = figure.panel(
            675, 48, 625, 535, title="FP8 E4M3 KV · no swap",
            xlabel="Simultaneous requests", ylabel="TTFT p99 (ms)",
            xlim=(8, 32), ylim=(0, max(right_values, default=2000) * 1.08),
            xticks=[(8, "8"), (16, "16"), (32, "32")])
        for context in (2048, 4096):
            subset = sorted((row for row in fp8
                             if row["context_length"] == context),
                            key=lambda row: row["concurrency"])
            if subset:
                right.add_series(
                    [(float(row["concurrency"]), float(row["ttft_p99_ms"]))
                     for row in subset if row.get("ttft_p99_ms") is not None],
                    color=context_colors[context], label=f"{context} tokens")
        right.add_hline(2000, color="#555555", label="2 s TTFT target")
        figure.save(output_dir / "kv_capacity.svg")

        real = [row for row in kv_rows
                if row["source"] == "kv_real_prompts_no_swap.csv"]
        if real:
            values = [float(row["ttft_p99_ms"]) for row in real
                      if row.get("ttft_p99_ms") is not None]
            figure = _SvgFigure("ShareGPT prompt buckets · BF16 KV · no swap",
                                width=1000, height=590)
            panel = figure.panel(
                30, 46, 940, 520,
                title="Real prompt lengths grouped by maximum bucket",
                xlabel="Simultaneous requests", ylabel="TTFT p99 (ms)",
                xlim=_span([float(row["concurrency"]) for row in real]),
                ylim=(0, max(values, default=2000) * 1.08))
            for context in sorted({int(row["context_length"]) for row in real}):
                subset = sorted((row for row in real
                                 if int(row["context_length"]) == context),
                                key=lambda row: row["concurrency"])
                panel.add_series(
                    [(float(row["concurrency"]), float(row["ttft_p99_ms"]))
                     for row in subset if row.get("ttft_p99_ms") is not None],
                    color=context_colors.get(context, "#555555"),
                    label=f"≤{context} tokens")
            panel.add_hline(2000, color="#555555", label="2 s TTFT target")
            figure.save(output_dir / "kv_real_prompts.svg")

    micro = [row for row in micro_rows if row["variant"] != "loop"
             and _number(row.get("speedup_vs_loop")) is not None]
    if micro:
        figure = _SvgFigure("INT4 MoE microbenchmark · DeepSeek-V2-Lite",
                            width=1000, height=590)
        panel = figure.panel(
            30, 46, 940, 520, title="CUDA-event median speedup over expert loop",
            xlabel="MoE input rows", ylabel="Speedup vs loop (×)",
            xlim=(1, 128), ylim=_span(
                [float(row["speedup_vs_loop"]) for row in micro],
                include_zero=True, padding=0.08),
            xticks=[(1, "1"), (4, "4"), (16, "16"), (64, "64"),
                    (128, "128")])
        for variant in ("segment_split", "segment_fused"):
            subset = sorted((row for row in micro if row["variant"] == variant),
                            key=lambda row: row["token_batch"])
            panel.add_series(
                [(float(row["token_batch"]), float(row["speedup_vs_loop"]))
                 for row in subset], color=colors[variant], label=variant,
                marker="square" if variant == "segment_fused" else "circle")
        panel.add_hline(1, color="#555555", label="loop baseline")
        figure.save(output_dir / "moe_micro_speedup.svg")

    if online_rows:
        groups = sorted({(int(row["tpot_slo_ms"]),
                          int(row["arrival_rate_req_s"]),
                          int(row["output_tokens_cap"])) for row in online_rows})
        position = {group: index for index, group in enumerate(groups)}
        labels = [f"{tpot}/{rate}/{length}" for tpot, rate, length in groups]
        xlim = (-0.5, max(0.5, len(groups) - 0.5))
        figure = _SvgFigure("INT4 MoE fused/split · paired real-prompt traces",
                            width=1500, height=950)
        for row_index, (field, label) in enumerate((
                ("throughput_delta_percent", "Fused throughput delta (%)"),
                ("joint_slo_output_delta_tok_s",
                 "Joint-SLO output goodput delta (tokens/s)"))):
            values = [_number(row.get(field)) for row in online_rows]
            panel = figure.panel(
                28, 45 + row_index * 425, 1440, 410,
                title=(label if row_index == 0 else
                       "Goodput depends on the SLO threshold"),
                xlabel="TPOT target / arrival rate / output cap",
                ylabel=label, xlim=xlim,
                ylim=_span([v for v in values if v is not None],
                           include_zero=True, padding=0.10),
                xticks=[(index, text) for index, text in enumerate(labels)],
                rotate_xticks=True)
            for target, color in ((100, "#2d708e"), (150, "#d17c2f")):
                subset = [row for row in online_rows
                          if int(row["tpot_slo_ms"]) == target]
                points: list[tuple[float, float]] = []
                tooltips: list[str] = []
                for row in subset:
                    group = (int(row["tpot_slo_ms"]),
                             int(row["arrival_rate_req_s"]),
                             int(row["output_tokens_cap"]))
                    x = position[group] + (int(row["seed"]) % 3 - 1) * 0.08
                    value = _number(row.get(field))
                    if value is None:
                        continue
                    points.append((x, value))
                    tooltips.append(
                        f"TPOT {target} ms, rate {group[1]} req/s, "
                        f"output cap {group[2]}, seed {row['seed']}: {value:.3f}")
                panel.add_series(points, color=color,
                                 label=f"TPOT {target} ms", connect=False,
                                 tooltips=tooltips)
            panel.add_hline(0, color="#555555", label="no change")
        figure.save(output_dir / "moe_online_deltas.svg")

    content = [row for row in spec_rows if row["mode"] == "ngram"
               and row["source"] == "spec_content_batch8_g4_seed012.json"]
    if content:
        styles = sorted({str(row["style"]) for row in content})
        positions = {style: index for index, style in enumerate(styles)}
        ticks = [(positions[style], style) for style in styles]
        figure = _SvgFigure("N-gram draft results by content class",
                            width=1220, height=610)
        for index, (field, label) in enumerate((
                ("alpha", "Accepted / proposed draft tokens (α)"),
                ("speedup_vs_none", "Output throughput speedup (×)"))):
            values = [_number(row.get(field)) for row in content]
            panel = figure.panel(
                20 + index * 600, 48, 585, 530,
                title=("Acceptance by content class" if index == 0
                       else "N-gram vs baseline"),
                xlabel="Prompt / output class", ylabel=label,
                xlim=(-0.5, len(styles) - 0.5),
                ylim=_span([v for v in values if v is not None],
                           include_zero=True, padding=0.10),
                xticks=ticks, rotate_xticks=True)
            points = [(positions[str(row["style"])]
                       + (int(row["seed"]) % 3 - 1) * 0.08,
                       float(row[field])) for row in content
                      if _number(row.get(field)) is not None]
            panel.add_series(points, color=("#2d708e" if index == 0
                                            else "#d17c2f"),
                             connect=False)
            if field == "speedup_vs_none":
                panel.add_hline(1, color="#555555", label="baseline")
        figure.save(output_dir / "spec_content.svg")

    sweep = [row for row in spec_rows if row["mode"] == "ngram"
             and row["source"] == "spec_batch_gamma_free_repeat_seed012.json"]
    if sweep:
        figure = _SvgFigure("N-gram acceptance and speedup by batch and draft length",
                            width=1220, height=610)
        style_color = {"free": "#2d708e", "repeat": "#d17c2f"}
        gamma_marker = {2: "circle", 4: "square"}
        groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in sweep:
            groups[(str(row["style"]), int(row["max_draft_len"]),
                    int(row["batch_size"]))].append(row)
        for index, (field, label) in enumerate((
                ("speedup_vs_none", "Output throughput speedup (×)"),
                ("alpha", "Acceptance ratio α"))):
            values = [_number(row.get(field)) for row in sweep]
            panel = figure.panel(
                20 + index * 600, 48, 585, 530,
                title=("Speedup vs baseline" if index == 0
                       else "Draft acceptance"),
                xlabel="Batch size", ylabel=label,
                xlim=(1, 32), ylim=_span([v for v in values if v is not None],
                                         include_zero=True, padding=0.10),
                xticks=[(1, "1"), (8, "8"), (32, "32")])
            for style in ("free", "repeat"):
                for gamma in (2, 4):
                    points = []
                    for batch in (1, 8, 32):
                        entries = groups.get((style, gamma, batch), [])
                        measured = [_number(row.get(field)) for row in entries]
                        measured = [value for value in measured if value is not None]
                        if measured:
                            points.append((batch, statistics.median(measured)))
                    if points:
                        panel.add_series(
                            points, color=style_color[style],
                            label=f"{style}, γ={gamma}",
                            marker=gamma_marker[gamma])
            if field == "speedup_vs_none":
                panel.add_hline(1, color="#555555", label="baseline")
        figure.save(output_dir / "spec_batch_gamma.svg")

    repeat = [row for row in spec_rows if row["mode"] == "ngram"
              and _number(row.get("alpha")) is not None
              and _number(row.get("fourgram_repeat_rate_mean")) is not None]
    if repeat:
        styles = sorted({str(row["style"]) for row in repeat})
        palette = ["#2d708e", "#3b9b73", "#e1a340", "#a63a50",
                   "#5969a8", "#8c6bb1", "#bf6b5d"]
        figure = _SvgFigure("N-gram acceptance vs generated 4-gram repetition",
                            width=1000, height=590)
        rho = spec_correlation.get("alpha_vs_fourgram_repeat_rate_spearman")
        title = (f"Spearman ρ={rho:.2f}" if rho is not None else
                 "Each marker is one generated output")
        panel = figure.panel(
            30, 46, 940, 520, title=title,
            xlabel="Generated-output 4-gram repeat rate",
            ylabel="Accepted / proposed draft tokens (α)",
            xlim=(0, 1), ylim=(0, 1.05))
        for index, style in enumerate(styles):
            subset = [row for row in repeat if str(row["style"]) == style]
            points = [(float(row["fourgram_repeat_rate_mean"]),
                       float(row["alpha"])) for row in subset]
            panel.add_series(points, color=palette[index % len(palette)],
                             label=style, connect=False)
        figure.save(output_dir / "spec_acceptance_repeat.svg")


def _plot_scheduling(plt, rows: list[dict[str, Any]], paired: list[dict[str, Any]],
                     output: Path) -> None:
    if not rows or not paired:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = {"baseline": "#355f8a", "all_on": "#d17c2f"}
    metrics = (
        ("output_tok_s", "Output throughput (tokens/s)"),
        ("joint_slo_output_tokens_per_s", "Joint-SLO output goodput (tokens/s)"),
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    for axis, (metric, label) in zip(axes, metrics):
        for seed in seeds:
            pair = {row["variant"]: row for row in rows
                    if int(row["seed"]) == seed}
            if "baseline" not in pair or "all_on" not in pair:
                continue
            y = [pair[name][metric] for name in ("baseline", "all_on")]
            axis.plot([0, 1], y, color="#b7bec7", linewidth=1.0, zorder=1)
            for x, name in enumerate(("baseline", "all_on")):
                axis.scatter(x, y[x], color=colors[name], s=42, zorder=2)
        axis.set_xticks([0, 1], ["baseline", "all_on"])
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
    axes[0].set_title("ShareGPT · 500 requests · 8 req/s")
    axes[1].set_title("All requests count toward SLO goodput")
    fig.suptitle("Paired scheduling results across five arrival seeds", y=1.02)
    fig.tight_layout()


def _plot_admission(plt, rows: list[dict[str, Any]], output: Path) -> None:
    data = [row for row in rows if int(row.get("requests", 0)) == 500
            and row.get("joint_slo_output_tok_s") is not None]
    if not data:
        return
    markers = {
        "dynamic_off": "o",
        "dynamic_on_no_cache": "s",
        "dynamic_on_cache_aware": "^",
    }
    fig, axis = plt.subplots(figsize=(10, 6))
    points = None
    for variant, marker in markers.items():
        subset = [row for row in data if row["variant"] == variant]
        if not subset:
            continue
        points = axis.scatter(
            [row["reject_percent"] for row in subset],
            [row["joint_slo_output_tok_s"] for row in subset],
            c=[row["ttft_p99_ms"] or 0 for row in subset],
            cmap="viridis", vmin=0, vmax=max(
                2000.0, max((row["ttft_p99_ms"] or 0) for row in data)),
            marker=marker, s=70, edgecolors="white", linewidths=0.6,
            label=variant,
        )
    for row in data:
        axis.annotate(
            f"{row['offered_rate_req_s']:g}r/{row['prefix_percent']}%",
            (row["reject_percent"], row["joint_slo_output_tok_s"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )
    if points is not None:
        colorbar = fig.colorbar(points, ax=axis)
        colorbar.set_label("TTFT p99 (ms)")
    axis.set_xlabel("Rejected requests (% of arrivals)")
    axis.set_ylabel("Joint-SLO output goodput (tokens/s)")
    axis.set_title("Admission operating points · 500-request ShareGPT trace")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()


def _plot_admission_calibration(plt, rows: list[dict[str, Any]],
                               output: Path) -> None:
    data = [row for row in rows
            if row.get("phase") == "target_work_grid"
            and int(row.get("requests", 0)) >= 300
            and _number(row.get("offered_rate_req_s")) == 96.0
            and _number(row.get("slo_tpot_ms")) == 20.0]
    aware = [row for row in data
             if row.get("variant") == "dynamic_on_cache_aware"
             and _number(row.get("joint_slo_output_tok_s")) is not None]
    if not aware:
        return
    colors = {2000.0: "#2d708e", 5000.0: "#3b9b73",
              10000.0: "#d17c2f"}
    fig, axis = plt.subplots(figsize=(10, 6))
    for budget in sorted({float(row["work_budget_ms"]) for row in aware
                          if row.get("work_budget_ms") is not None}):
        subset = [row for row in aware
                  if float(row["work_budget_ms"]) == budget]
        axis.scatter(
            [row["reject_percent"] for row in subset],
            [row["joint_slo_output_tok_s"] for row in subset],
            color=colors.get(budget, "#5969a8"), marker="o", s=75,
            edgecolors="white", linewidths=0.6,
            label=f"cache-aware · work {budget:g} ms")
        for row in subset:
            axis.annotate(
                f"T{float(row['target_ttft_ms']):g}",
                (row["reject_percent"], row["joint_slo_output_tok_s"]),
                xytext=(4, 4), textcoords="offset points", fontsize=7)
    blind = [row for row in data
             if row.get("variant") == "dynamic_on_no_cache"
             and _number(row.get("joint_slo_output_tok_s")) is not None]
    if blind:
        axis.scatter(
            [row["reject_percent"] for row in blind],
            [row["joint_slo_output_tok_s"] for row in blind],
            color="#8c6bb1", marker="^", s=60, label="cache-blind")
    off = [row for row in data
           if row.get("variant") == "dynamic_off"
           and _number(row.get("joint_slo_output_tok_s")) is not None]
    if off:
        axis.scatter(
            [statistics.median(row["reject_percent"] for row in off)],
            [statistics.median(row["joint_slo_output_tok_s"] for row in off)],
            color="#355f8a", marker="s", s=95, label="admission off")
    frontier: list[dict[str, Any]] = []
    best_goodput = -math.inf
    for row in sorted(aware, key=lambda item: float(item["reject_percent"])):
        goodput = float(row["joint_slo_output_tok_s"])
        if goodput >= best_goodput:
            frontier.append(row)
            best_goodput = goodput
    if len(frontier) > 1:
        axis.plot([row["reject_percent"] for row in frontier],
                  [row["joint_slo_output_tok_s"] for row in frontier],
                  color="#263238", linestyle="--", linewidth=1.2,
                  label="cache-aware Pareto frontier")
    axis.set_xlabel("Rejected requests (% of arrivals)")
    axis.set_ylabel("Joint-SLO output goodput (tokens/s)")
    axis.set_title("Admission target/work grid · 96 req/s · TPOT ≤ 20 ms")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()


def _plot_admission_refinement(plt, rows: list[dict[str, Any]],
                               output: Path) -> None:
    data = [row for row in rows
            if row.get("phase") == "soft_wait_refinement"
            and int(row.get("requests", 0)) >= 300
            and _number(row.get("offered_rate_req_s")) == 96.0
            and _number(row.get("slo_tpot_ms")) == 80.0
            and row.get("variant") == "dynamic_on_cache_aware"
            and _number(row.get("joint_slo_output_tok_s")) is not None]
    if not data:
        return
    pressure_colors = {0.5: "#5969a8", 0.7: "#2d708e",
                       0.85: "#3b9b73", 1.0: "#d17c2f"}
    candidate_markers = {
        (500.0, 5000.0): "o",
        (2000.0, 10000.0): "s",
    }
    fig, axis = plt.subplots(figsize=(11, 6))
    for candidate, marker in candidate_markers.items():
        target, budget = candidate
        for pressure in (0.5, 0.7, 0.85, 1.0):
            subset = sorted(
                (row for row in data
                 if _number(row.get("target_ttft_ms")) == target
                 and _number(row.get("work_budget_ms")) == budget
                 and _number(row.get("soft_pressure")) == pressure),
                key=lambda row: float(row["max_wait_ms"]))
            if not subset:
                continue
            axis.scatter(
                [row["reject_percent"] for row in subset],
                [row["joint_slo_output_tok_s"] for row in subset],
                color=pressure_colors[pressure], marker=marker, s=76,
                edgecolors="white", linewidths=0.6,
                label=f"target/work {target:g}/{budget:g}, soft {pressure:g}")
    frontier: list[dict[str, Any]] = []
    best_goodput = -math.inf
    for row in sorted(data, key=lambda item: float(item["reject_percent"])):
        goodput = float(row["joint_slo_output_tok_s"])
        if goodput >= best_goodput:
            frontier.append(row)
            best_goodput = goodput
    if len(frontier) > 1:
        axis.plot([row["reject_percent"] for row in frontier],
                  [row["joint_slo_output_tok_s"] for row in frontier],
                  color="#263238", linestyle="--", linewidth=1.2,
                  label="single-seed screening Pareto frontier")
    axis.set_xlabel("Rejected requests (% of arrivals)")
    axis.set_ylabel("Joint-SLO output goodput (tokens/s)")
    axis.set_title("Soft-pressure / max-wait refinement · 96 req/s · TPOT ≤ 80 ms")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()


def _plot_kv(plt, rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = {512: "#2d708e", 1024: "#3b9b73", 2048: "#e1a340",
              4096: "#a63a50"}
    base = [row for row in rows if row["source"] == "kv_auto_no_swap_warmed.csv"
            and row["prompt_length_mode"] == "synthetic_exact_length"]
    for context in (512, 1024, 2048, 4096):
        subset = sorted((r for r in base if r["context_length"] == context),
                        key=lambda r: r["concurrency"])
        if subset:
            axes[0].plot([r["concurrency"] for r in subset],
                         [r["ttft_p99_ms"] for r in subset], marker="o",
                         color=colors[context], label=f"{context} tokens")
    high = [r for r in rows if r["context_length"] == 4096
            and r["source"] in {
                "kv_auto_swap_4096_highconc.csv",
                "kv_auto_no_swap_4096_highconc.csv",
            }]
    for source, label, marker in (
        ("kv_auto_no_swap_4096_highconc.csv", "4096 no-swap extension", "x"),
        ("kv_auto_swap_4096_highconc.csv", "4096 swap enabled", "D"),
    ):
        subset = sorted((r for r in high if r["source"] == source),
                        key=lambda r: r["concurrency"])
        if subset:
            axes[0].plot([r["concurrency"] for r in subset],
                         [r["ttft_p99_ms"] for r in subset], marker=marker,
                         linestyle="--", color=colors[4096], label=label)
    axes[0].set_title("BF16 KV · synthetic exact-length prompts")
    axes[0].set_ylabel("TTFT p99 (ms)")
    axes[0].set_xlabel("Simultaneous requests")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16, 32, 48, 64],
                       ["1", "2", "4", "8", "16", "32", "48", "64"])
    axes[0].axhline(2000, color="#555555", linestyle=":", linewidth=1,
                    label="2 s TTFT target")
    axes[0].grid(color="#d7dce1", linewidth=0.7)
    axes[0].legend(fontsize=7, frameon=False)

    fp8 = [r for r in rows if r["source"] == "kv_fp8_no_swap_warmed.csv"
           and r["prompt_length_mode"] == "synthetic_exact_length"]
    for context in (2048, 4096):
        subset = sorted((r for r in fp8 if r["context_length"] == context),
                        key=lambda r: r["concurrency"])
        if subset:
            axes[1].plot([r["concurrency"] for r in subset],
                         [r["ttft_p99_ms"] for r in subset], marker="o",
                         color=colors[context], label=f"{context} tokens")
    axes[1].set_title("FP8 E4M3 KV · no swap")
    axes[1].set_ylabel("TTFT p99 (ms)")
    axes[1].set_xlabel("Simultaneous requests")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([8, 16, 32], ["8", "16", "32"])
    axes[1].axhline(2000, color="#555555", linestyle=":", linewidth=1)
    axes[1].grid(color="#d7dce1", linewidth=0.7)
    axes[1].legend(frameon=False)
    fig.suptitle("KV capacity and service boundary")
    fig.tight_layout()


def _plot_real_kv(plt, rows: list[dict[str, Any]], output: Path) -> None:
    data = [row for row in rows if row["source"] == "kv_real_prompts_no_swap.csv"]
    if not data:
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    colors = {512: "#2d708e", 1024: "#3b9b73", 2048: "#a63a50"}
    for context in sorted({row["context_length"] for row in data}):
        subset = sorted((r for r in data if r["context_length"] == context),
                        key=lambda r: r["concurrency"])
        axis.plot([r["concurrency"] for r in subset],
                  [r["ttft_p99_ms"] for r in subset], marker="o",
                  color=colors.get(context, "#555555"),
                  label=f"≤{context} token bucket")
    axis.axhline(2000, color="#555555", linestyle=":", linewidth=1,
                 label="2 s TTFT target")
    axis.set_xlabel("Simultaneous requests")
    axis.set_ylabel("TTFT p99 (ms)")
    axis.set_title("ShareGPT prompt-length buckets · BF16 KV · no swap")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()


def _plot_micro(plt, rows: list[dict[str, Any]], output: Path) -> None:
    data = [row for row in rows if row["variant"] != "loop"
            and row.get("speedup_vs_loop") is not None]
    if not data:
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    for variant, marker, color in (
        ("segment_split", "o", "#2d708e"),
        ("segment_fused", "s", "#d17c2f"),
    ):
        subset = sorted((r for r in data if r["variant"] == variant),
                        key=lambda r: r["token_batch"])
        axis.plot([r["token_batch"] for r in subset],
                  [r["speedup_vs_loop"] for r in subset], marker=marker,
                  color=color, label=variant)
    axis.axhline(1, color="#555555", linestyle=":", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 4, 16, 64, 128], ["1", "4", "16", "64", "128"])
    axis.set_xlabel("MoE input rows")
    axis.set_ylabel("Speedup vs per-expert loop (×)")
    axis.set_title("DeepSeek-V2-Lite · INT4 group64 · CUDA-event medians")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False)
    fig.tight_layout()


def _plot_online_moe(plt, rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    groups = sorted({(int(row["tpot_slo_ms"]),
                      int(row["arrival_rate_req_s"]),
                      int(row["output_tokens_cap"])) for row in rows})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x_positions = {group: index for index, group in enumerate(groups)}
    for row in rows:
        group = (int(row["tpot_slo_ms"]), int(row["arrival_rate_req_s"]),
                 int(row["output_tokens_cap"]))
        x = x_positions[group] + (int(row["seed"]) % 3 - 1) * 0.08
        axes[0].scatter(x, row["throughput_delta_percent"],
                        c="#2d708e" if int(row["tpot_slo_ms"]) == 100 else "#d17c2f",
                        s=28)
        axes[1].scatter(x, row["joint_slo_output_delta_tok_s"],
                        c="#2d708e" if int(row["tpot_slo_ms"]) == 100 else "#d17c2f",
                        s=28)
    for axis in axes:
        axis.axhline(0, color="#555555", linestyle=":", linewidth=1)
        axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
    axes[0].set_ylabel("Fused throughput delta (%)")
    axes[1].set_ylabel("Joint-SLO goodput delta (tokens/s)")
    axes[1].set_xlabel("TPOT target / arrival rate / output cap")
    axes[1].set_xticks(list(x_positions.values()),
                       [f"{tpot}/{rate}/{length}"
                        for tpot, rate, length in groups], rotation=65,
                       ha="right", fontsize=7)
    axes[0].set_title("DeepSeek-V2-Lite · paired ShareGPT prompt traces")
    fig.tight_layout()


def _plot_spec_content(plt, rows: list[dict[str, Any]], output: Path) -> None:
    data = [row for row in rows if row["mode"] == "ngram"
            and row["source"] == "spec_content_batch8_g4_seed012.json"]
    if not data:
        return
    styles = sorted({row["style"] for row in data})
    positions = {style: index for index, style in enumerate(styles)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for row in data:
        x = positions[row["style"]] + (int(row["seed"]) % 3 - 1) * 0.08
        axes[0].scatter(x, row["alpha"], color="#2d708e", s=38)
        axes[1].scatter(x, row["speedup_vs_none"], color="#d17c2f", s=38)
    axes[0].set_ylabel("Accepted / proposed draft tokens (α)")
    axes[0].set_title("Acceptance by content class")
    axes[1].axhline(1, color="#555555", linestyle=":", linewidth=1)
    axes[1].set_ylabel("Output throughput speedup (×)")
    axes[1].set_title("N-gram vs baseline")
    for axis in axes:
        axis.set_xticks(list(positions.values()), styles, rotation=30, ha="right")
        axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
    fig.tight_layout()


def _plot_spec_batch_gamma(plt, rows: list[dict[str, Any]], output: Path) -> None:
    data = [row for row in rows if row["mode"] == "ngram"
            and row["source"] == "spec_batch_gamma_free_repeat_seed012.json"]
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    style_colors = {"free": "#2d708e", "repeat": "#d17c2f"}
    gamma_markers = {2: "o", 4: "s"}
    for style in ("free", "repeat"):
        for gamma in (2, 4):
            subset = [row for row in data if row["style"] == style
                      and int(row["max_draft_len"]) == gamma]
            if not subset:
                continue
            grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in subset:
                grouped[int(row["batch_size"])].append(row)
            batches = sorted(grouped)
            label = f"{style}, γ={gamma}"
            for axis, field in zip(axes, ("speedup_vs_none", "alpha")):
                medians = [statistics.median(
                    float(row[field]) for row in grouped[batch])
                    for batch in batches]
                axis.plot(batches, medians, marker=gamma_markers[gamma],
                          color=style_colors[style], label=label)
    axes[0].axhline(1, color="#555555", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Output throughput speedup (×)")
    axes[1].set_ylabel("Acceptance ratio α")
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 8, 32], ["1", "8", "32"])
        axis.set_xlabel("Batch size")
        axis.grid(axis="y", color="#d7dce1", linewidth=0.7)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("N-gram acceptance and speedup vs batch and draft length")
    fig.tight_layout()


def _plot_spec_repeat(plt, rows: list[dict[str, Any]], output: Path,
                      correlation: dict[str, Any]) -> None:
    data = [row for row in rows if row["mode"] == "ngram"
            and row.get("alpha") is not None
            and row.get("fourgram_repeat_rate_mean") is not None]
    if not data:
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    for style in sorted({str(row["style"]) for row in data}):
        subset = [row for row in data if row["style"] == style]
        axis.scatter([row["fourgram_repeat_rate_mean"] for row in subset],
                     [row["alpha"] for row in subset], s=34, label=style)
    rho = correlation.get("alpha_vs_fourgram_repeat_rate_spearman")
    axis.set_xlabel("Generated-output 4-gram repeat rate")
    axis.set_ylabel("Accepted / proposed draft tokens (α)")
    axis.set_title(f"N-gram acceptance vs generated repetition (Spearman ρ={rho:.2f})"
                   if rho is not None else
                   "N-gram acceptance vs generated repetition")
    axis.grid(color="#d7dce1", linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()


def _write_markdown(path: Path, *, data_dir: Path,
                    scheduling_rows: list[dict[str, Any]],
                    scheduling_ci: dict[str, Any],
                    scheduling_one_factor_rows: list[dict[str, Any]],
                    admission_rows: list[dict[str, Any]],
                    admission_calibration_rows: list[dict[str, Any]],
                    admission_validation_pairs: list[dict[str, Any]],
                    kv_rows: list[dict[str, Any]],
                    fp8_validation_rows: list[dict[str, Any]],
                    fp8_validation_summary: dict[str, Any],
                    micro_rows: list[dict[str, Any]],
                    micro_correctness: dict[str, Any],
                    online_rows: list[dict[str, Any]],
                    online_summary: dict[str, Any],
                    online_repeat_rows: list[dict[str, Any]],
                    online_repeat_summary: dict[str, Any],
                    spec_rows: list[dict[str, Any]],
                    spec_correlation: dict[str, Any],
                    chart_ext: str,
                    chart_prefix: str = "") -> None:
    def chart(name: str, label: str) -> str:
        return f"![{label}]({chart_prefix}{name}.{chart_ext})"

    def fmt(value: Any, digits: int = 2) -> str:
        number = _number(value)
        return "—" if number is None else f"{number:.{digits}f}"

    def ci_text(key: str) -> str:
        row = scheduling_ci.get(key, {})
        if row.get("mean") is None:
            return "无配对数据"
        return (f"{row['mean']:+.2f}%（95% bootstrap CI "
                f"{row['ci95_low']:+.2f}% 至 {row['ci95_high']:+.2f}%）")

    full_admission = [row for row in admission_rows
                      if int(row["requests"]) == 500]
    kv_bf16 = [row for row in kv_rows
               if row["source"] == "kv_auto_no_swap_warmed.csv"]
    kv_fp8 = [row for row in kv_rows
              if row["source"] == "kv_fp8_no_swap_warmed.csv"]
    kv_real = [row for row in kv_rows
               if row["source"] == "kv_real_prompts_no_swap.csv"]
    kv_swap_high = [row for row in kv_rows
                    if row["source"] == "kv_auto_swap_4096_highconc.csv"]
    micro_speedups = [float(row["speedup_vs_loop"])
                      for row in micro_rows
                      if row.get("variant") == "segment_fused"
                      and row.get("speedup_vs_loop") is not None]
    online_parity = [row for row in online_rows
                     if row.get("output_parity") is True]
    online_deltas = [float(row["throughput_delta_percent"])
                     for row in online_rows
                     if row.get("throughput_delta_percent") is not None]
    online_goodput_percent = [float(row["joint_slo_output_delta_percent"])
                              for row in online_rows
                              if row.get("joint_slo_output_delta_percent")
                              is not None]
    online_losses = sum(value < 0 for value in online_deltas)
    online_wins = sum(value > 0 for value in online_deltas)
    spec_ngram = [row for row in spec_rows if row["mode"] == "ngram"]
    spec_content = [row for row in spec_ngram
                    if row["source"] == "spec_content_batch8_g4_seed012.json"]
    spec_sweep = [row for row in spec_ngram
                  if row["source"] == "spec_batch_gamma_free_repeat_seed012.json"]
    metadata: dict[str, Any] = {}
    repro_path = data_dir / "moe_int4_mla_assemble_repro_seed42.json"
    if repro_path.exists():
        metadata = _read_json(repro_path).get("meta", {})
    revision = metadata.get("git_revision", "unknown")
    revision = str(revision)[:12]

    scheduled_by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scheduling_rows:
        scheduled_by_seed[int(row["seed"])][str(row["variant"])] = row
    output_losses = sum(
        _number(rows.get("all_on", {}).get("output_tok_s")) is not None
        and _number(rows.get("baseline", {}).get("output_tok_s")) is not None
        and float(rows["all_on"]["output_tok_s"])
        < float(rows["baseline"]["output_tok_s"])
        for rows in scheduled_by_seed.values())
    goodput_losses = sum(
        _number(rows.get("all_on", {}).get(
            "joint_slo_output_tokens_per_s")) is not None
        and _number(rows.get("baseline", {}).get(
            "joint_slo_output_tokens_per_s")) is not None
        and float(rows["all_on"]["joint_slo_output_tokens_per_s"])
        < float(rows["baseline"]["joint_slo_output_tokens_per_s"])
        for rows in scheduled_by_seed.values())
    one_factor = {str(row["variant"]): row
                  for row in scheduling_one_factor_rows}

    admission_24 = [row for row in full_admission
                    if float(row["offered_rate_req_s"]) == 24.0]
    admission_lines = []
    for prefix in (0, 20, 50):
        off = next((row for row in admission_24
                    if int(row["prefix_percent"]) == prefix
                    and row["variant"] == "dynamic_off"), None)
        aware = next((row for row in admission_24
                      if int(row["prefix_percent"]) == prefix
                      and row["variant"] == "dynamic_on_cache_aware"), None)
        if off and aware:
            admission_lines.append(
                f"| {prefix}% | {off['accepted']}/500 | "
                f"{fmt(off['joint_slo_output_tok_s'], 0)} | "
                f"{aware['accepted']}/500 ({fmt(aware['reject_percent'], 1)}%) | "
                f"{fmt(aware['joint_slo_output_tok_s'], 0)} |")
    high_rate_lines = []
    for rate in (32, 48, 64, 96):
        off = next((row for row in full_admission
                    if float(row["offered_rate_req_s"]) == rate
                    and row["variant"] == "dynamic_off"), None)
        aware = next((row for row in full_admission
                      if float(row["offered_rate_req_s"]) == rate
                      and row["variant"] == "dynamic_on_cache_aware"), None)
        if off and aware:
            high_rate_lines.append(
                f"| {rate} | {fmt(off['accept_percent'], 1)}% / "
                f"{fmt(off['joint_slo_output_tok_s'], 0)} | "
                f"{fmt(aware['accept_percent'], 1)}% / "
                f"{fmt(aware['joint_slo_output_tok_s'], 0)} |")

    calibration_sources: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in admission_calibration_rows:
        if row.get("phase") in {
                "target_work_grid", "soft_wait_refinement",
                "calibration_validation", "heldout_validation"}:
            calibration_sources[str(row["source"])][str(row["variant"])] = row
    admission_grid_lines = []
    admission_refine_lines = []
    admission_validation_lines = []
    source_items = sorted(
        calibration_sources.items(),
        key=lambda item: (
            str(item[1].get("dynamic_on_cache_aware", {}).get("phase", "")),
            _number(item[1].get("dynamic_on_cache_aware", {}).get(
                "target_ttft_ms"), 0.0) or 0.0,
            _number(item[1].get("dynamic_on_cache_aware", {}).get(
                "work_budget_ms"), 0.0) or 0.0,
            _number(item[1].get("dynamic_on_cache_aware", {}).get(
                "soft_pressure"), 0.0) or 0.0,
            _number(item[1].get("dynamic_on_cache_aware", {}).get(
                "max_wait_ms"), 0.0) or 0.0,
            str(item[0]),
        ))
    for _source, variants in source_items:
        aware = variants.get("dynamic_on_cache_aware")
        if aware is None:
            continue
        phase = str(aware.get("phase"))
        if phase == "target_work_grid":
            off = variants.get("dynamic_off", {})
            blind = variants.get("dynamic_on_no_cache", {})
            admission_grid_lines.append(
                f"| {fmt(aware['target_ttft_ms'], 0)} | "
                f"{fmt(aware['work_budget_ms'], 0)} | "
                f"{fmt(aware['reject_percent'], 1)}% | "
                f"{fmt(aware['joint_slo_output_tok_s'], 0)} | "
                f"{fmt(blind.get('joint_slo_output_tok_s'), 0)} | "
                f"{fmt(off.get('joint_slo_output_tok_s'), 0)} | "
                f"{fmt(aware['ttft_p99_ms'], 0)} | "
                f"{fmt(aware['tpot_p99_ms'], 1)} |")
        elif phase == "soft_wait_refinement":
            admission_refine_lines.append(
                f"| {fmt(aware['target_ttft_ms'], 0)} | "
                f"{fmt(aware['work_budget_ms'], 0)} | "
                f"{fmt(aware['soft_pressure'], 2)} | "
                f"{fmt(aware['max_wait_ms'], 0)} | "
                f"{fmt(aware['reject_percent'], 1)}% | "
                f"{fmt(aware['joint_slo_output_tok_s'], 0)} | "
                f"{fmt(aware['ttft_p99_ms'], 0)} | "
                f"{fmt(aware['tpot_p99_ms'], 1)} |")
        elif phase in {"calibration_validation", "heldout_validation"}:
            continue

    admission_validation_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in admission_validation_pairs:
        key = (pair.get("phase"), pair.get("requests"),
               pair.get("offered_rate_req_s"), pair.get("slo_tpot_ms"),
               pair.get("target_ttft_ms"), pair.get("work_budget_ms"),
               pair.get("soft_pressure"), pair.get("max_wait_ms"))
        admission_validation_groups[key].append(pair)
    def admission_validation_sort(item: tuple[tuple[Any, ...],
                                              list[dict[str, Any]]]) -> tuple[Any, ...]:
        key = item[0]
        return (
            0 if key[0] == "calibration_validation" else 1,
            _number(key[3], math.inf),
            _number(key[4], 0.0),
            _number(key[5], 0.0),
            _number(key[6], 0.0),
            _number(key[7], 0.0),
        )

    for key, pairs in sorted(admission_validation_groups.items(),
                              key=admission_validation_sort):
        (phase, request_count, offered_rate, tpot_slo, target, budget,
         pressure, max_wait) = key
        values = lambda field: [float(pair[field]) for pair in pairs
                                if pair.get(field) is not None]
        off_goodputs = values("admission_off_joint_slo_output_tok_s")
        aware_goodputs = values("cache_aware_joint_slo_output_tok_s")
        off_rejects = values("admission_off_reject_percent")
        aware_rejects = values("cache_aware_reject_percent")
        aware_ttft_p99 = values("cache_aware_ttft_p99_ms")
        delta_goodputs = values("joint_slo_output_delta_tok_s")
        delta_rejects = values("reject_delta_percentage_points")
        if not (off_goodputs and aware_goodputs):
            continue
        phase_label = ("calibration" if phase == "calibration_validation"
                       else "held-out")
        goodput_ci = _bootstrap_mean_ci(
            delta_goodputs, seed=20261020 + len(admission_validation_lines))
        reject_ci = _bootstrap_mean_ci(
            delta_rejects, seed=20261030 + len(admission_validation_lines))
        matched_fingerprints = sum(
            pair.get("fingerprint_match") is True for pair in pairs)
        validation_label = (
            f"{phase_label} ({request_count} req, "
            f"{fmt(offered_rate, 0)} req/s, TPOT≤{fmt(tpot_slo, 0)} ms)")
        admission_validation_lines.append(
            f"| {validation_label} | {fmt(target, 0)}/{fmt(budget, 0)}/"
            f"{fmt(pressure, 2)}/{fmt(max_wait, 0)} | "
            f"{len(pairs)} ({matched_fingerprints}/{len(pairs)} fp) | "
            f"{statistics.median(off_rejects):.1f}% / "
            f"{statistics.median(off_goodputs):.0f} | "
            f"{statistics.median(aware_rejects):.1f}% / "
            f"{statistics.median(aware_goodputs):.0f} | "
            f"{goodput_ci['mean']:+.0f} "
            f"({goodput_ci['ci95_low']:+.0f} to "
            f"{goodput_ci['ci95_high']:+.0f}) | "
            f"{reject_ci['mean']:+.1f} pp | "
            f"{statistics.median(aware_ttft_p99):.0f} |")

    admission_grid_candidates = [row for row in admission_calibration_rows
                                 if row.get("phase") == "target_work_grid"
                                 and row.get("variant") ==
                                 "dynamic_on_cache_aware"
                                 and _number(row.get("slo_tpot_ms")) == 20.0
                                 and _number(row.get("joint_slo_output_tok_s"))
                                 is not None]
    best_admission_grid = max(
        admission_grid_candidates,
        key=lambda row: float(row["joint_slo_output_tok_s"]),
        default=None)
    admission_grid_requests = max(
        (int(row["requests"]) for row in admission_grid_candidates),
        default=0)
    admission_grid_rate = next(
        (_number(row.get("offered_rate_req_s"))
         for row in admission_grid_candidates
         if row.get("offered_rate_req_s") is not None), None)
    admission_grid_ttft_slo = next(
        (_number(row.get("slo_ttft_ms")) for row in admission_grid_candidates
         if row.get("slo_ttft_ms") is not None), None)
    admission_grid_tpot_slo = next(
        (_number(row.get("slo_tpot_ms")) for row in admission_grid_candidates
         if row.get("slo_tpot_ms") is not None), None)
    admission_refinement_candidates = [
        row for row in admission_calibration_rows
        if row.get("phase") == "soft_wait_refinement"
        and row.get("variant") == "dynamic_on_cache_aware"
        and int(row.get("requests", 0)) >= 300
        and _number(row.get("offered_rate_req_s")) == 96.0
        and _number(row.get("slo_tpot_ms")) == 80.0
        and _number(row.get("joint_slo_output_tok_s")) is not None
    ]
    best_admission_refinement = max(
        admission_refinement_candidates,
        key=lambda row: float(row["joint_slo_output_tok_s"]),
        default=None)
    least_rejected_refinement = min(
        admission_refinement_candidates,
        key=lambda row: float(row["reject_percent"]),
        default=None)

    blocks_bf16 = sorted({int(row["kv_total_blocks"]) for row in kv_bf16})
    slots_bf16 = sorted({int(row["kv_capacity_token_slots"]) for row in kv_bf16})
    blocks_fp8 = sorted({int(row["kv_total_blocks"]) for row in kv_fp8})
    slots_fp8 = sorted({int(row["kv_capacity_token_slots"]) for row in kv_fp8})
    blocks_real = sorted({int(row["kv_total_blocks"]) for row in kv_real})
    slots_real = sorted({int(row["kv_capacity_token_slots"])
                         for row in kv_real})
    swap_count_high = sum(int(row.get("swaps", 0)) for row in kv_swap_high)
    kv_default_probe = [row for row in kv_rows
                        if row["source"] ==
                        "kv_auto_swap_probe_4096_c32_o512.csv"]
    kv_swap_force = [row for row in kv_rows
                     if row["source"] ==
                     "kv_auto_swap_force_4096_c32_o512_rep3.csv"]
    kv_no_swap_force = [row for row in kv_rows
                        if row["source"] ==
                        "kv_auto_no_swap_force_4096_c32_o512_rep3.csv"]
    kv_swap_force_delta = (
        _percent_delta(kv_swap_force[0].get("output_tok_s"),
                       kv_no_swap_force[0].get("output_tok_s"))
        if kv_swap_force and kv_no_swap_force else None)
    kv_swap_diagnostic_lines = []
    for label, rows in (("swap on; cost comparison off", kv_swap_force),
                        ("swap off; cost comparison off", kv_no_swap_force)):
        if rows:
            row = rows[0]
            kv_swap_diagnostic_lines.append(
                f"| {label} | {fmt(row.get('output_tok_s'), 1)} | "
                f"{fmt(row.get('ttft_p99_ms'), 0)} | "
                f"{fmt(row.get('tpot_p99_ms'), 1)} | "
                f"{fmt(row.get('preemptions'), 0)} | "
                f"{fmt(row.get('swaps'), 0)} | "
                f"{fmt(row.get('recompute_preemptions'), 0)} |")
    kv_default_probe_text = (
        f"默认成本策略诊断点发生 {sum(int(row.get('preemptions', 0)) for row in kv_default_probe)} 次抢占，"
        f"其中重算 {sum(int(row.get('recompute_preemptions', 0)) for row in kv_default_probe)} 次、"
        f"swap {sum(int(row.get('swaps', 0)) for row in kv_default_probe)} 次。")
    bf16_full_context = {
        int(context): max(int(row["max_full_context_requests"])
                          for row in kv_bf16
                          if int(row["context_length"]) == int(context))
        for context in {row["context_length"] for row in kv_bf16}
    }
    fp8_full_context = {
        int(context): max(int(row["max_full_context_requests"])
                          for row in kv_fp8
                          if int(row["context_length"]) == int(context))
        for context in {row["context_length"] for row in kv_fp8}
    }
    bf16_capacity_text = ", ".join(
        f"{context} tokens: {count} requests"
        for context, count in sorted(bf16_full_context.items())) or "—"
    fp8_capacity_text = ", ".join(
        f"{context} tokens: {count} requests"
        for context, count in sorted(fp8_full_context.items())) or "—"

    fp8_lines = []
    for row in fp8_validation_rows:
        if row.get("scale_margin") is None:
            continue
        fp8_lines.append(
            f"| {fmt(row['scale_margin'], 2)} | "
            f"{fmt(row['first_token_agreement_percent'], 1)}% | "
            f"{fmt(row['top1_agreement_percent'], 1)}% | "
            f"{fmt(row['top5_overlap_percent'], 1)}% | "
            f"{fmt(row['kl_ref_to_fp8'], 5)} | "
            f"{fmt(row['max_evaluation_range_utilization'], 3)} | "
            f"{fmt(row['components_over_e4m3_limit'], 0)} |")

    layer_variants = micro_correctness.get("all_layer_variant_checks", {})
    fused_checks = layer_variants.get("segment_fused", {})
    split_checks = layer_variants.get("segment_split", {})
    layer_cases = micro_correctness.get("all_layer_checks", 0)

    spec_content_lines = []
    for style in sorted({str(row["style"]) for row in spec_content}):
        subset = [row for row in spec_content if str(row["style"]) == style]
        alphas = [_number(row.get("alpha")) for row in subset]
        speeds = [_number(row.get("speedup_vs_none")) for row in subset]
        alphas = [value for value in alphas if value is not None]
        speeds = [value for value in speeds if value is not None]
        spec_content_lines.append(
            f"| {style} | {len(subset)} | "
            f"{statistics.median(alphas):.3f} | "
            f"{statistics.median(speeds):.3f}× |")
    spec_sweep_lines = []
    for style in ("free", "repeat"):
        subset = [row for row in spec_sweep if row["style"] == style]
        alphas = [_number(row.get("alpha")) for row in subset]
        speeds = [_number(row.get("speedup_vs_none")) for row in subset]
        alphas = [value for value in alphas if value is not None]
        speeds = [value for value in speeds if value is not None]
        if subset and alphas and speeds:
            spec_sweep_lines.append(
                f"| {style} | {len(subset)} | "
                f"{statistics.median(alphas):.3f} | "
                f"{statistics.median(speeds):.3f}× |")

    lines = [
        "# 实验补充报告（2026-10-15）",
        "",
        f"实验基线提交：`{revision}`（运行时工作区有未提交改动）。原始 JSON/CSV 位于 `results/supplemental/`；汇总表、图和本报告副本由 `python -m benchmarks.supplemental_report` 生成。",
        "",
        "## 环境与负载",
        "",
        "- RTX 5060 Ti 16 GB，TP=1；Python 3.12.13、PyTorch 2.8.0+cu128、CUDA 12.8、Triton 3.4.0、transformers 5.15.0、flash-attn 2.8.3.post1。",
        "- Qwen3-0.6B BF16 负载使用清理后的 ShareGPT Vicuna user prompt，500 条、Poisson 合成到达；语料 revision 为 `192ab2185289094fc556ec8ce5ce1e8e587154ca`。只保存脱敏 user prompt 文本，assistant 回复 token 长度仅用于派生截断的 `max_tokens` 预算，不将参考回复作为模型输入。MoE 负载重新以 DeepSeek-V2-Lite tokenizer 编码。到达回放不是生产 HTTP 日志。",
        "",
        "## 调度",
        "",
        f"在 8 req/s 下完成 5 个相同 prompt trace 的配对 seed。全开策略的输出吞吐平均变化为 {ci_text('output_tok_s')}；联合 SLO 输出 goodput 平均变化为 {ci_text('joint_slo_output_tokens_per_s')}。输出吞吐在 {output_losses}/{len(scheduled_by_seed)} 个 seed 下降，联合 goodput 在 {goodput_losses}/{len(scheduled_by_seed)} 个 seed 下降。当前证据不支持把这一组 all-on 配置作为吞吐或 SLO 优化。",
        chart("scheduling_slo", "五组调度配对结果"),
        "",
    ]
    all_on = one_factor.get("all_on", {})
    without_tpot = one_factor.get("without_tpot_aware", {})
    if all_on and without_tpot:
        lines.extend((
            f"seed0 单因素筛查中，关闭 TPOT-aware 后输出吞吐从 {fmt(all_on.get('output_tok_s'), 1)} 升至 {fmt(without_tpot.get('output_tok_s'), 1)} token/s，联合 SLO goodput 从 {fmt(all_on.get('joint_slo_output_tok_s'), 1)} 升至 {fmt(without_tpot.get('joint_slo_output_tok_s'), 1)} token/s；这是单次筛查，仍低于同 seed 的独立 baseline 配对结果，不能据此定默认策略。",
            "",
        ))
    lines.extend((
        "## 准入",
        "",
        f"共整理 {len(full_admission)} 行 500 请求完整 trace 数据。24 req/s 下，admission-off 与当前 cache-aware 策略如下：",
        "20%/50% 共享前缀变体使用合成的 256-token 公共模板，不代表真实 system prompt。",
        "",
        "| 共享前缀请求比例 | admission-off 接收 | admission-off 联合 goodput (token/s) | cache-aware 接收 / 拒绝率 | cache-aware 联合 goodput (token/s) |",
        "|---:|---:|---:|---:|---:|",
        *admission_lines,
        "",
        "无共享前缀负载的高到达率筛查（接收率 / 联合 SLO 输出 goodput）：",
        "",
        "| 到达率 (req/s) | admission-off | cache-aware |",
        "|---:|---:|---:|",
        *high_rate_lines,
        "",
        f"第一阶段包含 {len(admission_grid_candidates)} 个 cache-aware target/work 组合；使用 {admission_grid_requests} 条 calibration 请求、{fmt(admission_grid_rate, 0)} req/s、TTFT≤{fmt(admission_grid_ttft_slo, 0)} ms、TPOT≤{fmt(admission_grid_tpot_slo, 0)} ms（单 seed）。",
        "| target TTFT (ms) | work budget (ms) | cache-aware 拒绝率 | cache-aware 联合 goodput | cache-blind 联合 goodput | admission-off 联合 goodput | cache-aware TTFT p99 (ms) | cache-aware TPOT p99 (ms) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *admission_grid_lines,
        (f"seed 0 网格中联合 goodput 最高的探索点为 target={fmt(best_admission_grid.get('target_ttft_ms'), 0)} ms、work={fmt(best_admission_grid.get('work_budget_ms'), 0)} ms，拒绝率 {fmt(best_admission_grid.get('reject_percent'), 1)}%、goodput {fmt(best_admission_grid.get('joint_slo_output_tok_s'), 0)} token/s。" if best_admission_grid else "目前没有完整的 admission 参数网格结果。"),
        "100 条 smoke 子集在 96 req/s 下三种变体都接收了全部请求，因此没有用于参数选择。",
        ("第二阶段围绕首轮的高-goodput组和低拒绝率组，使用 TPOT≤80 ms 对 soft pressure / max wait 做 24 点筛查。该阶段与 TPOT≤20 ms 的 target/work 网格属于不同 SLO，不能合并成同一条 Pareto 曲线。" if admission_refine_lines else ""),
        *( ["| target | work budget | soft pressure | max wait (ms) | 拒绝率 | 联合 goodput (token/s) | TTFT p99 (ms) | TPOT p99 (ms) |", "|---:|---:|---:|---:|---:|---:|---:|---:|", *admission_refine_lines] if admission_refine_lines else [] ),
        (f"TPOT≤80 ms 的 seed 0 细化最高 goodput 点为 target/work/soft/wait={fmt(best_admission_refinement.get('target_ttft_ms'), 0)}/{fmt(best_admission_refinement.get('work_budget_ms'), 0)}/{fmt(best_admission_refinement.get('soft_pressure'), 2)}/{fmt(best_admission_refinement.get('max_wait_ms'), 0)}，拒绝率 {fmt(best_admission_refinement.get('reject_percent'), 1)}%、goodput {fmt(best_admission_refinement.get('joint_slo_output_tok_s'), 0)} token/s；该点同时是本轮 refinement 的最低拒绝率点（{fmt(least_rejected_refinement.get('reject_percent'), 1)}%），仍是单次筛查结果。" if best_admission_refinement and least_rejected_refinement else ""),
        *( ["配对验证按同一 prompt/arrival fingerprint 汇总 seed 间差值；括号内为均值 bootstrap 95% CI：", "", "| 数据集 / 条件 | target/work/soft/wait | seeds (fingerprint 匹配) | admission-off 拒绝率 / goodput | cache-aware 拒绝率 / goodput | Δ goodput (token/s) mean (95% CI) | Δ reject (百分点) mean | cache-aware TTFT p99 median (ms) |", "|---|---|---:|---:|---:|---:|---:|---:|", *admission_validation_lines] if admission_validation_lines else [] ),
        "没有业务拒绝率预算；即使某个网格点提高了联合 goodput，也只作为可选 operating point，不据此推荐新的默认阈值。",
        chart("admission_pareto", "准入拒绝率与联合 SLO goodput"),
        chart("admission_calibration", "TPOT 20 ms target/work 网格的拒绝率与 goodput"),
        chart("admission_soft_wait", "TPOT 80 ms soft-pressure / max-wait 筛查"),
        "",
        "## KV 容量与 FP8 精度",
        "",
        f"BF16 synthetic 矩阵报告 {', '.join(map(str, blocks_bf16)) or '—'} 个 block、{', '.join(map(str, slots_bf16)) or '—'} 个 token slot；按 pool 容量 / context length 折算的完整上下文请求数为 {bf16_capacity_text}。FP8 E4M3 矩阵报告 {', '.join(map(str, blocks_fp8)) or '—'} 个 block、{', '.join(map(str, slots_fp8)) or '—'} 个 slot；同口径折算数为 {fp8_capacity_text}。这不是在线并发或 SLO 保证。真实 ShareGPT prompt 矩阵有 {len(kv_real)} 行，pool 为 {', '.join(map(str, blocks_real)) or '—'} 个 block / {', '.join(map(str, slots_real)) or '—'} 个 slot。",
        f"4096-token、48/64 并发的原始 swap-on 点累计 swap={swap_count_high}，未触发 offload。另在 4096 输入、32 并发、512 输出的压力诊断中，关闭重算成本比较后完成 3 次重复；swap-on 的吞吐变化相对 no-swap 为 {fmt(kv_swap_force_delta, 2)}%。这只验证了 swap 路径，不表示该路径提升服务性能。{kv_default_probe_text}",
        "| 抢占设置 | 输出 tok/s | TTFT p99 (ms) | TPOT p99 (ms) | 抢占 | swap | 重算 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *kv_swap_diagnostic_lines,
        chart("kv_capacity", "BF16 与 FP8 KV 并发曲线"),
        "",
        chart("kv_real_prompts", "真实 ShareGPT prompt KV 曲线"),
        "",
        f"FP8 校准使用 {fp8_validation_summary.get('calibration_prompt_count', '—')} 条 prompt / {fp8_validation_summary.get('calibration_token_count', '—')} tokens；held-out 检查使用 {fp8_validation_summary.get('evaluation_prompt_count', '—')} 条 / {fp8_validation_summary.get('evaluation_token_count', '—')} tokens。下表中的 logits 指标基于首个 decode 位置；样本量较小，不代表长文本生成质量。",
        "",
        "| scale margin | 首 token 一致率 | logits top-1 一致率 | top-5 overlap | KL(ref‖FP8) | 最大范围利用率 | 超 E4M3 limit 组件数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *fp8_lines,
        "",
        "## INT4 MoE",
        "",
        f"1/4/16/64/128 行微基准中，fused segment 相对 expert loop 的 median speedup 为 {min(micro_speedups):.2f}× 到 {max(micro_speedups):.2f}×。固定输入正确性覆盖 {micro_correctness.get('all_layer_count', 0)} 层、{micro_correctness.get('all_layer_checks', 0)} 个 layer×seed×batch case；split 精确匹配 {split_checks.get('exact_matches', 0)}/{split_checks.get('checks', 0)}，fused 精确匹配 {fused_checks.get('exact_matches', 0)}/{fused_checks.get('checks', 0)}。微基准收益不等同于 serving 收益。",
        chart("moe_micro_speedup", "INT4 MoE 微基准 speedup"),
        "",
        f"ShareGPT prompt 在线 sweep 的初始筛查有 {len(online_rows)} 个配对条件：fused 输出吞吐 median delta {statistics.median(online_deltas):+.2f}%（范围 {min(online_deltas):+.2f}% 至 {max(online_deltas):+.2f}%；{online_wins} 胜 / {online_losses} 负）。所有请求输出完全匹配的配对为 {len(online_parity)}/{len(online_rows)}；联合 SLO goodput 百分比的可计算配对 median 为 {statistics.median(online_goodput_percent):+.2f}%。",
        (f"针对 8 req/s、16 请求、128 输出、100 ms TPOT 条件补做 {online_repeat_summary.get('paired_measurements', 0)} 组配对重复（{online_repeat_summary.get('seed_count', 0)} seeds × {online_repeat_summary.get('repetitions_per_condition', 0)} repeats）：吞吐中位差 {fmt(_nested(online_repeat_summary, 'output_throughput_delta_percent', 'median'), 2)}%（范围 {fmt(_nested(online_repeat_summary, 'output_throughput_delta_percent', 'min'), 2)}% 至 {fmt(_nested(online_repeat_summary, 'output_throughput_delta_percent', 'max'), 2)}%；{_nested(online_repeat_summary, 'output_throughput_delta_percent', 'wins', default=0)} 胜 / {_nested(online_repeat_summary, 'output_throughput_delta_percent', 'losses', default=0)} 负），均值 95% bootstrap CI {fmt(_nested(online_repeat_summary, 'output_throughput_delta_percent', 'ci95_low'), 2)}% 至 {fmt(_nested(online_repeat_summary, 'output_throughput_delta_percent', 'ci95_high'), 2)}%；联合 SLO goodput 差值中位 {fmt(_nested(online_repeat_summary, 'joint_slo_output_delta_tok_s', 'median'), 2)} token/s；输出 token 完全匹配 {online_repeat_summary.get('output_token_parity_pairs', 0)}/{online_repeat_summary.get('output_token_parity_total', 0)}。单一在线条件仍不证明稳定融合收益，内核正确性以固定输入 layer 检查为准。" if online_repeat_summary else ""),
        chart("moe_online_deltas", "INT4 MoE 在线配对差值"),
        "",
        "MLA cache assembly 同时修正了 query range 计算：每行结束位置使用 `start + qlen`，并检查 query start/length 数量与 row 数一致。修正后 seed42 的 16 请求复现记录仍有 1 个请求输出不一致（request 5，0-based token 14；15/16 请求匹配），因此根因尚未完整关闭；不要把该修正描述为已解决所有在线分歧。",
        "",
        "## 投机解码",
        "",
        f"N-gram 共 {len(spec_ngram)} 条运行记录。batch=8、γ=4 的内容分层中位数：",
        "",
        "| 类别 | 运行数 | α median | 吞吐 speedup median |",
        "|---|---:|---:|---:|",
        *spec_content_lines,
        "",
        "free / repeat 的 batch×γ 扩展：",
        "",
        "| 类别 | 运行数 | α median | 吞吐 speedup median |",
        "|---|---:|---:|---:|",
        *spec_sweep_lines,
        "",
        f"混合这些运行的 α 与输出 4-gram 重复率 Spearman ρ={fmt(spec_correlation.get('alpha_vs_fourgram_repeat_rate_spearman'), 3)}，与 distinct-4 ρ={fmt(spec_correlation.get('alpha_vs_distinct4_spearman'), 3)}。相关性跨多个类别和 batch，不是因果或总体服务估计。repeat 对照 α 接近 1，真实对话及结构化内容更低且波动；没有可用的 Medusa/EAGLE draft 权重。",
        chart("spec_content", "按内容类型分层的投机结果"),
        "",
        chart("spec_batch_gamma", "batch 与 gamma 对接受率和 speedup 的影响"),
        "",
        chart("spec_acceptance_repeat", "投机接受率与输出重复率"),
        "",
        "## 未覆盖项与解释边界",
        "",
        "- 准入：已做 TPOT 20 ms 的 12 点 target/work 网格、TPOT 80 ms 的 24 点 soft/wait 网格，以及 20/40/80/120 ms 下的候选 seed 对照和 100 条 held-out prompt。target/work 与 refinement 使用不同 TPOT SLO；held-out 使用 384 req/s 短 trace，与 400 请求 calibration 负载不等价。没有产品拒绝率预算，仍不推荐上线阈值。",
        "- KV：默认成本策略压力点以重算处理了抢占；关闭重算成本比较后单独测了 swap on/off 三次，报告是 swap 路径诊断而非性能收益。未测到 OOM/物理容量极限；FP8 held-out 只比较少量 prompt 的首个 decode logits。",
        "- INT4 MoE 在线初筛每条件一次，16 请求、8/16 req/s、16/64/128 输出上限、100/150 ms TPOT、3 seeds；另对 8 req/s、128 输出、100 ms TPOT 做 5 seeds × 5 repeats。仍不足以确认所有负载上的稳定 goodput 提升。",
        "- 投机解码使用固定 96-token 输出、temperature 0.6、3 seeds 和 n-gram；没有自然停止、Medusa/EAGLE 权重或大规模 held-out workload。",
        "- 性能实验运行于带未提交代码的工作区；以上数据只对应所列 GPU、模型、负载和配置。",
        "",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    # Keep dated follow-ups after regenerating the 2026-10-15 aggregate.
    for addendum in sorted(
            Path(__file__).parent.glob("_supplemental_report_20*.md")):
        addition = addendum.read_text(encoding="utf-8").strip()
        if addition:
            with path.open("a", encoding="utf-8", newline="\n") as report:
                report.write(f"\n\n{addition}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/supplemental")
    parser.add_argument("--output-dir", default="results/supplemental/report")
    parser.add_argument("--tracked-report",
                        default="benchmarks/_supplemental_report.md")
    parser.add_argument("--figure-dir",
                        default="benchmarks/_supplemental_figures")
    args = parser.parse_args()
    data_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    tracked_report = Path(args.tracked_report)
    figure_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    scheduling_rows, scheduling_pairs, scheduling_ci = _collect_scheduling(data_dir)
    scheduling_one_factor_rows = _collect_scheduling_one_factor(data_dir)
    admission_rows = _collect_admission(data_dir)
    admission_calibration_rows = _collect_admission_calibration(data_dir)
    admission_validation_pairs = _pair_admission_validation(
        admission_calibration_rows)
    kv_rows = _collect_kv(data_dir)
    fp8_validation_rows, fp8_validation_summary = _collect_fp8_validation(data_dir)
    micro_rows, micro_correctness = _collect_micro(data_dir)
    online_rows, online_summary = _collect_online_moe(data_dir)
    online_repeat_rows, online_repeat_summary = _collect_online_moe_repeats(
        data_dir)
    spec_rows, spec_correlation = _collect_spec(data_dir)

    _write_csv(output_dir / "scheduling_summary.csv", scheduling_rows)
    _write_csv(output_dir / "scheduling_paired.csv", scheduling_pairs)
    _write_csv(output_dir / "scheduling_one_factor.csv",
               scheduling_one_factor_rows)
    _write_csv(output_dir / "admission_summary.csv", admission_rows)
    _write_csv(output_dir / "admission_calibration.csv",
               admission_calibration_rows)
    _write_csv(output_dir / "admission_validation_pairs.csv",
               admission_validation_pairs)
    _write_csv(output_dir / "kv_capacity.csv", kv_rows)
    _write_csv(output_dir / "kv_fp8_validation.csv", fp8_validation_rows)
    _write_csv(output_dir / "moe_micro.csv", micro_rows)
    _write_csv(output_dir / "moe_online_pairs.csv", online_rows)
    _write_csv(output_dir / "moe_online_repeat_pairs.csv", online_repeat_rows)
    _write_csv(output_dir / "spec_runs.csv", spec_rows)
    _write_json(output_dir / "scheduling_bootstrap.json", scheduling_ci)
    _write_json(output_dir / "kv_fp8_validation_summary.json",
                fp8_validation_summary)
    _write_json(output_dir / "moe_micro_correctness.json", micro_correctness)
    _write_json(output_dir / "moe_online_summary.json", online_summary)
    _write_json(output_dir / "moe_online_repeat_summary.json",
                online_repeat_summary)
    _write_json(output_dir / "spec_content_correlations.json", spec_correlation)

    try:
        import matplotlib.pyplot as pyplot
    except ImportError:
        chart_ext = "svg"
        _write_svg_charts(
            figure_dir, scheduling_rows=scheduling_rows,
            admission_rows=admission_rows,
            admission_calibration_rows=admission_calibration_rows,
            kv_rows=kv_rows,
            micro_rows=micro_rows, online_rows=online_rows,
            spec_rows=spec_rows, spec_correlation=spec_correlation)
    else:
        chart_ext = "png"
        _figure(figure_dir / "scheduling_slo.png",
                lambda plt: _plot_scheduling(plt, scheduling_rows,
                                             scheduling_pairs, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "admission_pareto.png",
                lambda plt: _plot_admission(plt, admission_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "admission_calibration.png",
                lambda plt: _plot_admission_calibration(
                    plt, admission_calibration_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "admission_soft_wait.png",
                lambda plt: _plot_admission_refinement(
                    plt, admission_calibration_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "kv_capacity.png",
                lambda plt: _plot_kv(plt, kv_rows, figure_dir), pyplot=pyplot)
        _figure(figure_dir / "kv_real_prompts.png",
                lambda plt: _plot_real_kv(plt, kv_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "moe_micro_speedup.png",
                lambda plt: _plot_micro(plt, micro_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "moe_online_deltas.png",
                lambda plt: _plot_online_moe(plt, online_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "spec_content.png",
                lambda plt: _plot_spec_content(plt, spec_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "spec_batch_gamma.png",
                lambda plt: _plot_spec_batch_gamma(plt, spec_rows, figure_dir),
                pyplot=pyplot)
        _figure(figure_dir / "spec_acceptance_repeat.png",
                lambda plt: _plot_spec_repeat(plt, spec_rows, figure_dir,
                                              spec_correlation),
                pyplot=pyplot)

    report_args = {
        "data_dir": data_dir,
        "scheduling_rows": scheduling_rows,
        "scheduling_ci": scheduling_ci,
        "scheduling_one_factor_rows": scheduling_one_factor_rows,
        "admission_rows": admission_rows,
        "admission_calibration_rows": admission_calibration_rows,
        "admission_validation_pairs": admission_validation_pairs,
        "kv_rows": kv_rows,
        "fp8_validation_rows": fp8_validation_rows,
        "fp8_validation_summary": fp8_validation_summary,
        "micro_rows": micro_rows,
        "micro_correctness": micro_correctness,
        "online_rows": online_rows,
        "online_summary": online_summary,
        "online_repeat_rows": online_repeat_rows,
        "online_repeat_summary": online_repeat_summary,
        "spec_rows": spec_rows,
        "spec_correlation": spec_correlation,
        "chart_ext": chart_ext,
    }
    output_chart_prefix = Path(os.path.relpath(
        figure_dir, output_dir)).as_posix().rstrip("/") + "/"
    tracked_chart_prefix = Path(os.path.relpath(
        figure_dir, tracked_report.parent)).as_posix().rstrip("/") + "/"
    _write_markdown(output_dir / "README.md",
                    chart_prefix=output_chart_prefix, **report_args)
    _write_markdown(tracked_report,
                    chart_prefix=tracked_chart_prefix, **report_args)
    print(f"Supplemental report written to {output_dir}")
    print(f"Tracked report written to {tracked_report}")
    print(f"Scheduling pairs: {len(scheduling_pairs)}; admission rows: "
          f"{len(admission_rows)}; KV rows: {len(kv_rows)}; "
          f"FP8 rows: {len(fp8_validation_rows)}; "
          f"MoE online pairs: {len(online_rows)}; spec runs: {len(spec_rows)}; "
          f"charts: {chart_ext}")


if __name__ == "__main__":
    main()
