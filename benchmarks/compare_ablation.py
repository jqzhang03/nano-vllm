"""Compare two arrival benchmark JSON files on the same exact workload trace.

Examples::

    python benchmarks/compare_ablation.py results/before.json results/after.json \
        --variant all_on
    python benchmarks/compare_ablation.py results/admission_before.json \
        results/admission_after.json --variant dynamic_on_cache_aware

Both reports must include the workload fingerprint emitted by the current
arrival benchmark scripts. Higher SLO goodput is better; lower latency is better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _kind_and_run(payload: dict[str, Any], variant: str) -> tuple[str, dict[str, Any]]:
    runs = payload.get("runs")
    if not isinstance(runs, dict) or variant not in runs:
        available = ", ".join(runs) if isinstance(runs, dict) else "none"
        raise ValueError(f"variant {variant!r} is not present (available: {available})")
    run = runs[variant]
    if isinstance(run.get("summary"), dict):
        return "scheduling", run["summary"]
    if isinstance(run.get("slo_goodput"), dict):
        return "admission", run
    raise ValueError(f"run {variant!r} does not contain SLO goodput metrics")


def _metric_values(payload: dict[str, Any], variant: str) -> dict[str, float | int | None]:
    kind, summary = _kind_and_run(payload, variant)
    goodput = summary["slo_goodput"]
    if kind == "scheduling":
        total_tok_s = summary["throughput_output_tok_per_s"]
        ttft = summary["ttft"]
        tpot = summary["tpot"]
        latency_scale = 1000.0
    else:
        total_tok_s = summary["output_tokens_per_second"]
        ttft = summary["ttft_ms"]
        tpot = summary["tpot_ms"]
        latency_scale = 1.0
    acceptance = summary.get("acceptance", {})
    return {
        "joint_slo_output_tok_s": goodput["joint_slo_output_tokens_per_second"],
        "joint_slo_requests_s": goodput["joint_slo_request_rate_per_second"],
        "joint_slo_arrival_percent": goodput["joint_slo_arrival_percent"],
        "all_output_tok_s": total_tok_s,
        "ttft_p50_ms": (None if ttft.get("p50") is None
                        else ttft["p50"] * latency_scale),
        "ttft_p99_ms": (None if ttft.get("p99") is None
                        else ttft["p99"] * latency_scale),
        "tpot_p50_ms": (None if tpot.get("p50") is None
                        else tpot["p50"] * latency_scale),
        "tpot_p99_ms": (None if tpot.get("p99") is None
                        else tpot["p99"] * latency_scale),
        "accept_rate_percent": acceptance.get("accept_rate_percent"),
        "rejected_requests": summary.get("rejected"),
    }


def compare_payloads(baseline: dict[str, Any], candidate: dict[str, Any],
                     variant: str) -> dict[str, Any]:
    """Validate workload identity and return comparable before/after metrics."""
    before_workload = baseline.get("workload", {})
    after_workload = candidate.get("workload", {})
    before_fingerprint = before_workload.get("fingerprint")
    after_fingerprint = after_workload.get("fingerprint")
    if not before_fingerprint or not after_fingerprint:
        raise ValueError("both reports must contain workload.fingerprint")
    if before_fingerprint != after_fingerprint:
        raise ValueError("reports use different prompts, arrival times, seeds, or SLO targets")

    before_meta = baseline.get("meta", {})
    after_meta = candidate.get("meta", {})
    for key in ("model", "gpu", "cuda", "torch"):
        if before_meta.get(key) != after_meta.get(key):
            raise ValueError(
                f"reports differ in meta.{key}: {before_meta.get(key)!r} vs "
                f"{after_meta.get(key)!r}")

    def execution_settings(payload: dict[str, Any]) -> dict[str, Any]:
        config = payload.get("benchmark_config", payload.get("config", {}))
        return {
            "tensor_parallel_size": config.get(
                "tensor_parallel_size", config.get("tp")),
            "max_model_len": config.get("max_model_len"),
            "gpu_memory_utilization": config.get("gpu_memory_utilization"),
            "enforce_eager": config.get("enforce_eager"),
            "kv_cache_dtype": config.get("kv_cache_dtype"),
            "quantization": config.get("quantization"),
            "kv_swap": config.get("kv_swap", not config.get("no_swap_kv", False)),
            "mixed_cudagraph": config.get(
                "mixed_cudagraph", not config.get("no_mixed_cudagraph", False)),
        }

    before_settings = execution_settings(baseline)
    after_settings = execution_settings(candidate)
    invariant_settings = ("tensor_parallel_size", "max_model_len",
                          "gpu_memory_utilization", "kv_cache_dtype", "quantization")
    if any(before_settings[key] != after_settings[key] for key in invariant_settings):
        differing = [key for key in invariant_settings
                     if before_settings[key] != after_settings[key]]
        raise ValueError("reports use different execution settings: "
                         + ", ".join(differing))

    before = _metric_values(baseline, variant)
    after = _metric_values(candidate, variant)
    return {
        "variant": variant,
        "fingerprint": before_fingerprint,
        "seed": before_workload.get("seed"),
        "before": before,
        "after": after,
        "before_execution_settings": before_settings,
        "after_execution_settings": after_settings,
        "deltas": {
            key: (None if before[key] is None or after[key] is None
                  else after[key] - before[key])
            for key in before
        },
    }


def _default_variant(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    before = set(baseline.get("runs", {}))
    after = set(candidate.get("runs", {}))
    common = before & after
    for preferred in ("all_on", "dynamic_on_cache_aware", "baseline"):
        if preferred in common:
            return preferred
    if not common:
        raise ValueError("reports have no benchmark variant in common")
    return sorted(common)[0]


def _format(value: float | int | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="before-change benchmark JSON")
    parser.add_argument("candidate", type=Path, help="after-change benchmark JSON")
    parser.add_argument("--variant", default=None,
                        help="variant to compare (auto-selects a common default)")
    args = parser.parse_args()
    with args.baseline.open(encoding="utf-8") as result_file:
        baseline = json.load(result_file)
    with args.candidate.open(encoding="utf-8") as result_file:
        candidate = json.load(result_file)
    variant = args.variant or _default_variant(baseline, candidate)
    result = compare_payloads(baseline, candidate, variant)
    setting_changes = {
        key: (result["before_execution_settings"][key],
              result["after_execution_settings"][key])
        for key in result["before_execution_settings"]
        if result["before_execution_settings"][key]
        != result["after_execution_settings"][key]
    }

    print(f"paired {variant!r}, seed={result['seed']}, "
          f"workload={result['fingerprint'][:16]}")
    if setting_changes:
        print(f"execution settings changed: {setting_changes}")
    print("metric                         before       after       delta")
    for key, before in result["before"].items():
        after = result["after"][key]
        delta = result["deltas"][key]
        unit = {
            "joint_slo_output_tok_s": " tok/s",
            "joint_slo_requests_s": " req/s",
            "joint_slo_arrival_percent": "%",
            "all_output_tok_s": " tok/s",
            "accept_rate_percent": "%",
            "rejected_requests": " req",
        }.get(key, " ms" if key.endswith("_ms") else "")
        delta_text = _format(delta, unit)
        print(f"{key:<29} {_format(before, unit):>11} "
              f"{_format(after, unit):>11} {delta_text:>11}")
    goodput_delta = result["deltas"]["joint_slo_output_tok_s"]
    if goodput_delta is None:
        assessment = "SLO output goodput is unavailable in one report"
    elif goodput_delta > 0:
        assessment = "SLO output goodput improved"
    elif goodput_delta < 0:
        assessment = "SLO output goodput regressed"
    else:
        assessment = "SLO output goodput is unchanged"
    print(f"\nAssessment: {assessment}.")
    print("Positive latency deltas are regressions; goodput deltas use tokens/s.")


if __name__ == "__main__":
    main()
