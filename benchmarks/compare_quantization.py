"""Compare quantization runs on an identical continuous-arrival trace.

Example::

    python benchmarks/compare_quantization.py results/int4.json results/nvfp4.json \
        --variant all_on

The workload fingerprint and all execution settings except the quantization mode
and NVFP4 GEMM backend must match. The report emphasizes joint TTFT/TPOT SLO
goodput as well as latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _run_summary(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    runs = payload.get("runs")
    if not isinstance(runs, dict) or variant not in runs:
        available = ", ".join(runs) if isinstance(runs, dict) else "none"
        raise ValueError(f"variant {variant!r} is not present (available: {available})")
    summary = runs[variant].get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("slo_goodput"), dict):
        raise ValueError(f"variant {variant!r} does not contain SLO goodput metrics")
    return summary


def _metric_values(payload: dict[str, Any], variant: str) -> dict[str, float | int | None]:
    summary = _run_summary(payload, variant)
    goodput = summary["slo_goodput"]

    def millis(name: str, percentile: str) -> float | None:
        value = summary.get(name, {}).get(percentile)
        return None if value is None else value * 1000.0

    return {
        "joint_slo_output_tok_s": goodput["joint_slo_output_tokens_per_second"],
        "joint_slo_requests_s": goodput["joint_slo_request_rate_per_second"],
        "joint_slo_arrival_percent": goodput["joint_slo_arrival_percent"],
        "all_output_tok_s": summary["throughput_output_tok_per_s"],
        "ttft_p50_ms": millis("ttft", "p50"),
        "ttft_p99_ms": millis("ttft", "p99"),
        "tpot_p50_ms": millis("tpot", "p50"),
        "tpot_p99_ms": millis("tpot", "p99"),
        "e2e_p99_ms": millis("e2e", "p99"),
    }


def compare_payloads(baseline: dict[str, Any], candidate: dict[str, Any],
                     variant: str = "all_on") -> dict[str, Any]:
    """Compare two quantization reports after validating workload identity."""
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
            raise ValueError(f"reports differ in meta.{key}")
    if baseline.get("variants") != candidate.get("variants"):
        raise ValueError("reports use different scheduler policy variants")

    before_config = dict(baseline.get("benchmark_config", {}))
    after_config = dict(candidate.get("benchmark_config", {}))
    before_quantization = before_config.pop("quantization", None)
    after_quantization = after_config.pop("quantization", None)
    before_backend = before_config.pop("nvfp4_backend", "software")
    after_backend = after_config.pop("nvfp4_backend", "software")
    if before_config != after_config:
        raise ValueError("reports use different execution or scheduler settings")

    before = _metric_values(baseline, variant)
    after = _metric_values(candidate, variant)
    return {
        "variant": variant,
        "fingerprint": before_fingerprint,
        "seed": before_workload.get("seed"),
        "baseline_quantization": before_quantization,
        "candidate_quantization": after_quantization,
        "baseline_nvfp4_backend": before_backend,
        "candidate_nvfp4_backend": after_backend,
        "before": before,
        "after": after,
        "deltas": {
            key: (None if before[key] is None or after[key] is None
                  else after[key] - before[key])
            for key in before
        },
    }


def _format(value: float | int | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="baseline quantization benchmark JSON")
    parser.add_argument("candidate", type=Path, help="candidate quantization benchmark JSON")
    parser.add_argument("--variant", default="all_on", help="arrival policy variant to compare")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional JSON file for the paired comparison")
    args = parser.parse_args()

    with args.baseline.open(encoding="utf-8") as f:
        baseline = json.load(f)
    with args.candidate.open(encoding="utf-8") as f:
        candidate = json.load(f)
    result = compare_payloads(baseline, candidate, args.variant)
    before = result["before"]
    after = result["after"]
    delta = result["deltas"]
    print(f"paired quantization run: {result['baseline_quantization']} "
          f"({result['baseline_nvfp4_backend']}) -> "
          f"{result['candidate_quantization']} ({result['candidate_nvfp4_backend']}) | "
          f"variant={args.variant}")
    print("metric                       baseline     candidate       delta")
    for key, unit in (
        ("joint_slo_output_tok_s", " tok/s"),
        ("joint_slo_requests_s", " req/s"),
        ("joint_slo_arrival_percent", " pp"),
        ("all_output_tok_s", " tok/s"),
        ("ttft_p50_ms", " ms"),
        ("ttft_p99_ms", " ms"),
        ("tpot_p50_ms", " ms"),
        ("tpot_p99_ms", " ms"),
        ("e2e_p99_ms", " ms"),
    ):
        print(f"{key:<28} {_format(before[key]):>11} {_format(after[key]):>13} "
              f"{_format(delta[key], unit):>13}")
    print("Higher joint SLO output tok/s is better; lower TTFT/TPOT/E2E latency is better.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"comparison JSON -> {args.output}")


if __name__ == "__main__":
    main()
