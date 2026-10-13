from types import SimpleNamespace

import pytest

from benchmarks.compare_ablation import compare_payloads
from benchmarks.scheduling_ablation import _step_stats_empty, summarize_requests


def _report(fingerprint="same-trace", output_goodput=10.0):
    return {
        "meta": {"model": "model", "gpu": "gpu", "cuda": "12.8", "torch": "2.8"},
        "workload": {"fingerprint": fingerprint, "seed": 7},
        "runs": {
            "all_on": {
                "summary": {
                    "throughput_output_tok_per_s": 20.0,
                    "slo_goodput": {
                        "joint_slo_output_tokens_per_second": output_goodput,
                        "joint_slo_request_rate_per_second": 1.0,
                        "joint_slo_arrival_percent": 25.0,
                    },
                    "ttft": {"p50": 0.1, "p99": 0.4},
                    "tpot": {"p50": 0.02, "p99": 0.05},
                },
            },
        },
    }


def test_compare_payloads_reports_paired_goodput_and_latency():
    result = compare_payloads(_report(), _report(output_goodput=12.0), "all_on")

    assert result["seed"] == 7
    assert result["deltas"]["joint_slo_output_tok_s"] == 2.0
    assert result["after"]["ttft_p99_ms"] == 400.0


def test_compare_payloads_rejects_different_workloads():
    with pytest.raises(ValueError, match="different prompts"):
        compare_payloads(_report(), _report(fingerprint="other-trace"), "all_on")


def test_scheduling_goodput_reports_per_request_ttft_target_range():
    rows = [
        {
            "t_submitted": 0.0,
            "t_prefill_started": 0.0,
            "t_first_token": 0.1,
            "t_completed": 0.2,
            "completion_tokens": 3,
            "ttft_slo_ms": 50.0,
            "ttft_slo_met": False,
            "tpot_slo_ms": 80.0,
            "tpot_slo_met": True,
        },
        {
            "t_submitted": 0.0,
            "t_prefill_started": 0.0,
            "t_first_token": 0.1,
            "t_completed": 0.2,
            "completion_tokens": 3,
            "ttft_slo_ms": 150.0,
            "ttft_slo_met": True,
            "tpot_slo_ms": 80.0,
            "tpot_slo_met": True,
        },
    ]
    metrics = {"per_request": rows, "step_stats": _step_stats_empty()}

    summary = summarize_requests(
        metrics, wall=1.0,
        args=SimpleNamespace(slo_ttft_ms=1000.0, slo_tpot_ms=80.0))

    assert summary["slo_goodput"]["ttft_target_ms_min"] == 50.0
    assert summary["slo_goodput"]["ttft_target_ms_max"] == 150.0
    assert summary["slo_goodput"]["joint_slo_requests"] == 1
    assert summary["request_ttft_slo_percent"] == 50.0
