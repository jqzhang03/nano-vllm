import pytest

from benchmarks.compare_quantization import compare_payloads


def _report(mode="int4", fingerprint="same-trace", goodput=10.0,
            max_model_len=512, backend="software"):
    return {
        "meta": {"model": "model", "gpu": "gpu", "cuda": "12.8", "torch": "2.8"},
        "benchmark_config": {
            "quantization": mode,
            "nvfp4_backend": backend,
            "max_model_len": max_model_len,
            "kv_cache_dtype": "auto",
            "enforce_eager": True,
        },
        "workload": {"fingerprint": fingerprint, "seed": 7},
        "runs": {
            "all_on": {
                "summary": {
                    "throughput_output_tok_per_s": 20.0,
                    "slo_goodput": {
                        "joint_slo_output_tokens_per_second": goodput,
                        "joint_slo_request_rate_per_second": 1.0,
                        "joint_slo_arrival_percent": 25.0,
                    },
                    "ttft": {"p50": 0.1, "p99": 0.4},
                    "tpot": {"p50": 0.02, "p99": 0.05},
                    "e2e": {"p99": 0.8},
                },
            },
        },
    }


def test_compare_quantization_reports_paired_goodput_and_latency():
    result = compare_payloads(_report(), _report("nvfp4", goodput=12.0))

    assert result["baseline_quantization"] == "int4"
    assert result["candidate_quantization"] == "nvfp4"
    assert result["baseline_nvfp4_backend"] == "software"
    assert result["candidate_nvfp4_backend"] == "software"
    assert result["deltas"]["joint_slo_output_tok_s"] == 2.0
    assert result["after"]["ttft_p99_ms"] == 400.0


def test_compare_quantization_allows_backend_change_for_paired_nvfp4_runs():
    baseline = _report("nvfp4", backend="software")
    candidate = _report("nvfp4", backend="cublaslt", goodput=12.0)

    result = compare_payloads(baseline, candidate)

    assert result["baseline_quantization"] == result["candidate_quantization"] == "nvfp4"
    assert result["baseline_nvfp4_backend"] == "software"
    assert result["candidate_nvfp4_backend"] == "cublaslt"
    assert result["deltas"]["joint_slo_output_tok_s"] == 2.0


def test_compare_quantization_rejects_different_workloads():
    with pytest.raises(ValueError, match="different prompts"):
        compare_payloads(_report(), _report(fingerprint="different-trace"))


def test_compare_quantization_rejects_other_setting_changes():
    with pytest.raises(ValueError, match="different execution"):
        compare_payloads(_report(), _report("nvfp4", max_model_len=1024))
