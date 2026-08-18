"""Run a comparison workload on the real vLLM, emitting a JSON result file.

Must run in the isolated `vllm-compare` conda env (see run_vllm_compare.sh),
which pins torch 2.8.0+cu128 and our flash-attn 2.8.3.post1 wheel so both
engines share the same attention backend build on sm_120.

Emits the SAME JSON schema as compare_nanovllm.py; compare_merge.py combines
them. Latency definitions (TTFT/TPOT/E2E) match compare_common.py exactly.

Usage:
    python benchmarks/compare_vllm.py --workload results/compare_workload_clean.json \
        --kv-cache-dtype auto --output results/compare_vllm_clean_fp16.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_common import compute_summary, slo_attainment  # noqa: E402

from vllm import LLM, SamplingParams as VSP  # noqa: E402
from vllm.v1.metrics.reader import Histogram as VHistogram  # noqa: E402

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

# ---------------------------------------------------------------------------
# V1 engine: RequestOutput.metrics is always None (see vllm/v1/engine/output_processor.py),
# so per-request timing comes from the aggregated prometheus-style histograms exposed
# by LLM.get_metrics(). Mean = sum/count (exact); p50/p99 = linear interpolation over
# histogram buckets (approximate, marked with ~ in the report).
# ---------------------------------------------------------------------------

HIST_NAMES = {
    "ttft": "vllm:time_to_first_token_seconds",
    "tpot": "vllm:inter_token_latency_seconds",
    "tpot_fallback": "vllm:time_per_output_token_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
}


def snapshot_histograms(metrics) -> dict[str, dict]:
    """Aggregate histograms across label combos into {name: {count, sum, buckets}}.

    Bucket keys are normalized to float (kept as "+Inf" sentinel).
    """
    agg: dict[str, dict] = {}
    for m in metrics:
        if not isinstance(m, VHistogram):
            continue
        a = agg.setdefault(m.name, {"count": 0, "sum": 0.0, "buckets": {}})
        a["count"] += m.count
        a["sum"] += m.sum
        for le, c in m.buckets.items():
            key: float | str = float(le) if le != "+Inf" else "+Inf"
            a["buckets"][key] = a["buckets"].get(key, 0) + c
    return agg


def diff_hists(after: dict[str, dict], before: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for name, a in after.items():
        b = before.get(name, {"count": 0, "sum": 0.0, "buckets": {}})
        buckets = dict(a["buckets"])
        for le, c in b["buckets"].items():
            buckets[le] = buckets.get(le, 0) - c
        out[name] = {"count": a["count"] - b["count"],
                     "sum": a["sum"] - b["sum"], "buckets": buckets}
    return out


def hist_quantile(q: float, buckets: dict) -> float | None:
    """Prometheus-style histogram_quantile.

    NOTE: prometheus bucket counts are CUMULATIVE (le = "<= bound"), so the
    count in a bucket is samples with value <= bound. Convert to per-bucket
    spans and interpolate linearly inside the bucket that contains the rank.
    """
    keys = sorted(k for k in buckets if k != "+Inf")
    total = buckets.get("+Inf", 0)
    if total <= 0:
        total = sum(buckets.get(k, 0) for k in keys)
    if total <= 0:
        return None
    target = q * total
    prev_cum = 0
    prev_le = 0.0
    for le in keys:
        c = buckets[le]  # cumulative count of samples <= le
        if c >= target:
            span = c - prev_cum
            if span <= 0:
                return prev_le
            return prev_le + (target - prev_cum) / span * (le - prev_le)
        prev_cum = c
        prev_le = le
    return prev_le


def summary_from_hists(hists: dict[str, dict]) -> dict[str, dict]:
    def build(name: str) -> dict:
        h = hists.get(name)
        if not h or h["count"] <= 0:
            return {"avg": None, "p50": None, "p99": None, "min": None, "max": None, "count": 0}
        return {
            "avg": h["sum"] / h["count"],
            "p50": hist_quantile(0.50, h["buckets"]),
            "p99": hist_quantile(0.99, h["buckets"]),
            "min": None, "max": None,
            "count": h["count"],
        }

    tpot = build(HIST_NAMES["tpot"])
    if tpot["count"] == 0:
        tpot = build(HIST_NAMES["tpot_fallback"])
    return {"ttft": build(HIST_NAMES["ttft"]), "tpot": tpot, "e2e": build(HIST_NAMES["e2e"])}


def env_info() -> dict:
    import vllm
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    }
    try:
        import flash_attn
        info["flash_attn"] = flash_attn.__version__
    except ImportError:
        info["flash_attn"] = None
    info["engine_version"] = vllm.__version__
    return info


def probe(obj, attr_path: str):
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def kv_info(llm) -> dict:
    """Best-effort KV cache facts from the vLLM engine (path differs by version)."""
    cc = None
    for path in ("engine.cache_config", "llm_engine.cache_config",
                 "engine.model_executor.cache_config", "cache_config"):
        cc = probe(llm, path)
        if cc is not None:
            break
    if cc is None:
        return {}
    info = {}
    for attr in ("num_gpu_blocks", "block_size", "kv_cache_dtype"):
        v = getattr(cc, attr, None)
        if v is not None:
            info[attr] = v
    if "num_gpu_blocks" in info and "block_size" in info:
        info["capacity_tokens"] = info["num_gpu_blocks"] * info["block_size"]
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True)
    p.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e4m3"])
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--max-num-seqs", type=int, default=512)
    p.add_argument("--warmup-seqs", type=int, default=8)
    args = p.parse_args()

    with open(args.workload, encoding="utf-8") as f:
        spec = json.load(f)
    prompts = spec["prompts"]
    max_tokens = spec["max_tokens"]

    engine_args = dict(
        model=args.model,
        dtype="auto",  # load as stored (bf16 for Qwen3), same as nano-vllm
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=False,  # CUDA graphs on, matching nano-vllm default
        enable_prefix_caching=True,
        enable_chunked_prefill=True,  # match nano-vllm's chunked-prefill scheduler
        swap_space=0,  # no CPU offload — matches nano-vllm (and WSL has little RAM)
        disable_log_stats=False,  # per-request timings flow through the stat loggers
    )
    llm = LLM(**engine_args)
    # vLLM 0.10.2: V1 engine for fp16 (no per-request metrics in offline API),
    # falls back to V0 for fp8 kv cache (per-request metrics available).
    try:
        from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
        engine_variant = "V1" if isinstance(llm.llm_engine, V1LLMEngine) else "V0"
    except ImportError:
        engine_variant = "V0"
    # Warmup: profile run + CUDA graph capture happen on the first generate.
    llm.generate(["warm up"] * args.warmup_seqs, VSP(temperature=0.6, max_tokens=8),
                 use_tqdm=False)

    vprompts = [{"prompt_token_ids": p} for p in prompts]
    vsps = [VSP(temperature=0.6, ignore_eos=True, max_tokens=mt) for mt in max_tokens]
    before = snapshot_histograms(llm.get_metrics())
    t0 = time.perf_counter()
    outputs = llm.generate(vprompts, vsps, use_tqdm=False)
    wall = time.perf_counter() - t0
    after = snapshot_histograms(llm.get_metrics())

    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    kvi = kv_info(llm)

    # V0 populates RequestOutput.metrics -> exact per-request summary (same
    # definition as nano-vllm). V1 does not -> histogram-based summary.
    rows = []
    for out in outputs:
        m = getattr(out, "metrics", None)
        arr = getattr(m, "arrival_time", None) if m is not None else None
        fst = getattr(m, "first_token_time", None) if m is not None else None
        fin = getattr(m, "finished_time", None) if m is not None else None
        rows.append({"t_submitted": arr, "t_first_token": fst, "t_completed": fin,
                     "completion_tokens": len(out.outputs[0].token_ids)})

    if rows and rows[0]["t_submitted"] is not None:
        summary = compute_summary(rows)
        per_request = rows
        slo = slo_attainment(rows)
        metrics_note = ("V0 engine: exact per-request timings "
                        "(RequestOutput.metrics)")
    else:
        summary = summary_from_hists(diff_hists(after, before))
        per_request = []
        slo = None
        metrics_note = ("V1 engine: RequestOutput.metrics is always None; "
                        "avg = histogram sum/count (exact), p50/p99 = linear "
                        "interpolation over cumulative histogram buckets (approximate)")

    payload = {
        "engine": "vllm",
        "tag": spec["tag"],
        "kv_cache_dtype": args.kv_cache_dtype,
        "date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "meta": {
            **env_info(),
            "model": args.model,
            "config": {k: v for k, v in engine_args.items() if k != "model"},
            "engine_variant": engine_variant,
            "kv_cache": kvi,
            "metrics_note": metrics_note,
        },
        "workload": {k: spec[k] for k in ("num_seqs", "min_input_len", "max_input_len",
                                          "min_output_len", "max_output_len", "shared_prefix_len")},
        "wall": wall,
        "throughput_tok_per_s": total_out / wall,
        "total_output_tokens": total_out,
        "summary": summary,
        "slo": slo,
        "num_preemptions": None,  # vLLM does not expose a simple preemption counter here
        "decode_steps": None,
        "prefill_steps": None,
        "per_request": per_request,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    s = summary
    p50_ttft = "n/a" if s["ttft"]["p50"] is None else f"{s['ttft']['p50'] * 1000:.1f}ms"
    p99_ttft = "n/a" if s["ttft"]["p99"] is None else f"{s['ttft']['p99'] * 1000:.1f}ms"
    p50_tpot = "n/a" if s["tpot"]["p50"] is None else f"{s['tpot']['p50'] * 1000:.1f}ms"
    p99_tpot = "n/a" if s["tpot"]["p99"] is None else f"{s['tpot']['p99'] * 1000:.1f}ms"
    print(f"[vllm {args.kv_cache_dtype}] wall {wall:.2f}s | "
          f"throughput {total_out / wall:.1f} tok/s | "
          f"TTFT p50 {p50_ttft} p99 {p99_ttft} | "
          f"TPOT p50 {p50_tpot} p99 {p99_tpot} | "
          f"KV {kvi.get('capacity_tokens', 'n/a'):,} tok | -> {args.output}")


if __name__ == "__main__":
    main()
