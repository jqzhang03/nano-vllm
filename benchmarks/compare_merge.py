"""Merge nano-vllm / vLLM comparison JSONs into a side-by-side markdown report.

Usage:
    python benchmarks/compare_merge.py results/compare_*.json

or with explicit files. Writes results/compare_report.md and .csv.
Pure stdlib — runs anywhere.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

METRIC_ROWS = [
    ("throughput_tok_per_s", "throughput (output tok/s)", "hi"),
    ("summary.ttft.avg", "TTFT avg (ms)", "lo"),
    ("summary.ttft.p50", "TTFT p50 (ms)", "lo"),
    ("summary.ttft.p99", "TTFT p99 (ms)", "lo"),
    ("summary.tpot.avg", "TPOT avg (ms)", "lo"),
    ("summary.tpot.p50", "TPOT p50 (ms)", "lo"),
    ("summary.tpot.p99", "TPOT p99 (ms)", "lo"),
    ("summary.e2e.avg", "E2E avg (s)", "lo"),
    ("summary.e2e.p50", "E2E p50 (s)", "lo"),
    ("summary.e2e.p99", "E2E p99 (s)", "lo"),
    ("slo.ttft_ok", "SLO TTFT<500ms (%)", "hi"),
    ("slo.tpot_ok", "SLO TPOT<10ms (%)", "hi"),
    ("num_preemptions", "preemptions", "lo"),
]


def get_path(payload: dict, path: str):
    obj = payload
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


def fmt(v, path: str) -> str:
    if v is None:
        return "n/a"
    if path in ("throughput_tok_per_s", "slo.ttft_ok", "slo.tpot_ok"):
        return f"{v:.1f}"
    if path.startswith("summary.ttft.") or path.startswith("summary.tpot."):
        return f"{v * 1000:.1f}"
    if path.startswith("summary.e2e."):
        return f"{v:.2f}"
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="*")
    args = p.parse_args()

    files = args.files or sorted(glob.glob(os.path.join("results", "compare_*.json")))
    files = [f for f in files if "workload" not in f]
    if not files:
        print("no result JSON files found (results/compare_*.json)")
        return

    payloads = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            payloads.append(json.load(fh))

    groups: dict[tuple, dict] = defaultdict(dict)
    for pl in payloads:
        groups[(pl["tag"], pl["kv_cache_dtype"])][pl["engine"]] = pl

    md = ["# nano-vllm vs vLLM — side-by-side",
          "",
          "_generated " + " ".join(files) + "_",
          "",
          "**Ratio convention:** `nano/vllm`. For latencies (TTFT/TPOT/E2E) and "
          "preemptions **lower is better**; for throughput and SLO% **higher is better**.",
          "",
          "**Metric caveats:** nano-vllm latencies are exact per-request timings "
          "(driver-side timestamps). vLLM (V1 engine) does not expose per-request "
          "timings in the offline API, so its avg is exact (histogram sum/count) "
          "but its p50/p99 are approximate (linear interpolation over histogram "
          "buckets).",
          ""]

    csv_rows = [["tag", "kv_cache_dtype", "metric", "nanovllm", "vllm", "ratio_nano_over_vllm"]]

    for (tag, kvd), eng in sorted(groups.items()):
        nano = eng.get("nanovllm")
        vllm = eng.get("vllm")
        md.append(f"## workload `{tag}` · kv_cache_dtype=`{kvd}`")
        md.append("")
        if nano is None or vllm is None:
            md.append(f"_missing side: nanovllm={nano is not None} vllm={vllm is not None}_")
            md.append("")
            continue
        wl = nano["workload"]
        md.append(f"**{wl['num_seqs']} seqs | input {wl['min_input_len']}-{wl['max_input_len']} tok | "
                  f"output {wl['min_output_len']}-{wl['max_output_len']} tok | "
                  f"shared prefix {wl['shared_prefix_len']}**")
        md.append("")
        md.append("| metric | nano-vllm | vLLM | ratio (nano/vllm) |")
        md.append("|---|---|---|---|")
        for path, label, _ in METRIC_ROWS:
            nv, vv = get_path(nano, path), get_path(vllm, path)
            if nv is None and vv is None:
                continue
            ratio = "—" if (nv is None or vv is None or vv == 0) else f"{nv / vv:.2f}"
            md.append(f"| {label} | {fmt(nv, path)} | {fmt(vv, path)} | {ratio} |")
            csv_rows.append([tag, kvd, label, fmt(nv, path), fmt(vv, path), ratio])
        md.append("")

        # config difference block (honest-conditions section)
        md.append("### conditions")
        md.append("")
        md.append("| | nano-vllm | vLLM |")
        md.append("|---|---|---|")
        nc, vc = nano["meta"]["config"], vllm["meta"]["config"]
        nk, vk = nano["meta"].get("kv_cache", {}), vllm["meta"].get("kv_cache", {})
        pairs = [
            ("engine version", nano["meta"].get("engine_version"), vllm["meta"].get("engine_version")),
            ("torch", nano["meta"].get("torch"), vllm["meta"].get("torch")),
            ("flash-attn", nano["meta"].get("flash_attn"), vllm["meta"].get("flash_attn")),
            ("max_model_len", nc.get("max_model_len"), vc.get("max_model_len")),
            ("max_num_batched_tokens", nc.get("max_num_batched_tokens"), vc.get("max_num_batched_tokens")),
            ("max_num_seqs", nc.get("max_num_seqs"), vc.get("max_num_seqs")),
            ("gpu_memory_utilization", nc.get("gpu_memory_utilization"), vc.get("gpu_memory_utilization")),
            ("block size", nc.get("kvcache_block_size"), vk.get("block_size")),
            ("KV capacity (tokens)", nk.get("capacity_tokens"), vk.get("capacity_tokens")),
            ("CUDA graphs", "on" if not nc.get("enforce_eager") else "off",
             "off" if vc.get("enforce_eager") else "on"),
            ("prefix caching", "on", "on" if vc.get("enable_prefix_caching") else "off"),
            ("wall (s)", f"{nano['wall']:.2f}", f"{vllm['wall']:.2f}"),
            ("output tokens", nano.get("total_output_tokens"), vllm.get("total_output_tokens")),
        ]
        for label, nv, vv in pairs:
            md.append(f"| {label} | {nv} | {vv} |")
        md.append("")

    report_dir = "results"
    md_path = os.path.join(report_dir, "compare_report.md")
    csv_path = os.path.join(report_dir, "compare_report.csv")
    os.makedirs(report_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print("\n".join(md))
    print(f"\n-> {md_path}\n-> {csv_path}")


if __name__ == "__main__":
    main()
