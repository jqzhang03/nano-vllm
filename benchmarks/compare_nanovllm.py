"""Run a comparison workload on nano-vllm, emitting a JSON result file.

Must run in the nano-vllm conda env (see benchmarks/run_in_wsl.sh). The vLLM
side is run separately with compare_vllm.py in the isolated vllm-compare env;
compare_merge.py combines the JSON files.

Usage:
    python benchmarks/compare_nanovllm.py --workload results/compare_workload_clean.json \
        --kv-cache-dtype auto --output results/compare_nanovllm_clean_fp16.json
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

from nanovllm import LLM, SamplingParams  # noqa: E402

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")


def env_info(engine_version: str) -> dict:
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
    info["engine_version"] = engine_version
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

    llm = LLM(args.model, enforce_eager=False, tensor_parallel_size=1,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_batched_tokens=args.max_num_batched_tokens, max_num_seqs=args.max_num_seqs,
              kv_cache_dtype=args.kv_cache_dtype, quantization="none")
    # Warmup: JIT-compile torch.compile'd ops, allocate KV cache, capture CUDA graphs.
    llm.generate(["warm up"] * args.warmup_seqs,
                 SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)
                       for mt in max_tokens]
    t0 = time.perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    wall = time.perf_counter() - t0
    metrics = llm.collect_metrics()

    cfg = llm.model_runner.config  # must read before exit() (it deletes model_runner)
    kv_info = {
        "num_kvcache_blocks": cfg.num_kvcache_blocks,
        "kvcache_block_size": cfg.kvcache_block_size,
        "capacity_tokens": cfg.num_kvcache_blocks * cfg.kvcache_block_size,
    }
    llm.exit()
    rows = metrics["per_request"]
    total_out = sum(r["completion_tokens"] for r in rows)
    summary = compute_summary(rows)

    payload = {
        "engine": "nanovllm",
        "tag": spec["tag"],
        "kv_cache_dtype": args.kv_cache_dtype,
        "date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "meta": {
            **env_info("nano-vllm (this repo)"),
            "model": args.model,
            "config": {
                "max_num_batched_tokens": cfg.max_num_batched_tokens,
                "max_num_seqs": cfg.max_num_seqs,
                "max_model_len": cfg.max_model_len,
                "gpu_memory_utilization": cfg.gpu_memory_utilization,
                "kvcache_block_size": cfg.kvcache_block_size,
                "enforce_eager": cfg.enforce_eager,
                "quantization": cfg.quantization,
            },
            "kv_cache": kv_info,
        },
        "workload": {k: spec[k] for k in ("num_seqs", "min_input_len", "max_input_len",
                                          "min_output_len", "max_output_len", "shared_prefix_len")},
        "wall": wall,
        "throughput_tok_per_s": total_out / wall,
        "total_output_tokens": total_out,
        "summary": summary,
        "slo": slo_attainment(rows),
        "num_preemptions": metrics.get("num_preemptions", 0),
        "decode_steps": metrics["step_stats"]["decode_steps"],
        "prefill_steps": metrics["step_stats"]["prefill_steps"],
        "per_request": rows,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    s = summary
    print(f"[nanovllm {args.kv_cache_dtype}] wall {wall:.2f}s | "
          f"throughput {total_out / wall:.1f} tok/s | preemptions {payload['num_preemptions']} | "
          f"TTFT p50 {s['ttft']['p50'] * 1000:.1f}ms p99 {s['ttft']['p99'] * 1000:.1f}ms | "
          f"TPOT p50 {s['tpot']['p50'] * 1000:.1f}ms p99 {s['tpot']['p99'] * 1000:.1f}ms | "
          f"KV {kv_info['capacity_tokens']:,} tok | -> {args.output}")


if __name__ == "__main__":
    main()
