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

Run from the WSL conda env (see BENCHMARKS.md):

    python benchmarks/bench.py --num-seqs 256
    python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512
    python benchmarks/bench.py --num-seqs 256 --compare-vllm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime, timezone

import torch

from nanovllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# workload construction (shared with profile.py)
# ---------------------------------------------------------------------------

def add_workload_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--min-input-len", type=int, default=128)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=64)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--shared-prefix-len", type=int, default=0,
                        help="common prefix shared by all prompts (prefix-cache workload)")
    parser.add_argument("--prompts-file", default=None,
                        help="JSONL of {\"prompt\": ...} lines; overrides synthetic lengths")


def build_workload(args, tokenizer=None):
    """Return (prompts, sampling_params): token-id lists + per-seq params.

    prompt长度不超过max_model_len即可（默认范围在4096以内）。"""
    rng = random.Random(args.seed)
    if args.prompts_file:
        assert tokenizer is not None, "--prompts-file requires a tokenizer"
        with open(args.prompts_file, encoding="utf-8") as f:
            texts = [json.loads(line)["prompt"] for line in f if line.strip()]
        if len(texts) > args.num_seqs:
            texts = rng.sample(texts, args.num_seqs)
        prompts = [tokenizer.encode(t) for t in texts]
    else:
        prompts = [rng.sample(range(0, 10000), rng.randint(args.min_input_len, args.max_input_len))
                   for _ in range(args.num_seqs)]
    if args.shared_prefix_len > 0:
        prefix = rng.sample(range(0, 10000), args.shared_prefix_len)
        prompts = [prefix + p for p in prompts]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=rng.randint(args.min_output_len, args.max_output_len))
                       for _ in prompts]
    return prompts, sampling_params


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
              quantization=getattr(args, "quantization", "none"),
              awq_scales_path=getattr(args, "awq_scales_path", ""),
              quantize_lm_head=getattr(args, "quantize_lm_head", False),
              speculative=getattr(args, "speculative", "none"))
    # 预热：触发torch.compile/triton的JIT编译、分配KV Cache、捕获CUDA Graph
    llm.generate(["warm up"] * args.warmup_seqs,
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
    stats = metrics["step_stats"]
    prefill_tps = stats["prefill_tokens"] / stats["prefill_time"] if stats["prefill_time"] > 0 else None
    decode_tps = stats["decode_tokens"] / stats["decode_time"] if stats["decode_time"] > 0 else None
    return {
        "ttft": summarize(ttfts),
        "tpot": summarize(tpots),
        "e2e": summarize(e2es),
        "prefill_throughput_tok_per_s": prefill_tps,
        "decode_throughput_tok_per_s": decode_tps,
        "prefill_steps": stats["prefill_steps"],
        "decode_steps": stats["decode_steps"],
        "prefill_tokens": stats["prefill_tokens"],
        "decode_tokens": stats["decode_tokens"],
        "num_preemptions": metrics.get("num_preemptions", 0),
    }


def print_report(args, wall, metrics, kv_info, vllm_res=None, out_path=None):
    s = make_summary(metrics)
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
    line("preemptions", f"{s['num_preemptions']} (KV cache 不足导致的抢占；0 表示容量充足)")
    line("TTFT", f"avg {_fmt_seconds(s['ttft']['avg'])} | p50 {_fmt_seconds(s['ttft']['p50'])} | "
                 f"p99 {_fmt_seconds(s['ttft']['p99'])} (n={s['ttft']['count']})")
    line("TPOT", f"avg {_fmt_seconds(s['tpot']['avg'])} | p50 {_fmt_seconds(s['tpot']['p50'])} | "
                 f"p99 {_fmt_seconds(s['tpot']['p99'])} (n={s['tpot']['count']})")
    line("E2E", f"avg {_fmt_seconds(s['e2e']['avg'])} | p50 {_fmt_seconds(s['e2e']['p50'])} | "
                f"p99 {_fmt_seconds(s['e2e']['p99'])} (n={s['e2e']['count']})")
    line("SLO", f"TTFT<{args.slo_ttft_ms:.0f}ms: "
                f"{100 * sum(1 for r in metrics['per_request'] if r['t_first_token'] is not None and (r['t_first_token'] - r['t_submitted']) * 1000 < args.slo_ttft_ms) / max(1, len(metrics['per_request'])):.1f}% | "
                f"TPOT<{args.slo_tpot_ms:.0f}ms: "
                f"{100 * sum(1 for r in metrics['per_request'] if r['t_completed'] is not None and r['t_first_token'] is not None and r['completion_tokens'] > 1 and (r['t_completed'] - r['t_first_token']) / (r['completion_tokens'] - 1) * 1000 < args.slo_tpot_ms) / max(1, len(metrics['per_request'])):.1f}%")
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
    try:
        import flash_attn
        info["flash_attn"] = flash_attn.__version__
    except ImportError:
        info["flash_attn"] = None
    return info


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(p)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--kv-cache-dtype", default="auto",
                   help="KV缓存dtype: auto(模型dtype) 或 fp8_e4m3(FP8量化，容量翻倍)")
    p.add_argument("--quantization", default="none",
                   help="权重量化: none | w8a8(int8, Triton GEMM) | int4(per-group int4, Triton GEMM) | "
                        "awq(int4+激活感知缩放) | sparse24(2:4结构化剪枝, Triton稀疏GEMM)")
    p.add_argument("--awq-scales-path", default="",
                   help="AWQ缩放文件（benchmarks/awq_calibrate.py产出）；空=随机token内联校准")
    p.add_argument("--quantize-lm-head", action="store_true",
                   help="同时量化LM head（默认不量化，见BENCHMARKS.md §10）")
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
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True) if args.prompts_file else None
    prompts, sampling_params = build_workload(args, tokenizer)

    print(f"running nano-vllm ({args.num_seqs} seqs, input {args.min_input_len}-{args.max_input_len}, "
          f"output {args.min_output_len}-{args.max_output_len}, shared-prefix {args.shared_prefix_len}, "
          f"eager={args.enforce_eager}) ...")
    batches, kv_info = run_nanovllm(args, prompts, sampling_params)
    metrics = batches[0]

    vllm_res = run_vllm(args, prompts, sampling_params) if args.compare_vllm else None

    tag = (f"n{args.num_seqs}_i{args.min_input_len}-{args.max_input_len}_"
           f"o{args.min_output_len}-{args.max_output_len}"
           + (f"_prefix{args.shared_prefix_len}" if args.shared_prefix_len else "")
           + ("_eager" if args.enforce_eager else ""))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output or os.path.join("results", f"bench_{tag}_{ts}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = {
        "meta": {"model": args.model, **env_info(), "date": ts, "kv_cache": kv_info},
        "workload": {"num_seqs": args.num_seqs, "min_input_len": args.min_input_len,
                     "max_input_len": args.max_input_len, "min_output_len": args.min_output_len,
                     "max_output_len": args.max_output_len, "shared_prefix_len": args.shared_prefix_len,
                     "prompts_file": args.prompts_file, "enforce_eager": args.enforce_eager, "tp": args.tp,
                     "repeat_batches": args.repeat_batches},
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
