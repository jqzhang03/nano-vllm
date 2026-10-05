"""Compare predictive admission on/off against the same online request trace.

Run in the WSL inference environment::

    python benchmarks/admission_ablation.py --num-seqs 128 --arrival-mode poisson \
        --arrival-rate 16 --max-admission-wait-ms 2000
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import time

import torch
from fastapi import HTTPException

from nanovllm import LLM
from nanovllm.server import GenerationManager
from nanovllm.sampling_params import SamplingParams

try:
    from .bench import add_workload_args, build_workload, env_info, summarize
    from .scheduling_ablation import build_arrival_trace
except ImportError:
    from bench import add_workload_args, build_workload, env_info, summarize
    from scheduling_ablation import build_arrival_trace


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(parser)
    parser.add_argument("--arrival-mode", choices=("poisson", "constant", "burst"),
                        default="poisson")
    parser.add_argument("--arrival-rate", type=float, default=16.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8_e4m3"), default="auto")
    parser.add_argument("--no-swap-kv", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-seqs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-queued-requests", type=int, default=256)
    parser.add_argument("--max-deferred-requests", type=int, default=64)
    parser.add_argument("--max-admission-wait-ms", type=float, default=2000.0)
    parser.add_argument("--admission-work-budget-ms", type=float, default=10000.0)
    parser.add_argument("--admission-target-ttft-ms", type=float, default=2000.0)
    parser.add_argument("--admission-soft-pressure", type=float, default=0.85)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


async def run_variant(args, prompts, sampling_params, arrival_times,
                      dynamic_admission: bool) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="admission-bench")
    loop = asyncio.get_running_loop()
    engine = None
    manager = None
    try:
        engine = await loop.run_in_executor(executor, lambda: LLM(
            args.model,
            execution_mode="mixed",
            tensor_parallel_size=args.tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quantization=args.quantization,
            kv_cache_dtype=args.kv_cache_dtype,
            kv_swap=not args.no_swap_kv,
            enforce_eager=args.enforce_eager,
        ))
        warmup_count = min(args.warmup_seqs, len(prompts))
        if warmup_count:
            vocab = max(2, int(getattr(engine.tokenizer, "vocab_size", 32000)))
            warmup_prompts = []
            for index, prompt in enumerate(prompts[:warmup_count]):
                base = (vocab - 1 - index * 997) % vocab
                warmup_prompts.append([(base - offset * 7919) % vocab
                                       for offset in range(len(prompt))])
            await loop.run_in_executor(
                executor,
                lambda: engine.generate(
                    warmup_prompts,
                    SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=8),
                    use_tqdm=False,
                ),
            )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        engine._req_metrics = []
        engine._step_stats = engine._empty_step_stats()
        for scheduler in ((engine.prefill_scheduler, engine.decode_scheduler)
                          if engine._pd else (engine.scheduler,)):
            scheduler.reset_metrics()

        manager = GenerationManager(
            engine, executor, args.max_queued_requests,
            dynamic_admission=dynamic_admission,
            max_deferred_requests=args.max_deferred_requests,
            max_admission_wait_ms=args.max_admission_wait_ms,
            admission_work_budget_ms=args.admission_work_budget_ms,
            admission_target_ttft_ms=args.admission_target_ttft_ms,
            admission_soft_pressure=args.admission_soft_pressure,
        )
        manager.start()
        torch.cuda.synchronize()
        start = time.perf_counter()

        async def submit_one(index: int) -> dict:
            arrival = start + arrival_times[index]
            delay = arrival - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                handle = await manager.submit(
                    prompts[index], sampling_params[index], submitted_at=arrival)
            except HTTPException as exc:
                return {"index": index, "status": "rejected",
                        "status_code": exc.status_code, "error": str(exc.detail)}
            result = await handle.result
            return {
                "index": index,
                "status": "accepted" if "error" not in result else "failed",
                "admission_decision": handle.admission_decision,
                "admission_wait_ms": handle.admission_wait_ms,
                "predicted_ttft_ms": handle.predicted_ttft_ms,
                "queue_pressure": handle.queue_pressure,
                "estimated_output_tokens": handle.estimated_output_tokens,
                "ttft_ms": (None if handle.first_token_at is None else
                            (handle.first_token_at - arrival) * 1000.0),
                "e2e_ms": (None if handle.completed_at is None else
                           (handle.completed_at - arrival) * 1000.0),
                "completion_tokens": len(result.get("token_ids", [])),
                "error": result.get("error"),
            }

        rows = await asyncio.gather(*(submit_one(i) for i in range(len(prompts))))
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        admission = manager.admission_snapshot()
        await manager.close()
        accepted = [row for row in rows if row["status"] == "accepted"]
        total_output_tokens = sum(row["completion_tokens"] for row in accepted)
        ttfts = [row["ttft_ms"] for row in accepted if row["ttft_ms"] is not None]
        e2es = [row["e2e_ms"] for row in accepted if row["e2e_ms"] is not None]
        slo_met = [value <= args.admission_target_ttft_ms for value in ttfts]
        return {
            "dynamic_admission": dynamic_admission,
            "wall_seconds": wall,
            "requests": len(rows),
            "accepted": len(accepted),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "deferred_then_accepted": sum(
                row.get("admission_decision") == "accepted_after_defer" for row in rows),
            "output_tokens": total_output_tokens,
            "output_tokens_per_second": total_output_tokens / max(wall, 1e-9),
            "ttft_ms": summarize(ttfts),
            "e2e_ms": summarize(e2es),
            "target_ttft_met_percent": 100.0 * sum(slo_met) / max(1, len(slo_met)),
            "admission": admission,
            "per_request": rows,
        }
    finally:
        try:
            if manager is not None:
                await manager.close()
        finally:
            try:
                if engine is not None:
                    await loop.run_in_executor(executor, engine.exit)
            finally:
                executor.shutdown(wait=True)


async def async_main(args, prompts, sampling_params, arrival_times) -> dict:
    runs = {}
    for enabled in (False, True):
        name = "dynamic_on" if enabled else "dynamic_off"
        print(f"running {name}: {len(prompts)} requests, {args.arrival_mode}, "
              f"{args.arrival_rate:g} req/s ...", flush=True)
        runs[name] = await run_variant(
            args, prompts, sampling_params, arrival_times, enabled)
        result = runs[name]
        print(f"  accepted/rejected {result['accepted']}/{result['rejected']} | "
              f"deferred {result['deferred_then_accepted']} | "
              f"throughput {result['output_tokens_per_second']:.1f} tok/s | "
              f"TTFT p50/p99 {result['ttft_ms']['p50'] or 0:.1f}/"
              f"{result['ttft_ms']['p99'] or 0:.1f} ms")
    return runs


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    tokenizer = None
    if args.prompts_file:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    prompts, sampling_params = build_workload(args, tokenizer)
    arrivals = build_arrival_trace(len(prompts), args.arrival_mode,
                                   args.arrival_rate, args.seed + 1)
    runs = asyncio.run(async_main(args, prompts, sampling_params, arrivals))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or os.path.join(
        "results", f"admission_ablation_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as result_file:
        json.dump({
            "meta": {"model": args.model, "date": timestamp, **env_info()},
            "config": vars(args),
            "workload": {
                "num_requests": len(prompts),
                "arrival_mode": args.arrival_mode,
                "arrival_rate": args.arrival_rate,
                "arrival_times_seconds": arrivals,
                "prompt_tokens": [len(prompt) for prompt in prompts],
                "max_output_tokens": [params.max_tokens for params in sampling_params],
                "seed": args.seed,
            },
            "runs": runs,
        }, result_file, indent=2, default=str)
    print(f"full results -> {output_path}")


if __name__ == "__main__":
    main()
