"""Run a structured, low-concurrency generation smoke across local checkpoints.

The default set matches the 7B/8B INT4 streaming-load checks on a 16 GB GPU.
Sixteen prompts per model cover short, medium, and long inputs; requests run in
batches of four so the smoke checks generation without pretending to be a
concurrency-capacity measurement.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
import time
import traceback

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams

try:
    from .bench import env_info
except ImportError:
    from bench import env_info


DEFAULT_MODELS = (
    "/home/zjq/huggingface/Qwen2.5-7B,"
    "/home/zjq/huggingface/Mistral-7B-v0.1,"
    "/home/zjq/huggingface/Llama-3.1-8B"
)

PROMPT_SEEDS = [
    ("short", "Explain one practical use of a hash table."),
    ("short", "What is the difference between weather and climate?"),
    ("short", "Give a concise definition of numerical stability."),
    ("short", "Name two checks to run before deploying a database migration."),
    ("medium", "Describe how a small community library can organize a digital archive."),
    ("medium", "Explain why a city might combine buses, cycling lanes, and rail."),
    ("medium", "Outline a careful process for comparing two image compression formats."),
    ("medium", "Describe how a research team can make a data analysis reproducible."),
    ("medium", "Explain the tradeoffs between a local cache and a shared remote cache."),
    ("medium", "Give a practical plan for reducing energy use in a small office."),
    ("long", "A field team is monitoring river quality after seasonal storms."),
    ("long", "A software group is migrating a service from one database engine to another."),
    ("long", "A public school is planning a year-long robotics and science program."),
    ("long", "A small manufacturer wants to improve inventory forecasts and delivery times."),
    ("long", "A museum is building an accessible audio guide for a large collection."),
    ("long", "A regional clinic is reviewing how to schedule appointments and follow-ups."),
]

LONG_CONTEXT = (
    "The team records measurements with dates, units, and a short explanation of the method. "
    "They compare observations from several locations, check unusual values against the "
    "original notes, and describe uncertainty before drawing conclusions. The report should "
    "separate direct evidence from assumptions, list the limits of the sample, and explain "
    "which follow-up measurements would change the decision. "
)


def _prompts(tokenizer, max_prompt_tokens: int) -> list[dict]:
    prompts = []
    for index, (length_class, seed) in enumerate(PROMPT_SEEDS):
        if length_class == "short":
            text = f"{seed} Give the answer in two sentences."
        elif length_class == "medium":
            text = (f"{seed} Consider the people involved, the available resources, and the "
                    "risks of an incomplete plan. Give a clear sequence of steps and explain "
                    "how success could be checked. Include one likely failure mode and a "
                    "reasonable response to it.")
        else:
            text = (f"{seed} Prepare a concise planning note for the team. "
                    + LONG_CONTEXT * 6
                    + "Summarize the first actions, the information to preserve, and the "
                    "criteria for deciding whether the plan is working.")
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        truncated = len(token_ids) > max_prompt_tokens
        token_ids = token_ids[:max_prompt_tokens]
        if not token_ids:
            raise ValueError(f"prompt {index} encoded to no tokens")
        prompts.append({
            "index": index,
            "length_class": length_class,
            "prompt_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "prompt_tokens": len(token_ids),
            "truncated": truncated,
            "token_ids": token_ids,
        })
    return prompts


def _run_model(model: str, args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    model_dir = os.path.expanduser(model)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True,
                                              local_files_only=True)
    prompts = _prompts(tokenizer, args.max_prompt_tokens)
    max_input = max(row["prompt_tokens"] for row in prompts)
    max_model_len = max_input + args.max_output_tokens
    model_limit = int(getattr(config, "max_position_embeddings", 4096))
    if max_model_len > model_limit:
        raise ValueError(
            f"smoke needs {max_model_len} tokens, checkpoint limit is {model_limit}")

    torch.cuda.reset_peak_memory_stats()
    engine_started = time.perf_counter()
    llm = None
    runner = None
    model_tensors = []
    rows = []
    batches = []
    try:
        max_batched_tokens = max(args.batch_size * max_input,
                                 args.max_output_tokens * args.batch_size, 1024)
        llm = LLM(
            model_dir,
            execution_mode="mixed",
            tensor_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=max_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            quantization=args.quantization,
            int4_group_size=args.int4_group_size,
            kv_swap=False,
        )
        load_seconds = time.perf_counter() - engine_started
        runner = llm.model_runner
        model_tensors = [tensor for tensor in
                         list(runner.model.parameters()) + list(runner.model.buffers())
                         if tensor.is_cuda]
        weights_and_buffers = sum(
            tensor.numel() * tensor.element_size() for tensor in model_tensors)
        hf_config = llm.config.hf_config
        llm.generate([[tokenizer.eos_token_id or 1]],
                     SamplingParams(temperature=1.0, max_tokens=2), use_tqdm=False)
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start:start + args.batch_size]
            group_start = time.perf_counter()
            original_forward = runner.sampler.forward
            runner.sampler.forward = lambda logits, temperatures: logits.argmax(dim=-1)
            try:
                outputs = llm.generate(
                    [row["token_ids"] for row in batch],
                    SamplingParams(temperature=1.0, ignore_eos=False,
                                   max_tokens=args.max_output_tokens),
                    use_tqdm=False,
                )
            finally:
                runner.sampler.forward = original_forward
            torch.cuda.synchronize()
            batch_wall = time.perf_counter() - group_start
            metrics = sorted(llm.collect_metrics()["per_request"],
                             key=lambda row: row["seq_id"])
            batches.append({"first_index": start, "count": len(batch),
                            "wall_seconds": batch_wall})
            for index, (prompt_row, output) in enumerate(zip(batch, outputs)):
                token_ids = output["token_ids"]
                eos_id = tokenizer.eos_token_id
                request_metrics = metrics[index] if index < len(metrics) else {}
                first_token = request_metrics.get("t_first_token")
                submitted = request_metrics.get("t_submitted")
                completed = request_metrics.get("t_completed")
                rows.append({
                    "index": prompt_row["index"],
                    "length_class": prompt_row["length_class"],
                    "prompt_sha256": prompt_row["prompt_sha256"],
                    "prompt_tokens": prompt_row["prompt_tokens"],
                    "prompt_truncated": prompt_row["truncated"],
                    "completion_tokens": len(token_ids),
                    "first_token_id": token_ids[0] if token_ids else None,
                    "eos_token_id": eos_id,
                    "eos_generated": eos_id in token_ids if eos_id is not None else False,
                    "text": output.get("text", ""),
                    "ttft_ms": ((first_token - submitted) * 1000.0
                                if first_token is not None and submitted is not None else None),
                    "e2e_ms": ((completed - submitted) * 1000.0
                               if completed is not None and submitted is not None else None),
                    "batch_wall_seconds": batch_wall,
                    "status": "ok" if token_ids else "empty_output",
                })

        torch.cuda.synchronize()
        return {
            "model": model_dir,
            "model_type": getattr(hf_config, "model_type", None),
            "revision": getattr(hf_config, "_commit_hash", None),
            "quantization": args.quantization,
            "int4_group_size": args.int4_group_size,
            "streaming_load": getattr(runner, "streaming", None),
            "load_seconds": load_seconds,
            "total_seconds": time.perf_counter() - started,
            "max_model_len": max_model_len,
            "model_position_limit": model_limit,
            "gpu_weight_and_buffer_bytes": weights_and_buffers,
            "kv_cache_dtype": llm.config.kv_cache_dtype,
            "kv_cache_blocks": llm.config.num_kvcache_blocks,
            "kv_block_size": llm.config.kvcache_block_size,
            "kv_capacity_tokens": (llm.config.num_kvcache_blocks
                                   * llm.config.kvcache_block_size),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "requests": len(rows),
            "completed": sum(row["status"] == "ok" for row in rows),
            "empty_outputs": sum(row["status"] == "empty_output" for row in rows),
            "batches": batches,
            "per_request": rows,
            "status": "complete" if len(rows) == len(prompts) else "partial",
        }
    finally:
        if llm is not None:
            llm.exit()
        llm = None
        runner = None
        model_tensors.clear()
        del model_tensors
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=DEFAULT_MODELS,
                        help="comma-separated local model directories")
    parser.add_argument("--quantization", choices=("int4", "fp8", "none"),
                        default="int4")
    parser.add_argument("--int4-group-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="disable CUDA Graph capture for compatibility diagnostics")
    parser.add_argument("--output", default="results/supplemental/multimodel_smoke_20261016.json")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_output_tokens < 1:
        parser.error("batch size and output tokens must be positive")
    if args.max_prompt_tokens < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("prompt cap must be positive and GPU utilization must be in (0, 1]")

    output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    models = [os.path.expanduser(item.strip()) for item in args.models.split(",")
              if item.strip()]
    payload = {
        "schema_version": 1,
        "benchmark": "multimodel_int4_generation_smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {**env_info(), "gpu": torch.cuda.get_device_name(0)},
        "config": vars(args),
        "models": [],
    }
    for model in models:
        print(f"\nStarting smoke: {model}", flush=True)
        try:
            result = _run_model(model, args)
        except Exception as exc:
            memory = {}
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                memory = {
                    "cuda_free_bytes_at_failure": free_bytes,
                    "cuda_total_bytes": total_bytes,
                    "cuda_allocated_bytes_at_failure": torch.cuda.memory_allocated(),
                    "cuda_reserved_bytes_at_failure": torch.cuda.memory_reserved(),
                    "cuda_peak_allocated_bytes_before_failure": (
                        torch.cuda.max_memory_allocated()),
                }
            result = {
                "model": model,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
                **memory,
            }
            torch.cuda.empty_cache()
        payload["models"].append(result)
        with open(output + ".tmp", "w", encoding="utf-8") as result_file:
            json.dump(payload, result_file, indent=2, default=str)
        os.replace(output + ".tmp", output)
        print(f"{result['status']}: {model}", flush=True)
    print(f"JSON -> {output}", flush=True)


if __name__ == "__main__":
    main()
