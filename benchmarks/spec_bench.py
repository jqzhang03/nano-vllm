"""n-gram投机解码基准：同一workload下 spec on/off 对比 + 接受率统计。

三种workload风格（诚实光谱，同一引擎同一seed）：
  free     随机token段落（最差情况：几乎无重复n-gram）
  json     JSON记录续写（结构化：中度重复）
  repeat   重复指令（最好情况：模型输出高重复度 → α高）

报告：吞吐（decode_tokens/wall）、TTFT/TPOT p50、接受率α、平均草稿长γ、
tokens/步、verify步数、抢占数。输出CSV + JSON。

用法（WSL，GPU）：
    python benchmarks/spec_bench.py [--styles free,json,repeat] [--num-seqs 64] [--fp8]
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams

try:
    from .bench import build_workload_data, env_info, prompt_token_hash
except ImportError:
    from bench import build_workload_data, env_info, prompt_token_hash

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
EOS_IDS = {151643, 151645, 151646, 151647}  # Qwen3 special tokens（仅用于构造重复文本时避开）


def build_prompts(style: str, tokenizer, num_seqs: int, min_len: int,
                  max_len: int, seed: int) -> list[list[int]]:
    """Create one deterministic, distinct prompt per request."""
    rng = random.Random(seed)
    if style == "free":
        vocab = int(getattr(tokenizer, "vocab_size", 32000))
        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        candidates = [token for token in range(vocab) if token not in special_ids]
        return [[rng.choice(candidates) for _ in range(rng.randint(min_len, max_len))]
                for _ in range(num_seqs)]
    texts = []
    for index in range(num_seqs):
        if style == "repeat":
            digit = rng.randrange(10)
            pattern = " ".join(str((digit + offset) % 10) for offset in range(8))
            texts.append(
                f"Request {index}: repeat this numeric cycle and continue it: {pattern} "
                f"{pattern} {pattern}")
        elif style == "json":
            records = []
            for offset in range(12):
                item_id = index * 100 + offset
                records.append(
                    f'{{"id": {item_id}, "name": "user_{item_id}", '
                    f'"score": {(item_id % 100) / 100:.2f}, '
                    f'"active": {item_id % 2 == 0}, '
                    f'"tags": ["a", "b", "c_{item_id % 3}"]}}')
            texts.append(
                f"Request {index}: complete this JSON array with distinct records "
                f"that follow the schema exactly:\n[" + ",\n".join(records) + "]")
        else:
            raise ValueError(f"unsupported synthetic prompt style: {style}")
    return [list(tokenizer.encode(text)) for text in texts]


def _summary(values: list[float]) -> dict:
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "mean": None}
    pick = lambda q: values[min(int(len(values) * q), len(values) - 1)]
    return {"count": len(values), "p50": pick(0.50), "p95": pick(0.95),
            "p99": pick(0.99), "mean": statistics.fmean(values)}


def _output_stats(token_ids: list[int]) -> dict:
    grams = [tuple(token_ids[index:index + 4])
             for index in range(max(0, len(token_ids) - 3))]
    distinct = len(set(grams)) / len(grams) if grams else None
    counts = {}
    for token_id in token_ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    entropy = 0.0
    for count in counts.values():
        probability = count / max(1, len(token_ids))
        entropy -= probability * math.log2(probability)
    return {
        "tokens": len(token_ids),
        "distinct_4": distinct,
        "fourgram_repeat_rate": (1.0 - distinct if distinct is not None else None),
        "token_entropy_bits": entropy if token_ids else None,
    }


def _bootstrap_median_ci(values: list[float], seed: int,
                         samples: int = 2000) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "median": None, "ci95_low": None, "ci95_high": None}
    rng = random.Random(seed)
    medians = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(resample))
    medians.sort()
    return {
        "count": len(values),
        "median": statistics.median(values),
        "ci95_low": medians[int(samples * 0.025)],
        "ci95_high": medians[min(samples - 1, int(samples * 0.975))],
    }


def run_case(style: str, speculative: str, prompts_tok: list[list[int]],
             tokenizer, args, *, seed: int,
             request_metadata: list[dict] | None = None) -> dict:
    llm = LLM(args.model, speculative=speculative,
              medusa_path=args.medusa_path if speculative == "medusa" else "",
              eagle_path=args.eagle_path if speculative == "eagle" else "",
              max_draft_len=args.max_draft_len,
              kv_cache_dtype="fp8_e4m3" if args.fp8 else "auto",
              max_model_len=args.max_model_len,
              max_num_seqs=max(8, len(prompts_tok)),
              gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        if args.sampling_mode == "greedy":
            llm.model_runner.sampler.forward = lambda logits, _temps: logits.argmax(dim=-1)
        warmup_rng = random.Random(seed + 100003)
        vocab = int(getattr(tokenizer, "vocab_size", 32000))
        warmups = [[warmup_rng.randrange(max(1, vocab - 4)) for _ in range(64)]
                   for _ in range(min(4, len(prompts_tok)))]
        llm.generate(warmups, SamplingParams(temperature=0.6, max_tokens=8),
                     use_tqdm=False)
        llm.reset_benchmark_metrics()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampling_temperature = 1.0 if args.sampling_mode == "greedy" else args.temperature
        sampling_params = [SamplingParams(
            temperature=sampling_temperature,
            max_tokens=args.max_output_len,
            ignore_eos=args.ignore_eos,
        ) for _ in prompts_tok]
        started = time.perf_counter()
        outputs = llm.generate(prompts_tok, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        wall = time.perf_counter() - started
        metrics = llm.collect_metrics()
        per_request = sorted(metrics["per_request"], key=lambda row: row["seq_id"])
        if len(per_request) != len(prompts_tok):
            raise RuntimeError(
                f"expected {len(prompts_tok)} request metrics, received {len(per_request)}")
        if len({row["seq_id"] for row in per_request}) != len(per_request):
            raise RuntimeError("speculative benchmark returned duplicate request metrics")
        rows = []
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        for index, (request, output) in enumerate(zip(per_request, outputs)):
            token_stats = _output_stats(output["token_ids"])
            token_ids = output["token_ids"]
            rows.append({
                "request_index": index,
                "request_id": (request_metadata[index].get("request_id")
                               if request_metadata is not None else None),
                "content_class": (request_metadata[index].get("content_class", style)
                                  if request_metadata is not None else style),
                "prompt_tokens": len(prompts_tok[index]),
                "prompt_token_hash": prompt_token_hash(prompts_tok[index]),
                "ttft_ms": (None if request["t_first_token"] is None else
                            (request["t_first_token"] - request["t_submitted"]) * 1000),
                "tpot_ms": (None if request["completion_tokens"] <= 1 else
                            (request["t_completed"] - request["t_first_token"])
                            * 1000 / (request["completion_tokens"] - 1)),
                "e2e_ms": (None if request["t_completed"] is None else
                           (request["t_completed"] - request["t_submitted"]) * 1000),
                "spec_proposed_tokens": request.get("spec_proposed_tokens", 0),
                "spec_accepted_tokens": request.get("spec_accepted_tokens", 0),
                "spec_verify_steps": request.get("spec_verify_steps", 0),
                "model_eos_terminated": bool(
                    eos_token_id is not None and token_ids
                    and token_ids[-1] == eos_token_id),
                "alpha": (request.get("spec_accepted_tokens", 0)
                          / request["spec_proposed_tokens"]
                          if request.get("spec_proposed_tokens", 0) else None),
                **token_stats,
            })
        stats = metrics["step_stats"]
        alpha = (stats["spec_accepted_drafts"] / stats["spec_draft_tokens"]
                 if stats.get("spec_draft_tokens") else None)
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        return {
            "style": style,
            "speculative": speculative,
            "seed": seed,
            "wall_seconds": wall,
            "output_tokens": output_tokens,
            "mean_output_tokens_per_request": output_tokens / max(1, len(rows)),
            "model_eos_terminated_requests": sum(
                row["model_eos_terminated"] for row in rows),
            "model_eos_terminated_percent": (
                100.0 * sum(row["model_eos_terminated"] for row in rows)
                / max(1, len(rows))),
            "throughput_tok_per_s": output_tokens / max(wall, 1e-9),
            "ttft_ms": _summary([row["ttft_ms"] for row in rows]),
            "tpot_ms": _summary([row["tpot_ms"] for row in rows]),
            "e2e_ms": _summary([row["e2e_ms"] for row in rows]),
            "preemptions": metrics.get("num_preemptions", 0),
            "prefill_steps": stats["prefill_steps"],
            "decode_steps": stats["decode_steps"],
            "spec_steps": stats.get("spec_steps", 0),
            "spec_verify_tokens": stats.get("spec_verify_tokens", 0),
            "spec_proposed_tokens": stats.get("spec_draft_tokens", 0),
            "spec_accepted_tokens": stats.get("spec_accepted_drafts", 0),
            "alpha": alpha,
            "avg_draft_len": (stats["spec_draft_tokens"] / stats["spec_rows"]
                              if stats.get("spec_rows") else None),
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "output_token_fingerprint": hashlib.sha256(json.dumps(
                [output["token_ids"] for output in outputs],
                separators=(",", ":")).encode("utf-8")).hexdigest(),
            "content_metrics": {
                "distinct_4": _summary([row["distinct_4"] for row in rows]),
                "fourgram_repeat_rate": _summary(
                    [row["fourgram_repeat_rate"] for row in rows]),
                "token_entropy_bits": _summary(
                    [row["token_entropy_bits"] for row in rows]),
                "per_request_alpha": _summary(
                    [row["alpha"] for row in rows]),
            },
            "per_request": rows,
        }
    finally:
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--seeds must be comma-separated non-negative integers") from exc
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("--seeds must contain non-negative integers")
    return seeds


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _trace_prompts(style: str, prompts: list[list[int]], metadata: list[dict],
                   count: int, seed: int) -> tuple[list[list[int]], list[dict]]:
    if style == "sharegpt":
        accepted_classes = {"sharegpt", "sharegpt_summary"}
    else:
        accepted_classes = {style}
    candidates = [index for index, row in enumerate(metadata)
                  if row.get("content_class") in accepted_classes]
    if style == "sharegpt" and not any(row.get("content_class") for row in metadata):
        candidates = list(range(len(prompts)))
    if len(candidates) < count:
        raise ValueError(
            f"content class {style!r} has {len(candidates)} prompts, needs {count}")
    selected = random.Random(seed).sample(candidates, count)
    return [prompts[index] for index in selected], [metadata[index] for index in selected]


def _prompt_fingerprint(prompts: list[list[int]], output_len: int,
                        ignore_eos: bool) -> str:
    payload = {"prompts": prompts, "max_output_len": output_len,
               "ignore_eos": ignore_eos}
    encoded = json.dumps(payload, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_results(output_path: Path, payload: dict,
                  flat_rows: list[dict]) -> tuple[Path, Path]:
    """Atomically checkpoint completed pairs so a later failure keeps prior data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix(".csv")
    temp_json = output_path.with_name(output_path.name + ".tmp")
    temp_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temp_json.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2, default=str)
    rows = list(flat_rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temp_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_json, output_path)
    os.replace(temp_csv, csv_path)
    return output_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL, help="model directory (default Qwen3-0.6B)")
    parser.add_argument("--styles", default="free,json,repeat",
                        help="free,json,repeat,sharegpt,structured,sharegpt_summary,open_ended")
    parser.add_argument("--prompts-file", default=None,
                        help="optional request JSONL built by sharegpt_trace.py")
    parser.add_argument("--chat-template", action="store_true",
                        help="wrap prompt-file user text with the tokenizer chat template")
    parser.add_argument("--speculative", default="none,ngram",
                        help="comma-separated modes: none,ngram,medusa,eagle")
    parser.add_argument("--medusa-path", default="results/medusa_heads.pt")
    parser.add_argument("--eagle-path", default="results/eagle_layer.pt")
    parser.add_argument("--max-draft-len", type=int, default=4,
                        help="fallback draft length gamma")
    parser.add_argument("--max-draft-lens", default=None,
                        help="optional gamma sweep, for example 2,4")
    parser.add_argument("--num-seqs", type=int, default=64)
    parser.add_argument("--batch-sizes", default=None,
                        help="comma-separated batch sizes; defaults to --num-seqs")
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--min-input-len", type=int, default=256)
    parser.add_argument("--max-output-len", type=int, default=96)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--fp8", action="store_true", help="use FP8 KV cache")
    parser.add_argument("--seeds", default=None,
                        help="comma-separated input/sampler seeds; defaults to --seed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sampling-mode", choices=("random", "greedy"), default="random")
    parser.add_argument("--temperature", type=float, default=0.6)
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument("--ignore-eos", dest="ignore_eos", action="store_true")
    eos_group.add_argument("--natural-stop", dest="ignore_eos", action="store_false")
    parser.set_defaults(ignore_eos=True)
    parser.add_argument("--output", default="results/spec_bench_supplemental.json")
    parser.add_argument("--resume", action="store_true",
                        help="resume an incomplete run from its checkpointed output")
    args = parser.parse_args()
    args.model = os.path.expanduser(args.model)
    if args.repeats < 1 or args.num_seqs < 1:
        raise ValueError("--repeats and --num-seqs must be positive")
    if args.max_output_len < 1 or args.temperature <= 0:
        raise ValueError("output length and random-sampling temperature must be positive")

    seeds = (_parse_seeds(args.seeds) if args.seeds is not None
             else [args.seed])
    batch_sizes = (_parse_ints(args.batch_sizes, "--batch-sizes")
                   if args.batch_sizes is not None else [args.num_seqs])
    draft_lengths = (_parse_ints(args.max_draft_lens, "--max-draft-lens")
                     if args.max_draft_lens is not None else [args.max_draft_len])
    spec_modes = [item.strip() for item in args.speculative.split(",") if item.strip()]
    if not spec_modes or "none" not in spec_modes:
        raise ValueError("--speculative must include the none baseline")
    if any(mode not in {"none", "ngram", "medusa", "eagle"} for mode in spec_modes):
        raise ValueError("unsupported speculative mode")
    if "medusa" in spec_modes and not os.path.isfile(args.medusa_path):
        raise FileNotFoundError(f"Medusa weights do not exist: {args.medusa_path}")
    if "eagle" in spec_modes and not os.path.isfile(args.eagle_path):
        raise FileNotFoundError(f"EAGLE weights do not exist: {args.eagle_path}")
    if args.chat_template and not args.prompts_file:
        raise ValueError("--chat-template requires --prompts-file")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,
                                              local_files_only=True)
    trace_prompts = []
    trace_metadata = []
    if args.prompts_file:
        loader_args = SimpleNamespace(
            seed=args.seed,
            prompts_file=args.prompts_file,
            num_seqs=0,
            min_input_len=args.min_input_len,
            max_input_len=args.max_input_len,
            min_output_len=args.max_output_len,
            max_output_len=args.max_output_len,
            shared_prefix_len=0,
        )
        workload = build_workload_data(loader_args, tokenizer)
        trace_prompts, trace_metadata = workload.prompts, workload.requests
        if args.chat_template:
            with open(args.prompts_file, "r", encoding="utf-8") as trace_file:
                request_rows = [json.loads(line) for line in trace_file if line.strip()]
            if len(request_rows) != len(trace_prompts):
                raise ValueError("chat-template input rows do not match loaded trace rows")
            if any(row.get("prompt_prefix_token_ids") for row in request_rows):
                raise ValueError("--chat-template cannot be combined with token-ID prefixes")
            chat_prompts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
                for row in request_rows]
            trace_prompts = [tokenizer.encode(prompt) for prompt in chat_prompts]

    pair_results = []
    flat_rows = []
    condition_index = 0
    styles = [item.strip() for item in args.styles.split(",") if item.strip()]
    output_path = Path(args.output).expanduser()
    planned_pairs = (len(styles) * len(batch_sizes) * len(draft_lengths)
                     * len(seeds) * args.repeats)
    expected_config = {
        "styles": styles,
        "speculative_modes": spec_modes,
        "batch_sizes": batch_sizes,
        "seeds": seeds,
        "repeats": args.repeats,
        "max_draft_lens": draft_lengths,
        "max_output_len": args.max_output_len,
        "ignore_eos": args.ignore_eos,
        "sampling_mode": args.sampling_mode,
        "temperature": args.temperature,
        "fp8_kv": args.fp8,
        "prompts_file": args.prompts_file,
        "chat_template": args.chat_template,
        "max_model_len": args.max_model_len,
    }
    completed_pair_keys = set()
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"cannot resume; checkpoint does not exist: {output_path}")
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get("status") != "running":
            raise ValueError("--resume requires a checkpoint with status='running'")
        if previous.get("meta", {}).get("model") != args.model:
            raise ValueError("--resume model does not match checkpoint")
        if previous.get("config") != expected_config:
            raise ValueError("--resume options do not match checkpoint config")
        pair_results = previous.get("pairs", [])
        completed_pair_keys = {
            (pair["style"], pair["batch_size"], pair["max_draft_len"],
             pair["seed"], pair["repeat_index"])
            for pair in pair_results
        }
        csv_path = output_path.with_suffix(".csv")
        if not csv_path.is_file():
            raise FileNotFoundError(f"cannot resume; CSV checkpoint does not exist: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            saved_rows = list(csv.DictReader(csv_file))
        complete_mode_count = sum(len(pair.get("runs", {})) for pair in pair_results)
        if len(saved_rows) < complete_mode_count:
            raise ValueError("checkpoint CSV contains fewer rows than completed pairs")
        flat_rows = saved_rows[:complete_mode_count]
        print(f"resuming from {len(pair_results)}/{planned_pairs} completed pairs",
              flush=True)

    def checkpoint(status: str, aggregate: dict | None = None) -> None:
        payload = {
            "meta": {
                "model": args.model,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "spec_bench_source": os.path.abspath(__file__),
                **env_info(),
            },
            "config": expected_config,
            "status": status,
            "progress": {
                "completed_pairs": len(pair_results),
                "completed_mode_runs": len(flat_rows),
                "planned_pairs": planned_pairs,
            },
            "aggregate": aggregate or {},
            "pairs": pair_results,
        }
        _save_results(output_path, payload, flat_rows)

    checkpoint("running")
    for style in styles:
        if style not in {"free", "json", "repeat", "sharegpt", "structured",
                         "sharegpt_summary", "open_ended"}:
            raise ValueError(f"unsupported prompt style: {style}")
        if style not in {"free", "json", "repeat"} and not args.prompts_file:
            raise ValueError(f"style {style!r} requires --prompts-file")
        for batch_size in batch_sizes:
            if style in {"free", "json", "repeat"}:
                if batch_size != args.num_seqs and args.batch_sizes is None:
                    raise ValueError("batch-size mismatch")
            for draft_len in draft_lengths:
                args.max_draft_len = draft_len
                for seed in seeds:
                    if style in {"free", "json", "repeat"}:
                        prompts = build_prompts(
                            style, tokenizer, batch_size, args.min_input_len,
                            args.max_input_len, seed)
                        metadata = [{"content_class": style} for _ in prompts]
                    else:
                        prompts, metadata = _trace_prompts(
                            style, trace_prompts, trace_metadata, batch_size, seed)
                    if max(map(len, prompts)) + args.max_output_len > args.max_model_len:
                        raise ValueError(
                            f"prompt + output exceeds --max-model-len for {style}")
                    fingerprint = _prompt_fingerprint(
                        prompts, args.max_output_len, args.ignore_eos)
                    condition_index += 1
                    for repeat_index in range(args.repeats):
                        pair_key = (style, batch_size, draft_len, seed,
                                    repeat_index + 1)
                        if pair_key in completed_pair_keys:
                            continue
                        order = (list(spec_modes) if (condition_index + repeat_index) % 2 == 0
                                 else list(reversed(spec_modes)))
                        runs = {}
                        print(
                            f"\nstyle={style} batch={batch_size} gamma={draft_len} "
                            f"seed={seed} repeat={repeat_index + 1}/{args.repeats} "
                            f"order={' -> '.join(order)} fingerprint={fingerprint[:16]}",
                            flush=True,
                        )
                        for mode_index, speculative in enumerate(order):
                            run = run_case(style, speculative, prompts, tokenizer,
                                           args, seed=seed + repeat_index * 1009,
                                           request_metadata=metadata)
                            run.update({
                                "batch_size": batch_size,
                                "max_draft_len": draft_len,
                                "repeat_index": repeat_index + 1,
                                "input_fingerprint": fingerprint,
                            })
                            runs[speculative] = run
                            flat_rows.append({
                                "style": style,
                                "batch_size": batch_size,
                                "max_draft_len": draft_len,
                                "seed": seed,
                                "repeat_index": repeat_index + 1,
                                "speculative": speculative,
                                "content_class": ",".join(sorted({
                                    row.get("content_class", style)
                                    for row in metadata})),
                                "wall_seconds": run["wall_seconds"],
                                "output_tokens": run["output_tokens"],
                                "mean_output_tokens_per_request": run[
                                    "mean_output_tokens_per_request"],
                                "model_eos_terminated_percent": run[
                                    "model_eos_terminated_percent"],
                                "throughput_tok_per_s": run["throughput_tok_per_s"],
                                "ttft_p50_ms": run["ttft_ms"]["p50"],
                                "ttft_p95_ms": run["ttft_ms"]["p95"],
                                "tpot_p50_ms": run["tpot_ms"]["p50"],
                                "tpot_p95_ms": run["tpot_ms"]["p95"],
                                "e2e_p95_ms": run["e2e_ms"]["p95"],
                                "alpha": run["alpha"],
                                "spec_proposed_tokens": run["spec_proposed_tokens"],
                                "spec_accepted_tokens": run["spec_accepted_tokens"],
                                "input_fingerprint": fingerprint,
                                "output_token_fingerprint": run[
                                    "output_token_fingerprint"],
                            })
                            print(
                                f"  {speculative:<7} {run['throughput_tok_per_s']:.1f} tok/s | "
                                f"TTFT p50/p95 {_format_ms(run['ttft_ms']['p50'])}/"
                                f"{_format_ms(run['ttft_ms']['p95'])} ms | "
                                f"TPOT p50/p95 {_format_ms(run['tpot_ms']['p50'])}/"
                                f"{_format_ms(run['tpot_ms']['p95'])} ms | alpha={run['alpha']}",
                                flush=True,
                            )
                            if mode_index == 0:
                                checkpoint("running")
                        if len({run["input_fingerprint"] for run in runs.values()}) != 1:
                            raise AssertionError("paired modes received different prompt inputs")
                        baseline = runs["none"]
                        comparisons = {}
                        for speculative in spec_modes:
                            if speculative == "none":
                                continue
                            measured = runs[speculative]
                            speedup = 100.0 * (
                                measured["throughput_tok_per_s"]
                                / max(baseline["throughput_tok_per_s"], 1e-12) - 1.0)
                            comparisons[speculative] = {
                                "throughput_delta_percent": speedup,
                                "ttft_p50_delta_ms": (
                                    measured["ttft_ms"]["p50"]
                                    - baseline["ttft_ms"]["p50"]),
                                "tpot_p50_delta_ms": (
                                    measured["tpot_ms"]["p50"]
                                    - baseline["tpot_ms"]["p50"]),
                                "output_token_match": (
                                    measured["output_token_fingerprint"]
                                    == baseline["output_token_fingerprint"]),
                            }
                        pair_results.append({
                            "style": style,
                            "batch_size": batch_size,
                            "max_draft_len": draft_len,
                            "seed": seed,
                            "repeat_index": repeat_index + 1,
                            "execution_order": order,
                            "input_fingerprint": fingerprint,
                            "content_class_counts": dict(Counter(
                                row.get("content_class") or style
                                for row in metadata)),
                            "comparisons": comparisons,
                            "runs": runs,
                        })
                        checkpoint("running")

    grouped = {}
    for pair in pair_results:
        for speculative, comparison in pair["comparisons"].items():
            key = (pair["style"], pair["batch_size"],
                   pair["max_draft_len"], speculative)
            grouped.setdefault(key, []).append(comparison["throughput_delta_percent"])
    aggregate = {
        f"{style}_batch{batch}_gamma{gamma}_{speculative}": {
            "style": style,
            "batch_size": batch,
            "max_draft_len": gamma,
            "speculative": speculative,
            **_bootstrap_median_ci(values, args.seed + batch + gamma),
            "wins": sum(value > 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        for (style, batch, gamma, speculative), values in grouped.items()
    }
    checkpoint("complete", aggregate)
    csv_path = output_path.with_suffix(".csv")
    print(f"\nJSON -> {output_path}, CSV -> {csv_path}")


if __name__ == "__main__":
    main()
