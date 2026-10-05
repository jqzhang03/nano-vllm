"""Build a real-text FP8 KV calibration set and measure FP8 cache logit drift.

By default this script writes a token-ID calibration file, then compares an
auto-dtype KV baseline against each requested FP8 scale margin. It evaluates
the first pure decode step, where both runs have the same prompt and generated
prefix, so the comparison measures reading quantized KV cache values.

Examples (WSL inference environment)::

    python benchmarks/kv_fp8_calibrate.py --model ~/huggingface/Qwen3-0.6B
    python benchmarks/kv_fp8_calibrate.py --model /mnt/d/models/Qwen3-0.6B \
        --calibration-file calibration.jsonl --eval-file heldout.jsonl \
        --margins 0.9,1.0,1.1,1.25

Input text files may be plain text or JSONL with a prompt or text field.
The calibration JSON is accepted directly by --kv-calibration-path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
import statistics
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams

try:
    from .bench import env_info
except ImportError:  # direct script execution from repository root
    from bench import env_info


DEFAULT_CALIBRATION_TEXTS = [
    "A language model processes a prompt by converting text into token IDs, then building key and value tensors for every attention layer.",
    "Long-context inference increases memory traffic because each generated token reads the keys and values from all earlier positions.",
    "The GPU stores model weights, temporary activations, and the KV cache in device memory; the cache capacity depends on model architecture and dtype.",
    "FP8 E4M3 uses one byte per value. A calibration pass estimates activation ranges and chooses a scale before values are stored in the cache.",
    "A robust evaluation set should contain held-out natural language, code, numbers, and prompts with different lengths.",
    "def fibonacci(n):\n    a, b = 0, 1\n    values = []\n    for _ in range(n):\n        values.append(a)\n        a, b = b, a + b\n    return values",
    "In 2024, the project measured throughput at batch sizes 1, 4, 8, and 16, recording median latency and the 99th percentile.",
    "When quantization scales are too small, values saturate at the representable maximum; when they are too large, more precision is lost near zero.",
]
DEFAULT_EVAL_TEXTS = [
    "Explain why a paged key-value cache can improve concurrent text generation.",
    "Write a Python function that returns the first n square numbers.",
    "A 16 GiB GPU has a fixed memory budget. Describe how model weights and the KV cache compete for it.",
    "The experiment uses 32 prompts, each with a 2048-token context and 64 generated tokens.",
    "Summarize the tradeoffs between cache capacity, numerical precision, and decode speed.",
    "What is the difference between prefill latency and the time per generated token?",
    "A calibration set contains ordinary prose, code, identifiers, dates, and numeric values.",
    "Given a measured maximum activation magnitude, an FP8 scale maps it near the E4M3 range limit.",
]


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _float_list(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("margins must be comma-separated numbers") from exc
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError("margins must be finite positive numbers")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("margins must not contain duplicates")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B"))
    parser.add_argument("--calibration-file", default=None,
                        help="plain text or JSONL input; built-in texts are used if omitted")
    parser.add_argument("--eval-file", default=None,
                        help="held-out plain text or JSONL input; built-in texts if omitted")
    parser.add_argument("--max-calibration-prompts", type=_positive_int, default=32)
    parser.add_argument("--max-eval-prompts", type=_positive_int, default=8)
    parser.add_argument("--max-calibration-prompt-tokens", type=_positive_int, default=2048)
    parser.add_argument("--max-eval-prompt-tokens", type=_positive_int, default=2048)
    parser.add_argument("--calibration-context-length", type=int, default=0,
                        help="repeat short prompts to this length; 0 preserves corpus lengths")
    parser.add_argument("--eval-context-length", type=int, default=0,
                        help="repeat short prompts to this length; 0 preserves corpus lengths")
    parser.add_argument("--max-output-tokens", type=_positive_int, default=2,
                        help="2 is sufficient to capture the first pure decode logits")
    parser.add_argument("--max-num-batched-tokens", type=_positive_int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--margins", type=_float_list, default=[0.9, 1.0, 1.1, 1.25],
                        help="FP8 scale margin values to evaluate")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", default=None,
                        help="JSON results path (calibration token JSON and CSV use same stem)")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="write the token-ID calibration file without loading a model on CUDA")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.calibration_context_length < 0 or args.eval_context_length < 0:
        parser.error("context lengths must be non-negative")
    if args.calibration_context_length > args.max_num_batched_tokens:
        parser.error("--calibration-context-length cannot exceed --max-num-batched-tokens")
    if args.max_output_tokens < 2:
        parser.error("--max-output-tokens must be at least 2 to capture a decode logits row")
    return args


def load_texts(path: str | None, defaults: list[str], limit: int) -> tuple[list[str], str]:
    if path is None:
        return defaults[:limit], "built-in"
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"input corpus not found: {path}")
    if path.lower().endswith(".jsonl"):
        texts = []
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get("prompt", item.get("text"))
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{path}:{line_number} needs a non-empty prompt or text string")
                texts.append(text)
                if len(texts) >= limit:
                    break
    else:
        with open(path, encoding="utf-8") as source:
            contents = source.read()
        texts = [part.strip() for part in contents.split("\n\n") if part.strip()]
        if len(texts) == 1 and "\n" in texts[0]:
            texts = [line.strip() for line in texts[0].splitlines() if line.strip()]
        texts = texts[:limit]
    if not texts:
        raise ValueError(f"no usable text records found in {path}")
    return texts, path


def tokenize_texts(tokenizer, texts: list[str], max_tokens: int) -> list[list[int]]:
    result = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        if token_ids:
            result.append(token_ids)
    if not result:
        raise ValueError("tokenization produced no non-empty prompts")
    return result


def extend_prompts(prompts: list[list[int]], target_length: int) -> list[list[int]]:
    """Repeat each short prompt to a target length when explicitly requested."""
    if target_length <= 0:
        return prompts
    extended = []
    for prompt in prompts:
        if len(prompt) >= target_length:
            extended.append(prompt[:target_length])
            continue
        repeats = (target_length + len(prompt) - 1) // len(prompt)
        extended.append((prompt * repeats)[:target_length])
    return extended


def _percentile_summary(values: list[float]) -> dict[str, float | int | None]:
    vals = sorted(values)
    if not vals:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {"min": vals[0], "mean": statistics.fmean(vals),
            "median": statistics.median(vals), "max": vals[-1]}


def extract_fp8_scales(engine: LLM, *, observed: bool = False) -> list[dict[str, Any]]:
    rows = []
    range_prefix = "evaluation" if observed else "calibration"
    for name, module in engine.model_runner.model.named_modules():
        if not hasattr(module, "cal_max_k") and not hasattr(module, "cal_c_max"):
            continue
        row: dict[str, Any] = {"module": name}
        if hasattr(module, "cal_max_k"):
            row.update({f"{range_prefix}_max_k": module.cal_max_k,
                        f"{range_prefix}_max_v": module.cal_max_v})
            if not observed:
                row.update({"k_scale": module.k_scale, "v_scale": module.v_scale})
        if hasattr(module, "cal_c_max"):
            row.update({f"{range_prefix}_max_c": module.cal_c_max,
                        f"{range_prefix}_max_rope": module.cal_r_max})
            if not observed:
                row.update({"c_scale": module.mla_c_scale,
                            "rope_scale": module.mla_r_scale})
        if len(row) > 1:
            rows.append(row)
    return rows


def observe_fp8_ranges(engine: LLM, enabled: bool) -> None:
    """Reuse attention's range collectors without changing the installed scales."""
    for module in engine.model_runner.model.modules():
        if not hasattr(module, "calibrating"):
            continue
        if enabled:
            for name in ("cal_max_k", "cal_max_v", "cal_c_max", "cal_r_max"):
                if hasattr(module, name):
                    setattr(module, name, 0.0)
        module.calibrating = enabled


def range_utilization(calibration_scales: list[dict[str, Any]],
                      observed_ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Observed held-out maxima divided by each scale's E4M3 representable limit."""
    observed_by_name = {row["module"]: row for row in observed_ranges}
    result = []
    for scales in calibration_scales:
        observed = observed_by_name.get(scales["module"])
        if observed is None:
            continue
        row: dict[str, Any] = {"module": scales["module"]}
        pairs = (("calibration_max_k", "k_scale", "evaluation_max_k", "k"),
                 ("calibration_max_v", "v_scale", "evaluation_max_v", "v"),
                 ("calibration_max_c", "c_scale", "evaluation_max_c", "c"),
                 ("calibration_max_rope", "rope_scale", "evaluation_max_rope", "rope"))
        for cal_key, scale_key, observed_key, output_key in pairs:
            if cal_key in scales and observed_key in observed:
                row[f"{output_key}_limit_utilization"] = (
                    observed[observed_key] / max(1e-30, 448.0 * scales[scale_key]))
                row[f"{output_key}_evaluation_max"] = observed[observed_key]
        result.append(row)
    return result


def _first_decode_logits(engine: LLM) -> torch.Tensor:
    for kind, logits in engine.collected_logits:
        if kind == "decode" and logits is not None:
            return logits.detach().float().cpu()
    raise RuntimeError(
        "did not collect a pure decode logits row; use --max-output-tokens >= 2 "
        "and run one evaluation prompt at a time")


def run_variant(args: argparse.Namespace, kv_dtype: str, margin: float | None,
                calibration_path: str, eval_prompts: list[list[int]],
                max_model_len: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_model_len": max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "kv_cache_dtype": kv_dtype,
    }
    if kv_dtype == "fp8_e4m3":
        kwargs["kv_calibration_path"] = calibration_path
        kwargs["kv_fp8_scale_margin"] = margin
    engine = LLM(args.model, **kwargs)
    try:
        calibration_scales = (extract_fp8_scales(engine)
                              if kv_dtype == "fp8_e4m3" else [])
        if kv_dtype == "fp8_e4m3":
            observe_fp8_ranges(engine, True)
        prompt_rows = []
        try:
            for index, prompt in enumerate(eval_prompts):
                torch.manual_seed(args.seed + index)
                torch.cuda.manual_seed_all(args.seed + index)
                output = engine.generate(
                    [prompt],
                    SamplingParams(temperature=0.1, ignore_eos=True,
                                   max_tokens=args.max_output_tokens),
                    use_tqdm=False,
                    collect_logits=True,
                    collect_decode_logits_only=True,
                )[0]
                prompt_rows.append({
                    "index": index,
                    "prompt_tokens": len(prompt),
                    "first_generated_token": output["token_ids"][0]
                    if output["token_ids"] else None,
                    "generated_tokens": len(output["token_ids"]),
                    "first_decode_logits": _first_decode_logits(engine),
                })
        finally:
            if kv_dtype == "fp8_e4m3":
                observe_fp8_ranges(engine, False)
        observed_ranges = (extract_fp8_scales(engine, observed=True)
                           if kv_dtype == "fp8_e4m3" else [])
        utilization = range_utilization(calibration_scales, observed_ranges)
        return {"kv_cache_dtype": kv_dtype, "scale_margin": margin,
                "prompts": prompt_rows, "fp8_layer_scales": calibration_scales,
                "evaluation_activation_ranges": observed_ranges,
                "evaluation_range_utilization": utilization}
    finally:
        engine.exit()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


def _score_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    a = reference.float()
    b = candidate.float()
    diff = a - b
    logp_a = torch.log_softmax(a, dim=-1)
    logp_b = torch.log_softmax(b, dim=-1)
    p_a = logp_a.exp()
    kl = (p_a * (logp_a - logp_b)).sum(dim=-1).mean().clamp_min(0.0)
    top1 = (a.argmax(dim=-1) == b.argmax(dim=-1)).float().mean()
    topk_a = a.topk(min(5, a.shape[-1]), dim=-1).indices
    topk_b = b.topk(min(5, b.shape[-1]), dim=-1).indices
    overlaps = [torch.isin(x, y).float().mean() for x, y in zip(topk_a, topk_b)]
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()
    return {
        "logits_max_abs_error": diff.abs().max().item(),
        "logits_mean_abs_error": diff.abs().mean().item(),
        "logits_rmse": diff.square().mean().sqrt().item(),
        "logits_cosine_similarity": cosine.item(),
        "kl_ref_to_fp8": kl.item(),
        "top1_agreement_percent": top1.item() * 100.0,
        "top5_overlap_percent": torch.stack(overlaps).mean().item() * 100.0,
    }


def compare_logits(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ref_rows = reference["prompts"]
    test_rows = candidate["prompts"]
    if len(ref_rows) != len(test_rows):
        raise ValueError("baseline and FP8 evaluation prompt counts differ")
    aligned = []
    per_prompt = []
    for ref, test in zip(ref_rows, test_rows):
        same_first = ref["first_generated_token"] == test["first_generated_token"]
        row = {"index": ref["index"], "prompt_tokens": ref["prompt_tokens"],
               "first_token_agreement": same_first}
        if same_first:
            a = ref["first_decode_logits"].float()
            b = test["first_decode_logits"].float()
            if a.shape != b.shape:
                raise ValueError(f"logit shape differs: {tuple(a.shape)} vs {tuple(b.shape)}")
            row.update(_score_logits(a, b))
            aligned.append((a, b))
        per_prompt.append(row)
    if aligned:
        ref_logits = torch.cat([pair[0] for pair in aligned], dim=0)
        test_logits = torch.cat([pair[1] for pair in aligned], dim=0)
        aggregate = _score_logits(ref_logits, test_logits)
    else:
        aggregate = {key: None for key in (
            "logits_max_abs_error", "logits_mean_abs_error", "logits_rmse",
            "logits_cosine_similarity", "kl_ref_to_fp8",
            "top1_agreement_percent", "top5_overlap_percent")}
    return {
        "evaluation_prompts": len(ref_rows),
        "aligned_first_token_prompts": len(aligned),
        "first_token_agreement_percent": (
            100.0 * sum(row["first_token_agreement"] for row in per_prompt)
            / max(1, len(per_prompt))),
        "metrics": aggregate,
        "per_prompt": per_prompt,
    }


def _scale_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = sorted({key for row in rows for key in row if key != "module"})
    summary = {}
    for field in fields:
        values = [float(row[field]) for row in rows if field in row]
        summary[field] = _percentile_summary(values)
    return summary


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    if not os.path.isdir(args.model):
        raise SystemExit(f"model directory does not exist: {args.model}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or os.path.join(
        "results", f"kv_fp8_calibration_{timestamp}.json")
    output = os.path.expanduser(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    stem, _ = os.path.splitext(output)
    calibration_path = stem + ".tokens.json"
    summary_csv = stem + ".csv"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    calibration_texts, calibration_source = load_texts(
        args.calibration_file, DEFAULT_CALIBRATION_TEXTS, args.max_calibration_prompts)
    eval_texts, eval_source = load_texts(
        args.eval_file, DEFAULT_EVAL_TEXTS, args.max_eval_prompts)
    calibration_token_limit = min(args.max_calibration_prompt_tokens,
                                   args.max_num_batched_tokens)
    calibration_prompts = tokenize_texts(
        tokenizer, calibration_texts, calibration_token_limit)
    eval_prompts = tokenize_texts(tokenizer, eval_texts, args.max_eval_prompt_tokens)
    calibration_context_length = args.calibration_context_length
    eval_context_length = args.eval_context_length
    # Make the no-argument example exercise non-trivial KV history, while
    # leaving user-provided corpora at their real lengths unless asked to pad.
    if args.calibration_file is None and calibration_context_length == 0:
        calibration_context_length = min(1024, args.max_num_batched_tokens)
    if args.eval_file is None and eval_context_length == 0:
        eval_context_length = 1024
    calibration_prompts = extend_prompts(calibration_prompts, calibration_context_length)
    eval_prompts = extend_prompts(eval_prompts, eval_context_length)

    hf = AutoConfig.from_pretrained(args.model)
    if hf.model_type == "gemma2":
        raise SystemExit("Gemma-2 logit-softcap attention is not compatible with FP8 KV")
    max_position = int(getattr(hf, "max_position_embeddings", 4096))
    longest_eval = max(len(prompt) for prompt in eval_prompts)
    longest_calibration = max(len(prompt) for prompt in calibration_prompts)
    max_model_len = max(longest_calibration, longest_eval + args.max_output_tokens)
    if max_model_len > max_position:
        raise SystemExit(
            f"calibration/evaluation needs max_model_len={max_model_len}, but model limit "
            f"is {max_position}; lower prompt caps or output length")

    calibration_payload = {
        "model": args.model,
        "model_type": hf.model_type,
        "source": calibration_source,
        "token_ids": calibration_prompts,
        "prompt_token_counts": [len(prompt) for prompt in calibration_prompts],
        "total_calibration_tokens": sum(map(len, calibration_prompts)),
        "max_model_len": max_model_len,
    }
    with open(calibration_path, "w", encoding="utf-8") as calibration_file:
        json.dump(calibration_payload, calibration_file, indent=2)

    result: dict[str, Any] = {
        "schema_version": 1,
        "meta": {"model": args.model, "model_type": hf.model_type, "date": timestamp},
        "calibration": {
            "source": calibration_source,
            "prompts": len(calibration_prompts),
            "total_tokens": sum(map(len, calibration_prompts)),
            "prompt_lengths": _percentile_summary(list(map(len, calibration_prompts))),
            "token_ids_path": calibration_path,
        },
        "evaluation": {
            "source": eval_source,
            "prompts": len(eval_prompts),
            "total_tokens": sum(map(len, eval_prompts)),
            "prompt_lengths": _percentile_summary(list(map(len, eval_prompts))),
        },
        "config": {
            "margins": args.margins,
            "max_output_tokens": args.max_output_tokens,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_calibration_prompt_tokens_effective": calibration_token_limit,
            "calibration_context_length": calibration_context_length,
            "eval_context_length": eval_context_length,
            "seed": args.seed,
            "eager": args.enforce_eager,
        },
        "runs": {},
        "warnings": [
            "FP8 scores compare first decode logits only when the first generated token matches; unaligned prompts are excluded from cache-logit metrics.",
            "A calibration maximum is data-dependent. Include representative production prompts and verify on a separate held-out corpus.",
            "Range utilization compares held-out activation maxima with the calibrated E4M3 limit; values above 1 indicate an observed value will be clamped, but do not give the elementwise saturation rate.",
        ],
    }
    if not args.calibrate_only:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for FP8 KV accuracy calibration")
        result["meta"].update(env_info())
        baseline = run_variant(args, "auto", None, calibration_path,
                               eval_prompts, max_model_len)
        result["runs"]["auto_baseline"] = baseline
        for margin in args.margins:
            key = f"fp8_e4m3_margin_{margin:g}"
            print(f"calibrating FP8 KV with margin={margin:g} ...", flush=True)
            candidate = run_variant(args, "fp8_e4m3", margin, calibration_path,
                                    eval_prompts, max_model_len)
            comparison = compare_logits(baseline, candidate)
            for prompt in candidate["prompts"]:
                prompt.pop("first_decode_logits", None)
            candidate["comparison_to_auto"] = comparison
            candidate["scale_summary"] = _scale_summary(candidate["fp8_layer_scales"])
            utilization_values = [value for row in candidate["evaluation_range_utilization"]
                                  for key, value in row.items()
                                  if key.endswith("_limit_utilization")]
            candidate["evaluation_range_summary"] = {
                "max_limit_utilization": max(utilization_values, default=None),
                "mean_limit_utilization": (statistics.fmean(utilization_values)
                                           if utilization_values else None),
                "components_over_limit": sum(value > 1.0 for value in utilization_values),
                "measured_layer_components": len(utilization_values),
            }
            result["runs"][key] = candidate
            metrics = comparison["metrics"]
            kl = metrics["kl_ref_to_fp8"]
            top1 = metrics["top1_agreement_percent"]
            mean_error = metrics["logits_mean_abs_error"]
            max_utilization = candidate["evaluation_range_summary"][
                "max_limit_utilization"]
            print(f"  aligned {comparison['aligned_first_token_prompts']}/"
                  f"{comparison['evaluation_prompts']} | "
                  f"KL {kl if kl is not None else float('nan'):.6g} | "
                  f"top-1 {top1 if top1 is not None else float('nan'):.1f}% | "
                  f"max range {max_utilization if max_utilization is not None else float('nan'):.3f}x | "
                  f"mean |delta logit| "
                  f"{mean_error if mean_error is not None else float('nan'):.6g}",
                  flush=True)

    # Logits are retained in-memory only for pairwise comparison. The JSON keeps
    # aggregate and per-prompt error metrics rather than huge vocabulary vectors.
    for run in result["runs"].values():
        for prompt in run["prompts"]:
            prompt.pop("first_decode_logits", None)
    with open(output, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, default=str)
    csv_rows = []
    for name, run in result["runs"].items():
        comparison = run.get("comparison_to_auto", {})
        row = {"variant": name, "kv_cache_dtype": run["kv_cache_dtype"],
               "scale_margin": run["scale_margin"]}
        row.update(comparison.get("metrics", {}))
        row["aligned_first_token_prompts"] = comparison.get("aligned_first_token_prompts")
        row["first_token_agreement_percent"] = comparison.get(
            "first_token_agreement_percent")
        range_summary = run.get("evaluation_range_summary", {})
        row["max_evaluation_range_utilization"] = range_summary.get(
            "max_limit_utilization")
        row["mean_evaluation_range_utilization"] = range_summary.get(
            "mean_limit_utilization")
        row["components_over_e4m3_limit"] = range_summary.get("components_over_limit")
        csv_rows.append(row)
    if csv_rows:
        import csv
        with open(summary_csv, "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    print(f"Calibration token IDs: {calibration_path}")
    print(f"JSON report:           {output}")
    if csv_rows:
        print(f"CSV summary:           {summary_csv}")
    elif args.calibrate_only:
        print("Use the token file with --kv-calibration-path for a later FP8 run.")


if __name__ == "__main__":
    main()
