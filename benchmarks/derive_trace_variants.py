"""Derive prefix-cache and smoke traces from one tokenized request trace.

This keeps prompts, output budgets, and arrival times identical across the
prefix-percentage conditions while changing only which requests share a common
tokenized system prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .bench import prompt_token_hash
    from .sharegpt_trace import _prefix_tokens
except ImportError:
    from bench import prompt_token_hash
    from sharegpt_trace import _prefix_tokens


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("input must be a non-empty JSONL object trace")
    return rows


def _write_trace(rows: list[dict[str, Any]], output: Path,
                 base_manifest: dict[str, Any], *, derived_from: str,
                 prefix_percent: int | None = None,
                 prefix_tokens: int = 0) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = json.loads(json.dumps(base_manifest))
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["derived_from"] = derived_from
    manifest["trace"] = {
        "path": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "requests": len(rows),
    }
    if prefix_percent is not None:
        manifest.setdefault("workload", {}).update({
            "shared_prefix_percent": prefix_percent,
            "shared_prefix_tokens": prefix_tokens if prefix_percent else 0,
            "shared_prefix_kind": (
                "synthetic_common_system_template" if prefix_percent else None),
        })
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False)


def _parse_percentages(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("percentages must be integers") from exc
    if not values or any(value not in (0, 20, 50) for value in values):
        raise argparse.ArgumentTypeError("percentages must be selected from 0,20,50")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("percentages must not contain duplicates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="base request-trace JSONL")
    parser.add_argument("--model", required=True, help="local tokenizer directory")
    parser.add_argument("--output-dir", default="results/supplemental")
    parser.add_argument("--prefix-percentages", type=_parse_percentages,
                        default=[20, 50])
    parser.add_argument("--prefix-tokens", type=int, choices=(256, 512), default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-count", type=int, default=100)
    parser.add_argument("--calibration-count", type=int, default=0,
                        help="optional disjoint calibration/held-out split size")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser()
    if args.smoke_count < 1:
        raise SystemExit("--smoke-count must be positive")
    rows = _read_jsonl(source)
    if args.smoke_count > len(rows):
        raise SystemExit(
            f"smoke count {args.smoke_count} exceeds {len(rows)} source requests")
    if args.calibration_count < 0 or args.calibration_count >= len(rows):
        if args.calibration_count:
            raise SystemExit("--calibration-count must be smaller than the trace size")
    if any(row.get("prompt_prefix_token_ids") for row in rows):
        raise SystemExit("input trace must not already contain a synthetic prefix")
    request_ids = [row.get("request_id") for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise SystemExit("input trace contains duplicate request IDs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True)
    from_input = [list(tokenizer.encode(row["prompt"])) for row in rows]
    for index, (row, token_ids) in enumerate(zip(rows, from_input)):
        if row.get("prompt_token_hash") != prompt_token_hash(token_ids):
            raise ValueError(
                f"source prompt hash mismatch on row {index}; tokenizer differs")
    prefix_ids = _prefix_tokens(tokenizer, args.prefix_tokens)
    source_manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    base_manifest = (json.loads(source_manifest_path.read_text(encoding="utf-8"))
                     if source_manifest_path.is_file() else {})
    source_workload = base_manifest.get("workload", {})
    for row in rows:
        row.setdefault("arrival_mode", source_workload.get("arrival_mode"))
        row.setdefault(
            "arrival_rate_requests_per_second",
            source_workload.get("arrival_rate_requests_per_second"))
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    smoke_order = random.Random(args.seed).sample(
        range(args.smoke_count), args.smoke_count)
    remaining_indexes = range(args.smoke_count, len(rows))
    remaining_order = random.Random(args.seed + 1).sample(
        remaining_indexes, len(rows) - args.smoke_count)

    smoke = rows[:args.smoke_count]
    smoke_output = output_dir / f"{source.stem}_smoke{args.smoke_count}.jsonl"
    _write_trace(smoke, smoke_output, base_manifest,
                 derived_from=source_sha256)
    print(f"Wrote smoke trace {smoke_output} ({len(smoke)} requests)")

    if args.calibration_count:
        calibration_indexes = set(random.Random(args.seed + 1).sample(
            range(len(rows)), args.calibration_count))
        calibration_rows = [
            {"request_id": row.get("request_id"), "prompt": row["prompt"]}
            for index, row in enumerate(rows) if index in calibration_indexes]
        heldout_rows = [
            {"request_id": row.get("request_id"), "prompt": row["prompt"]}
            for index, row in enumerate(rows) if index not in calibration_indexes]
        calibration_output = output_dir / (
            f"{source.stem}_calibration{len(calibration_rows)}.jsonl")
        heldout_output = output_dir / (
            f"{source.stem}_heldout{len(heldout_rows)}.jsonl")
        _write_trace(calibration_rows, calibration_output, base_manifest,
                     derived_from=source_sha256)
        _write_trace(heldout_rows, heldout_output, base_manifest,
                     derived_from=source_sha256)
        print(f"Wrote disjoint calibration/eval prompts: "
              f"{len(calibration_rows)}/{len(heldout_rows)}")

    for percentage in args.prefix_percentages:
        prefix_count = math.floor(len(rows) * percentage / 100 + 0.5)
        smoke_prefix_count = math.floor(
            args.smoke_count * percentage / 100 + 0.5)
        remaining_prefix_count = prefix_count - smoke_prefix_count
        selected = (set(smoke_order[:smoke_prefix_count])
                    | set(remaining_order[:remaining_prefix_count]))
        variant_rows = []
        for index, (row, body_tokens) in enumerate(zip(rows, from_input)):
            updated = dict(row)
            uses_prefix = index in selected
            request_prefix = prefix_ids if uses_prefix else []
            full_tokens = request_prefix + body_tokens
            updated.update({
                "prompt_prefix_token_ids": request_prefix,
                "prefix_group": (f"synthetic_system_template_{args.prefix_tokens}"
                                 if uses_prefix else None),
                "prefix_kind": ("synthetic_common_system_template"
                                if uses_prefix else None),
                "prefix_tokens": len(request_prefix),
                "prompt_tokens": len(full_tokens),
                "prompt_token_hash": prompt_token_hash(full_tokens),
            })
            variant_rows.append(updated)
        output = output_dir / (
            f"{source.stem}_prefix{percentage}_{args.prefix_tokens}.jsonl")
        _write_trace(variant_rows, output, base_manifest,
                     derived_from=source_sha256,
                     prefix_percent=percentage,
                     prefix_tokens=args.prefix_tokens)
        print(f"Wrote {percentage}% prefix trace {output} "
              f"({prefix_count}/{len(rows)} requests share the prefix)")
        smoke_variant_output = output_dir / (
            f"{source.stem}_prefix{percentage}_{args.prefix_tokens}_"
            f"smoke{args.smoke_count}.jsonl")
        _write_trace(variant_rows[:args.smoke_count], smoke_variant_output,
                     base_manifest, derived_from=source_sha256,
                     prefix_percent=percentage,
                     prefix_tokens=args.prefix_tokens)
        print(f"Wrote matched smoke prefix trace {smoke_variant_output} "
              f"({smoke_prefix_count}/{args.smoke_count} requests share the prefix)")


if __name__ == "__main__":
    main()
