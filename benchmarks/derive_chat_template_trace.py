"""Derive a chat-formatted prompt trace for natural-stop serving ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from transformers import AutoTokenizer

try:
    from .bench import prompt_token_hash
except ImportError:
    from bench import prompt_token_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="local model/tokenizer directory")
    parser.add_argument("--input", required=True,
                        help="prompt-only or request-trace JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=80.0)
    args = parser.parse_args()
    if args.num_requests < 1 or args.max_tokens < 1:
        parser.error("--num-requests and --max-tokens must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    source = Path(args.input).expanduser()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if args.num_requests > len(rows):
        parser.error(f"requested {args.num_requests} requests but source has {len(rows)}")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
           for row in rows):
        raise ValueError("every source row must contain a non-empty prompt")
    if any(row.get("prompt_prefix_token_ids") for row in rows):
        raise ValueError("chat-template derivation does not support token-ID prefixes")
    request_ids = [str(row.get("request_id", f"request-{index:06d}"))
                   for index, row in enumerate(rows)]
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("source request IDs must be unique")

    selected_indices = sorted(random.Random(args.seed).sample(
        range(len(rows)), args.num_requests))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True)
    if not tokenizer.chat_template:
        raise ValueError("tokenizer does not provide a chat template")

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_ids = []
    with output.open("w", encoding="utf-8", newline="\n") as trace:
        for index in selected_indices:
            source_row = rows[index]
            request_id = request_ids[index]
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": source_row["prompt"]}],
                tokenize=False, add_generation_prompt=True)
            token_ids = tokenizer.encode(chat_prompt)
            row = {
                "request_id": request_id,
                "prompt": chat_prompt,
                "prompt_token_hash": prompt_token_hash(token_ids),
                "prompt_tokens": len(token_ids),
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "ignore_eos": False,
                "ttft_slo_ms": args.ttft_slo_ms,
                "tpot_slo_ms": args.tpot_slo_ms,
                "content_class": source_row.get("content_class", "sharegpt"),
            }
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")
            selected_ids.append(request_id)

    chat_template = tokenizer.chat_template
    manifest = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "trace": str(output),
        "trace_sha256": _sha256(output),
        "model": args.model,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "num_requests": len(selected_ids),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": False,
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "chat_template_applied": True,
        "arrival_times_in_trace": False,
        "arrival_note": "The serving benchmark generates synthetic Poisson arrivals.",
        "request_ids": selected_ids,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
