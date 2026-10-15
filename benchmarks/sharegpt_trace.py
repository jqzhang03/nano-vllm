"""Build a de-identified, reproducible request trace from ShareGPT JSON data.

The output keeps user prompts only. Assistant replies are token-counted to set
an output budget and are never written to the trace. The parser accepts the
ShareGPT JSON array format and JSONL records with ``prompt`` or
``conversations``/``messages`` fields.

Example::

    python benchmarks/sharegpt_trace.py \
        --input ~/.cache/huggingface/sharegpt.json \
        --model ~/huggingface/Qwen3-0.6B --num-samples 500 \
        --arrival-mode poisson --arrival-rate 8 --seed 7 \
        --output results/supplemental/sharegpt_500_poisson8_seed7.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


LENGTH_BUCKETS = (
    (1, 256, "1-256"),
    (257, 512, "257-512"),
    (513, 1024, "513-1024"),
    (1025, 2048, "1025-2048"),
)
TRACE_FORMAT = "nano-vllm-request-trace-v1"

_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d(?!\w)")
_CODE_RE = re.compile(
    r"\b(json|sql|python|javascript|typescript|code|program|function|script|"
    r"class|query|regex|html|css|yaml|xml|api endpoint)\b|```|\{\s*\"",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"\b(summar(?:y|ize|ise)|recap|transcript|conversation|meeting notes|"
    r"bullet points|key points)\b",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\b(why|how|explain|reason|reasoning|compare|evaluate|analy[sz]e|"
    r"what would happen|what are the|can you tell me)\b|\?",
    re.IGNORECASE,
)


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Stream objects from a large top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    with path.open("r", encoding="utf-8") as source:
        while True:
            chunk = source.read(1 << 20)
            buffer += chunk
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position == len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("expected a top-level JSON array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and buffer[position] == ",":
                    position += 1
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                if position == len(buffer):
                    break
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                if isinstance(item, dict):
                    yield item
                position = end
                if position >= 1 << 20:
                    buffer = buffer[position:]
                    position = 0
            if not chunk:
                if not started:
                    raise ValueError("input file is empty")
                if position >= len(buffer):
                    return
                raise ValueError("incomplete JSON array at end of input")


def iter_source_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        first = ""
        while not first:
            char = source.read(1)
            if not char:
                raise ValueError("input file is empty")
            if not char.isspace():
                first = char
        if first == "[":
            yield from _iter_json_array(path)
            return
        source.seek(0)
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(item, dict):
                yield item


def _role(message: dict[str, Any]) -> str:
    value = message.get("from", message.get("role", message.get("speaker", "")))
    normalized = str(value).strip().lower()
    if normalized in {"human", "user", "prompter"}:
        return "user"
    if normalized in {"gpt", "assistant", "bot", "model"}:
        return "assistant"
    return normalized


def _conversation_text(record: dict[str, Any]) -> tuple[str | None, str | None]:
    direct_prompt = record.get("prompt")
    direct_reply = record.get("response", record.get("completion"))
    if isinstance(direct_prompt, str):
        return direct_prompt, direct_reply if isinstance(direct_reply, str) else None
    messages = record.get("conversations", record.get("messages"))
    if not isinstance(messages, list):
        return None, None
    first_user = None
    first_reply = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = message.get("value", message.get("content", message.get("text")))
        if not isinstance(text, str) or not text.strip():
            continue
        role = _role(message)
        if first_user is None and role == "user":
            first_user = text
        elif first_user is not None and role == "assistant":
            first_reply = text
            break
    return first_user, first_reply


def _redact_pii(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()

    def replace(pattern: re.Pattern, label: str, value: str) -> None:
        nonlocal text
        text, count = pattern.subn(value, text)
        counts[label] += count

    replace(_EMAIL_RE, "email", "[REDACTED_EMAIL]")
    replace(_SSN_RE, "ssn", "[REDACTED_ID]")
    replace(_IP_RE, "ip_address", "[REDACTED_IP]")

    def phone_replacement(match: re.Match) -> str:
        digits = sum(character.isdigit() for character in match.group(0))
        if digits >= 10:
            counts["phone"] += 1
            return "[REDACTED_PHONE]"
        return match.group(0)

    text = _PHONE_RE.sub(phone_replacement, text)
    return text, counts


def _clean_text(text: str) -> tuple[str, Counter[str]]:
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    text, counts = _redact_pii(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, counts


def classify_prompt(text: str) -> str:
    if _CODE_RE.search(text):
        return "structured"
    if _SUMMARY_RE.search(text):
        return "sharegpt_summary"
    if _REASONING_RE.search(text):
        return "open_ended"
    return "sharegpt"


def _bucket(token_count: int) -> str | None:
    for lower, upper, label in LENGTH_BUCKETS:
        if lower <= token_count <= upper:
            return label
    return None


def _token_hash(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocate_sample_counts(population: dict[str, int], requested: int) -> dict[str, int]:
    total = sum(population.values())
    if requested > total:
        raise ValueError(f"requested {requested} samples but only {total} passed filters")
    exact = {key: requested * count / total for key, count in population.items()}
    allocated = {key: math.floor(value) for key, value in exact.items()}
    remaining = requested - sum(allocated.values())
    by_fraction = sorted(population,
                         key=lambda key: (exact[key] - allocated[key], population[key]),
                         reverse=True)
    for key in by_fraction[:remaining]:
        allocated[key] += 1
    return allocated


def _prefix_tokens(tokenizer, target: int) -> list[int]:
    sentence = (
        "System guidance: answer the user's request directly, use accurate facts, "
        "show the key steps when useful, and say when information is uncertain. "
    )
    sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    repeated = (sentence_tokens * math.ceil(target / max(1, len(sentence_tokens))))[:target]
    if len(repeated) != target:
        raise RuntimeError("could not construct the requested common prefix length")
    return repeated


def _arrival_times(count: int, mode: str, rate: float, seed: int) -> list[float]:
    if mode == "burst":
        return [0.0] * count
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("arrival rate must be positive for constant/poisson mode")
    if mode == "constant":
        return [index / rate for index in range(count)]
    rng = random.Random(seed)
    arrivals = [0.0]
    for _ in range(1, count):
        arrivals.append(arrivals[-1] + rng.expovariate(rate))
    return arrivals


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="ShareGPT JSON array or JSONL")
    parser.add_argument("--model", required=True, help="local tokenizer/model directory")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--candidate-pool-per-bucket", type=int, default=5000)
    parser.add_argument("--min-input-tokens", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--min-output-tokens", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--arrival-mode", choices=("burst", "constant", "poisson"),
                        default="poisson")
    parser.add_argument("--arrival-rate", type=float, default=8.0)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=80.0)
    parser.add_argument("--shared-prefix-percent", type=int, choices=(0, 20, 50),
                        default=0)
    parser.add_argument("--shared-prefix-tokens", type=int, choices=(256, 512),
                        default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source-revision", default=None,
                        help="Hub commit hash or local source version label")
    parser.add_argument("--output", required=True, help="output request-trace JSONL")
    parser.add_argument("--prompt-only-output", default=None,
                        help="optional legacy JSONL containing only prompt text")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = Path(os.path.expanduser(args.input)).resolve()
    model_path = os.path.expanduser(args.model)
    output_path = Path(args.output).expanduser()
    if not source_path.is_file():
        raise SystemExit(f"input file does not exist: {source_path}")
    if args.num_samples < 1 or args.candidate_pool_per_bucket < args.num_samples:
        raise SystemExit("sample count must be positive and candidate pool must cover it")
    if args.min_input_tokens < 1 or args.max_input_tokens < args.min_input_tokens:
        raise SystemExit("input-token bounds must satisfy 1 <= min <= max")
    if args.min_output_tokens < 1 or args.max_output_tokens < args.min_output_tokens:
        raise SystemExit("output-token bounds must satisfy 1 <= min <= max")
    for name in ("ttft_slo_ms", "tpot_slo_ms"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                              local_files_only=True)
    rng = random.Random(args.seed)
    reservoirs: dict[str, list[dict[str, Any]]] = {
        label: [] for _, _, label in LENGTH_BUCKETS
    }
    bucket_counts = Counter()
    filter_counts = Counter()
    redaction_counts: Counter[str] = Counter()
    tokenized_count = 0

    print(f"Scanning public source {source_path} with tokenizer {model_path} ...",
          flush=True)
    for record in iter_source_records(source_path):
        filter_counts["source_records"] += 1
        prompt, reply = _conversation_text(record)
        if not isinstance(prompt, str) or not isinstance(reply, str):
            filter_counts["missing_user_or_assistant"] += 1
            continue
        prompt, prompt_redactions = _clean_text(prompt)
        reply, reply_redactions = _clean_text(reply)
        redaction_counts.update(prompt_redactions)
        redaction_counts.update({f"reply_{key}": value
                                 for key, value in reply_redactions.items()})
        if not prompt:
            filter_counts["empty_prompt"] += 1
            continue
        prompt_ids = list(tokenizer.encode(prompt))
        prompt_tokens = len(prompt_ids)
        if prompt_tokens < args.min_input_tokens:
            filter_counts["prompt_too_short"] += 1
            continue
        if prompt_tokens > args.max_input_tokens:
            filter_counts["prompt_too_long"] += 1
            continue
        label = _bucket(prompt_tokens)
        if label is None:
            filter_counts["prompt_outside_length_buckets"] += 1
            continue
        reference_reply_tokens_capped = len(tokenizer.encode(
            reply, truncation=True, max_length=args.max_output_tokens))
        if not reference_reply_tokens_capped:
            filter_counts["empty_reference_reply"] += 1
            continue
        output_budget = min(args.max_output_tokens,
                            max(args.min_output_tokens,
                                reference_reply_tokens_capped))
        item = {
            "prompt": prompt,
            "body_tokens": prompt_ids,
            "user_prompt_tokens": prompt_tokens,
            "reference_reply_tokens_capped": reference_reply_tokens_capped,
            "max_tokens": output_budget,
            "content_class": classify_prompt(prompt),
        }
        bucket_counts[label] += 1
        tokenized_count += 1
        pool = reservoirs[label]
        if len(pool) < args.candidate_pool_per_bucket:
            pool.append(item)
        else:
            replacement = rng.randrange(bucket_counts[label])
            if replacement < args.candidate_pool_per_bucket:
                pool[replacement] = item

    total_eligible = sum(bucket_counts.values())
    if args.num_samples > total_eligible:
        raise SystemExit(
            f"only {total_eligible} prompts passed filters; cannot sample {args.num_samples}")
    sample_counts = _allocate_sample_counts(bucket_counts, args.num_samples)
    selected = []
    for _, _, label in LENGTH_BUCKETS:
        count = sample_counts.get(label, 0)
        if count:
            selected.extend(rng.sample(reservoirs[label], count))
    rng.shuffle(selected)

    prefix_ids = (_prefix_tokens(tokenizer, args.shared_prefix_tokens)
                  if args.shared_prefix_percent else [])
    prefix_count = round(len(selected) * args.shared_prefix_percent / 100)
    prefix_indexes = set(rng.sample(range(len(selected)), prefix_count))
    arrivals = _arrival_times(len(selected), args.arrival_mode,
                              args.arrival_rate, args.seed + 1)
    rows = []
    prompt_only_rows = []
    selected_buckets = Counter()
    selected_classes = Counter()
    for index, (item, arrival) in enumerate(zip(selected, arrivals)):
        has_prefix = index in prefix_indexes
        request_prefix = prefix_ids if has_prefix else []
        full_prompt_ids = request_prefix + item["body_tokens"]
        request_id = f"sharegpt-{args.seed}-{index:04d}"
        row = {
            "request_id": request_id,
            "prompt": item["prompt"],
            "max_tokens": item["max_tokens"],
            "arrival_time_s": round(arrival, 9),
            "arrival_mode": args.arrival_mode,
            "arrival_rate_requests_per_second": (
                args.arrival_rate if args.arrival_mode != "burst" else None),
            "prefix_group": (f"synthetic_system_template_{args.shared_prefix_tokens}"
                             if has_prefix else None),
            "prefix_kind": ("synthetic_common_system_template" if has_prefix else None),
            "prompt_prefix_token_ids": request_prefix,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "prompt_token_hash": _token_hash(full_prompt_ids),
            "prompt_tokens": len(full_prompt_ids),
            "user_prompt_tokens": item["user_prompt_tokens"],
            "prefix_tokens": len(request_prefix),
            "content_class": item["content_class"],
        }
        rows.append(row)
        prompt_only_rows.append({"prompt": item["prompt"]})
        selected_buckets[_bucket(item["user_prompt_tokens"]) or ">2048"] += 1
        selected_classes[item["content_class"]] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.prompt_only_output:
        prompt_only_path = Path(args.prompt_only_output).expanduser()
        prompt_only_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_only_path.open("w", encoding="utf-8", newline="\n") as target:
            for row in prompt_only_rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "trace_format": TRACE_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
            "revision": args.source_revision,
            "license": "apache-2.0",
            "local_file_sha256": _source_sha256(source_path),
            "local_file_bytes": source_path.stat().st_size,
        },
        "trace": {
            "path": str(output_path),
            "sha256": _source_sha256(output_path),
            "requests": len(rows),
            "prompt_only_path": (
                str(Path(args.prompt_only_output).expanduser())
                if args.prompt_only_output else None),
            "prompt_only_sha256": (
                _source_sha256(Path(args.prompt_only_output).expanduser())
                if args.prompt_only_output else None),
        },
        "tokenizer": {
            "model_path": model_path,
            "name_or_path": tokenizer.name_or_path,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
        },
        "sampling": {
            "seed": args.seed,
            "requested_samples": args.num_samples,
            "tokenized_candidates": tokenized_count,
            "eligible_by_user_prompt_length_bucket": dict(bucket_counts),
            "sampled_by_user_prompt_length_bucket": dict(selected_buckets),
            "sampled_by_content_class": dict(selected_classes),
            "filters": dict(filter_counts),
            "redactions": dict(redaction_counts),
        },
        "workload": {
            "arrival_mode": args.arrival_mode,
            "arrival_rate_requests_per_second": (
                args.arrival_rate if args.arrival_mode != "burst" else None),
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "shared_prefix_percent": args.shared_prefix_percent,
            "shared_prefix_tokens": (args.shared_prefix_tokens
                                      if args.shared_prefix_percent else 0),
            "shared_prefix_kind": ("synthetic_common_system_template"
                                   if args.shared_prefix_percent else None),
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} requests to {output_path}")
    print(f"Manifest: {manifest_path}")
    print(f"User prompt buckets: {dict(selected_buckets)}")
    print(f"Content classes: {dict(selected_classes)}")
    print(f"PII replacements: {dict(redaction_counts)}")


if __name__ == "__main__":
    main()
