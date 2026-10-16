"""Build a deterministic repeated-text request trace for online n-gram ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ttft-slo-ms", type=float, default=2000.0)
    parser.add_argument("--tpot-slo-ms", type=float, default=80.0)
    args = parser.parse_args()
    if args.num_requests < 1 or args.max_tokens < 1:
        parser.error("--num-requests and --max-tokens must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with output.open("w", encoding="utf-8", newline="\n") as trace:
        for index in range(args.num_requests):
            digit = rng.randrange(10)
            pattern = " ".join(str((digit + offset) % 10) for offset in range(8))
            prompt = (
                f"Request {index}: repeat this numeric cycle and continue it: "
                f"{pattern} {pattern} {pattern}")
            row = {
                "request_id": f"synthetic-repeat-{index:04d}",
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.6,
                "ignore_eos": False,
                "ttft_slo_ms": args.ttft_slo_ms,
                "tpot_slo_ms": args.tpot_slo_ms,
                "content_class": "synthetic_repeat",
            }
            trace.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "trace": str(output),
        "trace_sha256": _sha256(output),
        "num_requests": args.num_requests,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
        "ignore_eos": False,
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "content_class": "synthetic_repeat",
        "arrival_times_in_trace": False,
        "arrival_note": "The online benchmark generates synthetic Poisson arrivals.",
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
