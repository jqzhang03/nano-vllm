"""Create disjoint, length/output-budget matched subsets from a prompt trace."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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


def _token_bucket(value: int, bounds: tuple[int, ...]) -> str:
    lower = 1
    for upper in bounds:
        if value <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{bounds[-1]}"


def _allocate(total: int, groups: dict[tuple[str, str], list[dict]]) -> dict:
    population = sum(map(len, groups.values()))
    quotas = {key: total * len(rows) / population for key, rows in groups.items()}
    counts = {key: int(quota) for key, quota in quotas.items()}
    remaining = total - sum(counts.values())
    order = sorted(groups, key=lambda key: (quotas[key] - counts[key], len(groups[key]), key),
                   reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--calibration-count", type=int, default=250)
    parser.add_argument("--heldout-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    source = Path(args.source)
    calibration_path = Path(args.calibration_output)
    heldout_path = Path(args.heldout_output)
    if args.calibration_count < 1 or args.heldout_count < 1:
        parser.error("split counts must be positive")

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    requested = args.calibration_count + args.heldout_count
    if requested > len(rows):
        parser.error(f"requested {requested} prompts but source has {len(rows)}")
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
           for row in rows):
        raise ValueError("every source row must contain a non-empty prompt")

    input_bounds = (256, 512, 1024, 2048)
    output_bounds = (64, 128, 256, 512)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        input_tokens = int(row.get("user_prompt_tokens", row.get("prompt_tokens", 0)))
        output_tokens = int(row.get("max_tokens", 0))
        groups[(_token_bucket(input_tokens, input_bounds),
                _token_bucket(output_tokens, output_bounds))].append(row)

    rng = random.Random(args.seed)
    heldout_counts = _allocate(args.heldout_count, groups)
    calibration = []
    heldout = []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        heldout.extend(members[:heldout_counts[key]])
        calibration.extend(members[heldout_counts[key]:])

    # If a smaller requested split leaves source rows unused, sample only the
    # requested calibration rows while preserving the same strata balance.
    if len(calibration) > args.calibration_count:
        calibration_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in calibration:
            key = (_token_bucket(int(row.get("user_prompt_tokens", row.get("prompt_tokens", 0))),
                                 input_bounds),
                   _token_bucket(int(row.get("max_tokens", 0)), output_bounds))
            calibration_groups[key].append(row)
        calibration_counts = _allocate(args.calibration_count, calibration_groups)
        selected = []
        for key, members in calibration_groups.items():
            selected.extend(members[:calibration_counts[key]])
        calibration = selected

    def output_row(row: dict) -> dict:
        return {
            "request_id": row.get("request_id"),
            "prompt": row["prompt"],
            "max_tokens": int(row["max_tokens"]),
            "ttft_slo_ms": float(row.get("ttft_slo_ms", 2000.0)),
            "tpot_slo_ms": float(row.get("tpot_slo_ms", 80.0)),
            "prompt_token_hash": row.get("prompt_token_hash"),
            "prompt_tokens": int(row.get("prompt_tokens", 0)),
            "user_prompt_tokens": int(row.get("user_prompt_tokens", 0)),
            "content_class": row.get("content_class"),
        }

    calibration = sorted(calibration, key=lambda row: row.get("request_id", ""))
    heldout = sorted(heldout, key=lambda row: row.get("request_id", ""))
    if len(calibration) != args.calibration_count or len(heldout) != args.heldout_count:
        raise RuntimeError("split did not produce the requested calibration/held-out sizes")
    calibration_ids = {row.get("request_id") for row in calibration}
    heldout_ids = {row.get("request_id") for row in heldout}
    if calibration_ids & heldout_ids:
        raise RuntimeError("request IDs overlap across calibration and held-out")
    if not all(calibration_ids) or not all(heldout_ids):
        raise RuntimeError("source request IDs must be non-empty and unique")
    if len(calibration_ids) != len(calibration) or len(heldout_ids) != len(heldout):
        raise RuntimeError("source request IDs are not unique")

    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    heldout_path.parent.mkdir(parents=True, exist_ok=True)
    for path, selected in ((calibration_path, calibration), (heldout_path, heldout)):
        with path.open("w", encoding="utf-8", newline="\n") as output:
            for row in selected:
                output.write(json.dumps(output_row(row), ensure_ascii=False) + "\n")

    def distributions(selected: list[dict]) -> dict:
        lengths = Counter(_token_bucket(int(row.get(
            "user_prompt_tokens", row.get("prompt_tokens", 0))), input_bounds)
                          for row in selected)
        budgets = Counter(_token_bucket(int(row.get("max_tokens", 0)), output_bounds)
                          for row in selected)
        joint = Counter(
            f"{_token_bucket(int(row.get('user_prompt_tokens', row.get('prompt_tokens', 0))), input_bounds)}|"
            f"{_token_bucket(int(row.get('max_tokens', 0)), output_bounds)}"
            for row in selected)
        content = Counter(row.get("content_class", "unknown") for row in selected)
        return {"input_length_buckets": dict(lengths),
                "output_budget_buckets": dict(budgets),
                "joint_buckets": dict(joint),
                "content_classes": dict(content)}

    manifest = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "seed": args.seed,
        "calibration_trace": str(calibration_path),
        "calibration_sha256": _sha256(calibration_path),
        "heldout_trace": str(heldout_path),
        "heldout_sha256": _sha256(heldout_path),
        "calibration_requests": len(calibration),
        "heldout_requests": len(heldout),
        "request_id_overlap": len(calibration_ids & heldout_ids),
        "arrival_times_in_output": False,
        "arrival_note": "The serving benchmark generates Poisson arrivals from its CLI seed/rate.",
        "stratified_by": ["user_prompt_length_bucket", "output_budget_bucket"],
        "calibration_distribution": distributions(calibration),
        "heldout_distribution": distributions(heldout),
        "request_ids": {
            "calibration": sorted(calibration_ids),
            "heldout": sorted(heldout_ids),
        },
    }
    manifest_path = calibration_path.with_name(
        calibration_path.stem.replace("_calibration", "") + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"request_ids"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
