"""Verify prefix-cache reuse through the live HTTP service.

Sends three waves against `nanovllm-serve` and reports, per request, what the
server says was reused (`context.prefix_cache_hit_tokens`, measured) alongside
what the admission controller predicted (`admission.*`, estimated):

  * wave 1: two unrelated long prompts -> must reuse 0 tokens;
  * wave 2: exact repeats of wave 1 + the shared prefix with a fresh suffix ->
    must reuse the whole shared prefix, block-aligned;
  * wave 3: same prefix under a different trailing instruction -> same reuse.

Between waves the script snapshots `/health`, so the scheduler's
`step_stats`/`prefix_cache_hit_tokens` counters can be checked against the
per-request numbers. Responses are requested with `stream` off; TTFT is measured
by the script as an upper bound (first byte of the JSON body would be earlier).

Usage (server already running)::

    python benchmarks/prefix_cache_verify.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango").split()


def build_prefix(sentences: int) -> str:
    return " ".join(
        f"Sentence {index} covers {WORDS[index % len(WORDS)]} and "
        f"{WORDS[(index * 3) % len(WORDS)]} in the long reference document."
        for index in range(sentences))


def post_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data, time.perf_counter() - started


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "p50": statistics.median(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix-sentences", type=int, default=40,
                        help="shared-prefix length in sentences (40 ~= 606 Qwen3 tokens)")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    prefix = build_prefix(args.prefix_sentences)
    suffix_a = " Question A: summarize the document in one word."
    suffix_b = " Question B: name the two words paired in sentence 5."
    unrelated = build_prefix(args.prefix_sentences) + " Completely different trailer."
    unrelated = unrelated.replace("alpha", "zulu")

    def complete(label: str, prompt: str) -> dict:
        payload = {"model": "qwen3", "prompt": prompt, "max_tokens": args.max_tokens,
                   "temperature": args.temperature, "stream": False}
        try:
            data, elapsed = post_json(f"{base}/v1/completions", payload, args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            return {"label": label, "status": "http_error", "code": exc.code,
                    "detail": detail[:400]}
        return {
            "label": label,
            "status": "ok",
            "wall_ms": elapsed * 1000.0,
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"],
            # 实测复用（引擎分配后回填）；估算值见 admission.estimated_cache_hit_tokens。
            # 两者口径不同：估算按提交时的缓存快照预测，实测含同批并发请求之间
            # 共享同一批块（后到者不再计入），因此并发波次下实测可能小于估算。
            "measured_prefix_hit_tokens": data["context"]["prefix_cache_hit_tokens"],
            "estimated_cache_hit_tokens": data["context"]["estimated_cache_hit_tokens"],
            "admission": data["admission"],
            "text_head": data["choices"][0]["text"][:40],
        }

    def health() -> dict:
        snapshot = get_json(f"{base}/health", args.timeout)
        admission = snapshot.get("admission", {})
        return {
            "active_requests": snapshot.get("active_requests"),
            "admission_counts": {key: admission.get(key) for key in (
                "accepted", "deferred_total", "rejected", "deferred_timeouts")},
            "prefill_tps_estimate": admission.get("prefill_tps_estimate"),
            "decode_tps_estimate": admission.get("decode_tps_estimate"),
            "output_length_ratio": admission.get("output_length_ratio"),
            "last_estimate": admission.get("last_estimate"),
            "step_stats": admission.get("step_stats"),
            "kv_pools": admission.get("kv_pools"),
            "decode": snapshot.get("decode"),
        }

    report = {"base_url": base, "prefix_sentences": args.prefix_sentences, "phases": []}
    report["health_before"] = health()

    wave1 = [complete("wave1_unrelated_a", unrelated),
             complete("wave1_shared_prefix_a", prefix + suffix_a)]
    report["phases"].append({"name": "wave1_publish", "requests": wave1,
                             "health": health()})

    wave2 = [complete("wave2_exact_repeat_a", prefix + suffix_a),
             complete("wave2_exact_repeat_b", prefix + suffix_b),
             complete("wave2_new_suffix", prefix + " Question C: count the sentences.")]
    report["phases"].append({"name": "wave2_reuse", "requests": wave2,
                             "health": health()})

    wave3 = [complete(f"wave3_repeat_{index}", prefix + suffix_a) for index in range(3)]
    report["phases"].append({"name": "wave3_reuse_repeat", "requests": wave3,
                             "health": health()})

    rows = [row for phase in report["phases"] for row in phase["requests"]]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    wave1_hits = [row["measured_prefix_hit_tokens"] for row in wave1
                  if row["status"] == "ok"]
    reuse_hits = [row["measured_prefix_hit_tokens"] for row in wave2 + wave3
                  if row["status"] == "ok"]
    block_size = 256  # Config default; reported for alignment checking
    report["summary"] = {
        "requests": len(rows),
        "failed": len(rows) - len(ok_rows),
        "wave1_hits": wave1_hits,
        "reuse_hits": reuse_hits,
        "reuse_hit_block_aligned": all(hit % block_size == 0 for hit in reuse_hits),
        "reuse_hit_min": min(reuse_hits) if reuse_hits else None,
        "reuse_hit_max": max(reuse_hits) if reuse_hits else None,
        "estimate_vs_actual": [
            {"label": row["label"],
             "estimated": row["estimated_cache_hit_tokens"],
             "actual": row["measured_prefix_hit_tokens"],
             "delta": row["estimated_cache_hit_tokens"]
                      - row["measured_prefix_hit_tokens"]}
            for row in ok_rows
        ],
        "wave1_wall_ms": summarize([row["wall_ms"] for row in wave1
                                    if row["status"] == "ok"]),
        "reuse_wall_ms": summarize([row["wall_ms"] for row in wave2 + wave3
                                    if row["status"] == "ok"]),
    }
    report["meta"] = {"date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}
    print(json.dumps(report, indent=2, default=str))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
