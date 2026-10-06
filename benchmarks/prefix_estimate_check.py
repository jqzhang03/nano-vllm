"""Minimal paired-request check of the admission cache-hit estimate.

Sends the same prompt twice against a running server (default
http://127.0.0.1:8000) and, optionally, two prompts that share a long prefix, so
the estimated vs committed reuse can be read straight off the response:

  * request 1 must report 0 reuse (nothing published yet);
  * request 2 must report the block-aligned shared prefix for both numbers.

Run inside the WSL env from the repo root::

    python benchmarks/prefix_estimate_check.py --prefix-sentences 40
"""
from __future__ import annotations

import argparse
import json
import urllib.request

WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
         "mike november oscar papa quebec romeo sierra tango").split()


def build_prefix(sentences: int) -> str:
    return " ".join(
        f"Sentence {index} covers {WORDS[index % len(WORDS)]} and "
        f"{WORDS[(index * 3) % len(WORDS)]} in the long reference document."
        for index in range(sentences))


def complete(base: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = json.dumps({"model": "qwen3", "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0.6, "stream": False}).encode("utf-8")
    request = urllib.request.Request(
        f"{base}/v1/completions", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return {
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "estimated_cache_hit_tokens": data["context"]["estimated_cache_hit_tokens"],
        "committed_cache_hit_tokens": data["context"]["prefix_cache_hit_tokens"],
        "predicted_ttft_ms": data["admission"]["predicted_ttft_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix-sentences", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    prefix = build_prefix(args.prefix_sentences)

    first = complete(base, prefix + " Question A?", args.max_tokens, args.timeout)
    repeat = complete(base, prefix + " Question A?", args.max_tokens, args.timeout)
    new_tail = complete(base, prefix + " Question B?", args.max_tokens, args.timeout)
    print(json.dumps({"publish": first, "exact_repeat": repeat,
                      "shared_prefix_new_tail": new_tail}, indent=2))


if __name__ == "__main__":
    main()
