"""Does a request that arrives *during* a decode burst end the burst?

`decode_burst_yield` only inspects `scheduler.waiting` at a round boundary. In
mixed mode requests are submitted between engine steps, so a burst that started
with an empty queue can still be running when a request arrives - and the
scheduler cannot see it. This probe drives the real HTTP service and therefore
arrives mid-burst through the cross-thread signal
(`decode_burst_yield_on_arrival` + `request_decode_burst_yield()`):

  1. submit one long-output request and wait until decode bursts actually run
     (`/health` -> `decode_iterations`, `decode_bursts`);
  2. while that burst is in flight, submit a short "late" request and measure its
     TTFT and the burst counters before/after.

Compare two servers started with and without `--no-decode-burst-yield-on-arrival`.
A late request that is *not* able to interrupt the burst waits for the remaining
rounds of it.

Run against a server already listening::

    python benchmarks/decode_burst_server_probe.py --base-url http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
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
    return data, (time.perf_counter() - started) * 1000.0


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def burst_counters(health: dict) -> dict:
    admission = health.get("admission", {})
    stats = admission.get("step_stats", {})
    return {
        "decode_iterations": stats.get("decode_iterations"),
        "multi_step_decode_steps": stats.get("multi_step_decode_steps"),
        "multi_step_decode_tokens": stats.get("multi_step_decode_tokens"),
        "burst_pressure_yields": stats.get("decode_burst_pressure_yields"),
        "burst_yield_on_arrival": health.get("decode", {}).get("burst_yield_on_arrival"),
        "max_decode_steps": health.get("decode", {}).get("max_steps"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prefix-sentences", type=int, default=20)
    parser.add_argument("--first-output", type=int, default=256)
    parser.add_argument("--late-output", type=int, default=8)
    parser.add_argument("--late-delay-ms", type=float, default=120.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base_url.rstrip("/")
    prompt = build_prefix(args.prefix_sentences)

    def complete(label: str, max_tokens: int, results: dict, index: int) -> None:
        try:
            data, wall_ms = post_json(
                f"{base}/v1/completions",
                {"model": "qwen3", "prompt": prompt, "max_tokens": max_tokens,
                 "temperature": 0.6, "stream": False},
                args.timeout)
            results[index] = {"label": label, "status": "ok", "wall_ms": wall_ms,
                              "completion_tokens": data["usage"]["completion_tokens"]}
        except Exception as exc:  # noqa: BLE001 - probe reports failures as data
            results[index] = {"label": label, "status": "error", "error": str(exc)}

    report = {"base_url": base, "rounds": []}
    report["health_before"] = burst_counters(get_json(f"{base}/health", args.timeout))

    for round_index in range(args.rounds):
        results: dict[int, dict] = {}
        before = burst_counters(get_json(f"{base}/health", args.timeout))

        # Phase 1: one long request that keeps decoding (and therefore bursts).
        long_thread = threading.Thread(
            target=complete, args=(f"long_{round_index}", args.first_output,
                                   results, 0))
        long_thread.start()
        time.sleep(args.late_delay_ms / 1000.0)

        # Phase 2: the late arrival, submitted while the burst is running.
        late_started = time.perf_counter()
        late_thread = threading.Thread(
            target=complete, args=(f"late_{round_index}", args.late_output,
                                   results, 1))
        late_thread.start()
        late_thread.join()
        late_ms = (time.perf_counter() - late_started) * 1000.0
        long_thread.join()

        after = burst_counters(get_json(f"{base}/health", args.timeout))
        report["rounds"].append({
            "round": round_index,
            "before": before,
            "after": after,
            "burst_pressure_yields_delta": (
                (after["burst_pressure_yields"] or 0)
                - (before["burst_pressure_yields"] or 0)),
            "decode_iterations_delta": (
                (after["decode_iterations"] or 0) - (before["decode_iterations"] or 0)),
            "late": results.get(1),
            "late_wall_ms": late_ms,
            "long": results.get(0),
        })
        late = results.get(1, {})
        print(f"round {round_index}: late wall {late_ms:7.1f} ms "
              f"({late.get('status')}) | burst arrival-yields "
              f"+{report['rounds'][-1]['burst_pressure_yields_delta']} | "
              f"decode iters +{report['rounds'][-1]['decode_iterations_delta']}",
              flush=True)

    late_walls = [row["late_wall_ms"] for row in report["rounds"]
                  if row["late"].get("status") == "ok"]
    report["summary"] = {
        "burst_yield_on_arrival": report["health_before"]["burst_yield_on_arrival"],
        "max_decode_steps": report["health_before"]["max_decode_steps"],
        "late_wall_ms": {
            "count": len(late_walls),
            "p50": statistics.median(late_walls) if late_walls else None,
            "min": min(late_walls) if late_walls else None,
            "max": max(late_walls) if late_walls else None,
        },
        "arrival_yield_total": sum(row["burst_pressure_yields_delta"]
                                   for row in report["rounds"]),
    }
    report["meta"] = {"date": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}
    print(json.dumps(report["summary"], indent=2, default=str))
    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"written -> {args.output}")


if __name__ == "__main__":
    main()
