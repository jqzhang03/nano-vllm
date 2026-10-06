"""Reproduce "a request lands while a decode burst is mid-flight".

`decode_burst_yield` alone cannot stop a mixed-mode burst for a *newly arriving*
request: the request is not in `scheduler.waiting` yet (it is only added between
engine steps), and the burst's entry guard already saw an empty queue. The
arrival-pressure signal (`decode_burst_yield_on_arrival`, plus the service-side
`request_decode_burst_yield()` or the trace callback) is what ends the burst in
its later rounds.

This harness drives `LLMEngine` directly with a synthetic arrival trace that is
constructed to fire during a burst:

  * phase 1 submits a few long-output requests and runs them alone, so the engine
    has a pure-decode window with an empty waiting queue (bursts are eligible);
  * phase 2 submits a second wave *while* phase 1 is still decoding, using a
    wall-clock trigger inside the driver loop - the callback reports "a request is
    due now" from within `engine.step()`, i.e. mid-burst.

Measured per variant: burst rounds (model forwards per burst), how often the
arrival signal ended a burst, and the TTFT of the late arrivals.

Run from the repo root in the WSL env::

    python benchmarks/decode_burst_arrival_probe.py --num-seqs 24 --max-decode-steps 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-seqs", type=int, default=24,
                        help="late arrivals submitted while the first wave decodes")
    parser.add_argument("--warmup-seqs", type=int, default=2)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-decode-steps", type=int, default=4)
    parser.add_argument("--first-wave-output", type=int, default=64,
                        help="output tokens of the first wave (keeps it decoding)")
    parser.add_argument("--late-output", type=int, default=16)
    parser.add_argument("--late-after-ms", type=float, default=300.0,
                        help="submit the late wave this long after the first wave starts")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--variants", nargs="+",
                        default=["arrival_yield_on", "arrival_yield_off",
                                 "multi_step_off"],
                        choices=["arrival_yield_on", "arrival_yield_off",
                                 "multi_step_off"])
    return parser.parse_args()


VARIANT_OPTIONS = {
    # 默认口径：到达压力可打断 burst
    "arrival_yield_on": {"multi_step_decode": True,
                         "decode_burst_yield": True,
                         "decode_burst_yield_on_arrival": True},
    # 消融一：到达压力忽略，burst 跑满轮数
    "arrival_yield_off": {"multi_step_decode": True,
                          "decode_burst_yield": True,
                          "decode_burst_yield_on_arrival": False},
    # 消融二：完全没有多步（每步 1 轮），作为下界参考
    "multi_step_off": {"multi_step_decode": False,
                       "decode_burst_yield": True,
                       "decode_burst_yield_on_arrival": True},
}


def run_variant(args, options) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    engine = LLM(
        args.model,
        execution_mode="mixed",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_decode_steps=args.max_decode_steps,
        **options,
    )
    try:
        # Warmup with synthetic prompts that never enter the measured trace.
        vocab = max(2, int(getattr(engine.tokenizer, "vocab_size", 32000)))
        engine.generate([[1000 + index for index in range(64)]],
                        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=4),
                        use_tqdm=False)
        engine._req_metrics = []
        engine._step_stats = engine._empty_step_stats()
        for scheduler in ((engine.prefill_scheduler, engine.decode_scheduler)
                          if engine._pd else (engine.scheduler,)):
            scheduler.reset_metrics()

        rng = random.Random(args.seed)
        first = [rng.sample(range(0, 10000), 128) for _ in range(2)]
        late = [rng.sample(range(0, 10000), args.max_input_len)
                for _ in range(args.num_seqs)]
        # The first wave must keep decoding while the late wave arrives.
        first_params = SamplingParams(temperature=0.6, ignore_eos=True,
                                      max_tokens=args.first_wave_output)
        late_params = SamplingParams(temperature=0.6, ignore_eos=True,
                                     max_tokens=args.late_output)

        # Arrival trigger read from inside engine.step(): True once the late wave
        # is due. This is the ablation harness' documented pressure callback.
        state = {"late_due": False}
        engine.set_decode_burst_yield_callback(lambda: state["late_due"])

        torch.cuda.synchronize()
        start = time.perf_counter()
        late_seq_ids: set[int] = set()
        for prompt in first:
            engine.add_request(prompt, first_params, submitted_at=time.perf_counter())
        first_submitted = time.perf_counter()

        next_late = 0
        completed = 0
        total = len(first) + len(late)
        while completed < total:
            now = time.perf_counter()
            if (next_late < len(late)
                    and now - first_submitted >= args.late_after_ms / 1000.0):
                # Arm the signal before adding, mirroring the server: accepted work
                # exists in the service queue while the burst is still running.
                state["late_due"] = True
                while next_late < len(late):
                    late_seq_ids.add(engine.add_request(
                        late[next_late], late_params, submitted_at=time.perf_counter()))
                    next_late += 1
            else:
                state["late_due"] = False

            if engine.is_finished():
                break
            finished, _kind, _n_prefill, _n_decode = engine.step()
            completed += len(finished)
            if (next_late >= len(late) and state["late_due"]
                    and not engine.scheduler.waiting):
                state["late_due"] = False

        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        metrics = engine.collect_metrics()
        rows = metrics["per_request"]
        late_rows = [row for row in rows if row["seq_id"] in late_seq_ids]
        late_ttft = [(row["t_first_token"] - row["t_submitted"]) * 1000.0
                     for row in late_rows
                     if row["t_first_token"] is not None]
        late_ttft.sort()
        bursts = metrics.get("decode_bursts", 0)
        return {
            "variant_options": options,
            "wall_seconds": wall,
            "requests_completed": len(rows),
            "late_submissions": next_late,
            "late_ttft_ms": {
                "count": len(late_ttft),
                "p50": late_ttft[len(late_ttft) // 2] if late_ttft else None,
                "max": late_ttft[-1] if late_ttft else None,
            },
            "bursts": bursts,
            "burst_rounds": metrics.get("decode_burst_rounds", 0),
            "rounds_per_burst": (metrics.get("decode_burst_rounds", 0) / bursts
                                 if bursts else None),
            "queued_yields": metrics.get("decode_burst_yields", 0),
            "arrival_yields": metrics.get("decode_burst_pressure_yields", 0),
            "output_tokens": sum(row["completion_tokens"] for row in rows),
            "decode_iterations": metrics.get("step_stats", {}).get(
                "decode_iterations", 0),
        }
    finally:
        engine.exit()
        del engine
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args.model = os.path.expanduser(args.model)
    runs = {}
    for name in args.variants:
        print(f"running {name} ...", flush=True)
        runs[name] = run_variant(args, VARIANT_OPTIONS[name])
        result = runs[name]
        rounds = result["rounds_per_burst"]
        print(f"  bursts {result['bursts']} | rounds/burst "
              f"{'n/a' if rounds is None else f'{rounds:.2f}'} | "
              f"queued-yield {result['queued_yields']} | arrival-yield "
              f"{result['arrival_yields']} | late TTFT p50/max "
              f"{result['late_ttft_ms']['p50'] or 0:.1f}/"
              f"{result['late_ttft_ms']['max'] or 0:.1f} ms | "
              f"wall {result['wall_seconds']:.2f}s")
    print("\nvariant                 bursts  rounds/burst  queued-yield  arrival-yield  late TTFT p50  wall s")
    for name, result in runs.items():
        rounds = result["rounds_per_burst"]
        print(f"{name:<23} {result['bursts']:>6} "
              f"{'n/a' if rounds is None else f'{rounds:.2f}':>12} "
              f"{result['queued_yields']:>13} {result['arrival_yields']:>14} "
              f"{result['late_ttft_ms']['p50'] or 0:>13.1f} {result['wall_seconds']:>7.2f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or os.path.join(
        "results", f"decode_burst_arrival_{timestamp}.json")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"meta": {"model": args.model, "date": timestamp},
                   "config": vars(args), "runs": runs},
                  handle, indent=2, default=str)
    print(f"full results -> {output_path}")


if __name__ == "__main__":
    main()
