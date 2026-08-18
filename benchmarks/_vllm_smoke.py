"""Smoke test v2: two timed generates + metrics snapshot probing.

Questions answered:
  1. Is the ~40s first-generate one-time (lazy LM-head compile) or per-run?
  2. Does llm.get_metrics() expose TTFT/TPOT/E2E histograms?
  3. What does RequestOutput expose?
"""
import os
import time

import torch
from vllm import LLM, SamplingParams

llm = LLM(model=os.path.expanduser("~/huggingface/Qwen3-0.6B/"), dtype="auto",
          gpu_memory_utilization=0.9, max_model_len=4096, max_num_batched_tokens=16384,
          max_num_seqs=512, kv_cache_dtype="auto", enforce_eager=False,
          enable_prefix_caching=True, swap_space=0, disable_log_stats=False)


def timed_generate(n: int, label: str):
    t0 = time.time()
    out = llm.generate(["Hello, world!"] * n, SamplingParams(temperature=0.6, max_tokens=8),
                       use_tqdm=False)
    dt = time.time() - t0
    print(f"[{label}] {n} prompt(s) x 8 tok: {dt:.2f}s -> {out[0].outputs[0].token_ids}")
    return dt


d1 = timed_generate(1, "first")
d2 = timed_generate(4, "second")
d3 = timed_generate(4, "third")
print(f"timings: first={d1:.2f}s second={d2:.2f}s third={d3:.2f}s")

print("=== get_metrics ===")
metrics = llm.get_metrics()
if not metrics:
    print("EMPTY")
for m in metrics:
    name = getattr(m, "name", "?")
    if any(k in name for k in ("ttft", "e2e", "output_token", "request_success")):
        if hasattr(m, "sum"):
            print(f"{name}: count={m.count} sum={m.sum:.3f} mean={(m.sum / m.count) if m.count else None:.4f}")
        else:
            print(f"{name}: value={getattr(m, 'value', None)}")

print("=== RequestOutput fields ===")
out = llm.generate(["Hello again!"], SamplingParams(temperature=0.6, max_tokens=4),
                   use_tqdm=False)
ro = out[0]
print("fields:", [a for a in dir(ro) if not a.startswith("_")])
print("metrics:", ro.metrics)
print("SMOKE_OK")
