"""Step-level timing breakdown of the nano-vllm engine.

Pins down where per-step decode time goes (schedule / COW / run / postprocess).
Usage: python benchmarks/_step_timing.py [auto|fp8_e4m3] [long|small] [eager|quant] [awq_path]
"""
import json
import os
import statistics
import sys
import time

from nanovllm import LLM, SamplingParams

kv_dtype = sys.argv[1] if len(sys.argv) > 1 else "fp8_e4m3"
tag = sys.argv[2] if len(sys.argv) > 2 else "long"
eager = len(sys.argv) > 3 and sys.argv[3] == "eager"
quant = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "eager" else "none"
awq_path = sys.argv[4] if len(sys.argv) > 4 else ""
with open(os.path.join("results", f"compare_workload_{tag}.json"), encoding="utf-8") as f:
    spec = json.load(f)
prompts = spec["prompts"]
max_tokens = spec["max_tokens"]

llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), enforce_eager=eager,
          max_model_len=4096, gpu_memory_utilization=0.9,
          kv_cache_dtype=kv_dtype, quantization=quant, awq_scales_path=awq_path)
llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

for p, mt in zip(prompts, max_tokens):
    llm.add_request(p, SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt))

t_sched, t_cow, t_run, t_post = [], [], [], []
step_kinds = []
while not llm.is_finished():
    t0 = time.perf_counter()
    seqs, kind = llm.scheduler.schedule()
    t1 = time.perf_counter()
    for old_id, new_id in llm.scheduler.cow_pairs:
        llm.model_runner.call("cow_block", old_id, new_id)
    t2 = time.perf_counter()
    result = llm.model_runner.call("run", seqs, kind, False)
    token_ids = result
    t3 = time.perf_counter()
    llm.scheduler.postprocess(seqs, token_ids)
    t4 = time.perf_counter()
    t_sched.append(t1 - t0)
    t_cow.append(t2 - t1)
    t_run.append(t3 - t2)
    t_post.append(t4 - t3)
    step_kinds.append(kind)

for name, arr, kind in (("schedule", t_sched, None), ("cow", t_cow, None),
                        ("run", t_run, None), ("postprocess", t_post, None)):
    if arr:
        print(f"{name:<12} n={len(arr):4d} avg={statistics.fmean(arr) * 1000:7.1f}ms "
              f"p50={statistics.median(arr) * 1000:7.1f}ms max={max(arr) * 1000:7.1f}ms")

dec = [r for r, k in zip(t_run, step_kinds) if k == "decode"]
pre = [r for r, k in zip(t_run, step_kinds) if k == "prefill"]
print(f"== kv={kv_dtype} tag={tag} quant={quant} awq={awq_path} ==")
print(f"run(decode)  n={len(dec):4d} avg={statistics.fmean(dec) * 1000:7.1f}ms p50={statistics.median(dec) * 1000:7.1f}ms p99={sorted(dec)[int(len(dec) * 0.99)] * 1000:7.1f}ms")
if pre:
    print(f"run(prefill) n={len(pre):4d} avg={statistics.fmean(pre) * 1000:7.1f}ms p50={statistics.median(pre) * 1000:7.1f}ms")
print(f"steps: {len(step_kinds)} (prefill {step_kinds.count('prefill')}, decode {step_kinds.count('decode')})")
llm.exit()
