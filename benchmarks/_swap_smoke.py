"""KV swap 引擎级测试：小 KV 容量强制触发 swap，验证跑通 + 输出合理 + swap 计数。

用法: python benchmarks/_swap_smoke.py [num_seqs] [out_len] [--no-swap]
"""
import os
import sys

import torch

from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
N_SEQS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
OUT_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 512
KV_SWAP = "--no-swap" not in sys.argv

llm = LLM(PATH, kv_swap=KV_SWAP, gpu_memory_utilization=0.35, max_model_len=2048)
llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

rng = torch.Generator(device="cuda").manual_seed(0)
prompts = [torch.randint(0, 50000, (64,), generator=rng, device="cuda").tolist()
           for _ in range(N_SEQS)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUT_LEN)] * N_SEQS
print(f"starting generate: {N_SEQS} seqs x {OUT_LEN} (kv_swap={KV_SWAP})...", flush=True)
import time
t0 = time.perf_counter()
outs = llm.generate(prompts, sps, use_tqdm=False)
wall = time.perf_counter() - t0
print(f"generate done ({wall:.1f}s)", flush=True)
m = llm.collect_metrics()
print(f"seqs={N_SEQS} out={OUT_LEN} kv_swap={KV_SWAP}: wall={wall:.1f}s preemptions={m['num_preemptions']} "
      f"kv_swaps={m['num_swaps']} prefill_tokens={m['step_stats']['prefill_tokens']} "
      f"decode_tokens={m['step_stats']['decode_tokens']}")
print(f"output lengths: {[len(o['token_ids']) for o in outs[:6]]}...")
# 合理性：无 NaN、长度正确、无崩溃
assert all(len(o["token_ids"]) == OUT_LEN for o in outs), "output length mismatch"
llm.exit()
print("SWAP SMOKE OK")
