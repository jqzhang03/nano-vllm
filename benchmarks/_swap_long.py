"""KV swap 长序列对比：长 input 下 recompute 恢复成本高（重算整个序列），swap 拷贝成本固定。
用法: python benchmarks/_swap_long.py [--no-swap]"""
import os
import sys
import time

import torch

from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
KV_SWAP = "--no-swap" not in sys.argv
N_SEQS, IN_LEN, OUT_LEN = 64, 2048, 128

llm = LLM(PATH, kv_swap=KV_SWAP, gpu_memory_utilization=0.5, max_model_len=4096)
llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

rng = torch.Generator(device="cuda").manual_seed(0)
prompts = [torch.randint(0, 50000, (IN_LEN,), generator=rng, device="cuda").tolist()
           for _ in range(N_SEQS)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=OUT_LEN)] * N_SEQS
print(f"long-seq: {N_SEQS} seqs x {IN_LEN}+{OUT_LEN} (kv_swap={KV_SWAP})...", flush=True)
t0 = time.perf_counter()
outs = llm.generate(prompts, sps, use_tqdm=False)
wall = time.perf_counter() - t0
m = llm.collect_metrics()
print(f"wall={wall:.1f}s preemptions={m['num_preemptions']} kv_swaps={m['num_swaps']} "
      f"decode_tokens={m['step_stats']['decode_tokens']}")
assert all(len(o["token_ids"]) == OUT_LEN for o in outs)
llm.exit()
print("OK")
