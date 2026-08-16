"""torch.profiler breakdown of prefill vs decode steps.

Profiles the prefill phase and the decode phase separately, then prints the
top CUDA kernels/ops of each phase. Default is eager mode for clean per-kernel
attribution; pass --cudagraph to profile the CUDA-graph decode path instead.

NOTE: this file must NOT be named profile.py — it would shadow the stdlib
`profile` module that cProfile (imported by torch._dynamo) depends on.

Run from the WSL conda env:

    python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64
"""
from __future__ import annotations

import argparse
import os

from torch.profiler import ProfilerActivity, profile

from nanovllm import LLM, SamplingParams
from bench import add_workload_args, build_workload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    add_workload_args(p)
    p.add_argument("--cudagraph", action="store_true",
                   help="profile the CUDA-graph decode path instead of eager")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--warmup-seqs", type=int, default=8)
    p.add_argument("--max-prefill-steps", type=int, default=10)
    p.add_argument("--max-decode-steps", type=int, default=20)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--sort-by", default="self_cuda_time_total",
                   choices=["cuda_time_total", "self_cuda_time_total",
                            "cpu_time_total", "self_cpu_time_total"])
    p.add_argument("--output-dir", default="profiles")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    llm = LLM(args.model, enforce_eager=not args.cudagraph, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    # 预热：在profiler之外完成torch.compile/triton的JIT编译与CUDA Graph捕获
    llm.generate(["warm up"] * args.warmup_seqs,
                 SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True) if args.prompts_file else None
    prompts, sampling_params = build_workload(args, tokenizer)
    for prompt, sp in zip(prompts, sampling_params):
        llm.add_request(prompt, sp)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- phase 1: prefill steps（waiting队列清空即prefill结束） ----
    print(f"== profiling prefill (eager={not args.cudagraph}) ==")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        n = 0
        while llm.scheduler.waiting:
            _, num_tokens = llm.step()
            assert num_tokens > 0, "expected a prefill step"
            n += 1
            if n >= args.max_prefill_steps:
                break
    table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.top_k)
    print(table)
    with open(os.path.join(args.output_dir, "prefill.txt"), "w") as f:
        f.write(table + "\n")

    # ---- phase 2: decode steps ----
    print(f"== profiling decode (eager={not args.cudagraph}) ==")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.max_decode_steps):
            _, num_tokens = llm.step()
            assert num_tokens < 0, "expected a decode step"
            if llm.is_finished():
                break
    table = prof.key_averages().table(sort_by=args.sort_by, row_limit=args.top_k)
    print(table)
    with open(os.path.join(args.output_dir, "decode.txt"), "w") as f:
        f.write(table + "\n")
    print(f"tables saved under {args.output_dir}/ (prefill.txt, decode.txt)")


if __name__ == "__main__":
    main()
