"""MoE toy 引擎吞吐小测：decode tok/s（fp16 vs int4 vs fp8）。

预期（诚实）：MoE 层的 Python 循环 + 小 GEMM 组织税 ~3.5ms/层 → toy(4层,2 MoE 层)
decode 每步 CPU 税 ~7ms → 吞吐被 CPU 主导（几十 tok/s 级），与模型计算无关。
量化的收益（专家权重字节）在本实现里被 CPU 税淹没。此数字的意义 = 显示
"循环专家实现是正确性版，吞吐需要 fused/grouped kernel"。

用法: python benchmarks/_moe_engine_bench.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from _moe_e2e import build_toy, PROMPTS

TOY = os.path.expanduser("~/moe_toy")


def bench_engine(quantization: str):
    from nanovllm import LLM, SamplingParams
    llm = LLM(TOY, max_model_len=128, quantization=quantization, kv_swap=False,
              enforce_eager=True)
    # 预热
    llm.generate(["warm up"], SamplingParams(temperature=0.6, max_tokens=4),
                 use_tqdm=False)
    prompts = PROMPTS * 6  # 12 seqs
    sp = SamplingParams(temperature=0.6, max_tokens=16)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    n_tok = sum(len(o["token_ids"]) for o in outs)
    llm.exit()
    torch.cuda.empty_cache()
    return n_tok, dt


def main():
    build_toy()
    print("=== MoE toy 引擎吞吐（12 seq × 16 tok decode）===")
    for q in ["none", "fp8", "int4"]:
        n_tok, dt = bench_engine(q)
        print(f"  {q:6s}: {n_tok / dt:7.1f} tok/s (decode {dt:.2f}s for {n_tok} tok)")


if __name__ == "__main__":
    main()
