"""MoE 量化引擎对照：toy 模型 int4(dual-path) / 纯 int4 / fp8 logits vs fp16 基线。

基线 = _moe_e2e 存的 fp16(bf16) prefill logits（/tmp/_moe_e2e_logits.pt）。
预期（诚实）：
- int4 dual-path：专家 inter=768 < 2048 → _int4_forward 全走 w_deq（cuBLAS bf16
  副本）→ logits 应与基线几乎位级一致（数值=同一 bf16 权重）；
- 纯 int4：真 Triton int4 GEMM（RTN 误差）→ mean diff 增大，top-1 期望仍 100%；
- fp8：e4m3 per-column（RTN）→ 小误差；
注意 toy 模型随机权重均匀（量化误差对 logits 影响被平均），真实模型数字需真权重。

用法: python benchmarks/_moe_quant.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from _moe_e2e import build_toy, PROMPTS

TOY = os.path.expanduser("~/moe_toy")
DUMP = "/tmp/_moe_e2e_logits.pt"


def run_quant(quantization: str, int4_dense_path=None):
    from nanovllm import LLM, SamplingParams
    kw = dict(max_model_len=128, quantization=quantization, kv_swap=False,
              enforce_eager=True)
    if int4_dense_path is not None:
        kw["int4_dense_path"] = int4_dense_path
    llm = LLM(TOY, **kw)
    llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1),
                 use_tqdm=False, collect_logits=True)
    lg = None
    for kind, x in llm.collected_logits:
        if kind == "prefill" and x is not None:
            lg = x
            break
    llm.exit()
    torch.cuda.empty_cache()  # 同进程连续建引擎：显存还给 caching allocator 才能复用
    return lg


def main():
    build_toy()
    if not os.path.exists(DUMP):
        print("baseline missing -> running fp16 engine first")
        from nanovllm import LLM, SamplingParams
        llm = LLM(TOY, max_model_len=128, quantization="none", kv_swap=False,
                  enforce_eager=True)
        llm.generate(["warm up"] * 2, SamplingParams(temperature=0.6, max_tokens=2),
                     use_tqdm=False)
        llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1),
                     use_tqdm=False, collect_logits=True)
        for kind, x in llm.collected_logits:
            if kind == "prefill" and x is not None:
                torch.save(x, DUMP)
                break
        llm.exit()
    base = torch.load(DUMP)
    print(f"baseline fp16 logits shape {tuple(base.shape)}\n")
    for name, q, dp in [("int4 dual-path", "int4", None),
                        ("int4 pure", "int4", False),
                        ("fp8", "fp8", None)]:
        lg = run_quant(q, dp)
        assert lg is not None
        diff = (lg - base).abs()
        top1 = (lg.argmax(-1) == base.argmax(-1)).float().mean().item()
        print(f"{name:14s}: mean diff {diff.mean().item():.6f} | "
              f"max diff {diff.max().item():.4f} | top-1 {100 * top1:.1f}%")


if __name__ == "__main__":
    main()
