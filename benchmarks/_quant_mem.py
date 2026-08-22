"""各量化模式的模型权重显存占用（按参数/缓冲区实际字节数统计）。

用法：python benchmarks/_quant_mem.py [none|w8a8|int4|awq|sparse24] [awq_scales_path] [nodense]
第三参 nodense：int4/awq 关闭稠密反量化路径（纯 int4 显存模式）。
"""
import os
import sys

import torch

from nanovllm import LLM

mode = sys.argv[1] if len(sys.argv) > 1 else "none"
awq_path = sys.argv[2] if len(sys.argv) > 2 else ""
dense_path = not (len(sys.argv) > 3 and sys.argv[3] == "nodense")
llm = LLM(os.path.expanduser("~/huggingface/Qwen3-0.6B/"), quantization=mode,
          awq_scales_path=awq_path, int4_dense_path=dense_path, max_model_len=4096)

total = 0
counts = {"int4": 0, "sparse24": 0, "int8": 0, "dense": 0}
for n, m in llm.model_runner.model.named_modules():
    if hasattr(m, "w_int4"):
        total += m.w_int4.numel() * m.w_int4.element_size() + m.w_int4_scale.numel() * m.w_int4_scale.element_size()
        if hasattr(m, "w_deq"):  # 双路径模式的稠密反量化副本（bf16）
            total += m.w_deq.numel() * m.w_deq.element_size()
        counts["int4"] += 1
    elif hasattr(m, "w_s24_v"):
        total += m.w_s24_v.numel() * m.w_s24_v.element_size() + m.w_s24_idx.numel() * m.w_s24_idx.element_size()
        counts["sparse24"] += 1
    elif hasattr(m, "w_int8"):
        total += m.w_int8.numel() * m.w_int8.element_size() + m.w_scale.numel() * m.w_scale.element_size()
        counts["int8"] += 1
    elif getattr(m, "weight", None) is not None:
        total += m.weight.numel() * m.weight.element_size()
        counts["dense"] += 1

print(f"mode={mode}: weights bytes = {total / 1e9:.3f} GB  "
      f"(layers: {counts})")
llm.exit()
