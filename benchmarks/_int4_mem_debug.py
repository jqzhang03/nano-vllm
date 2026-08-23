"""诊断 int4 双路径模式的缓冲明细与 dtype（为什么 _quant_mem 显示 1.73GB）。

用法：python benchmarks/_int4_mem_debug.py [model]
"""
import os
import sys

from nanovllm import LLM

MODEL = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
llm = LLM(MODEL, quantization="int4", max_model_len=4096)
tot = {}
for n, m in llm.model_runner.model.named_modules():
    for b in ("w_int4", "w_int4_scale", "w_deq"):
        if hasattr(m, b):
            t = getattr(m, b)
            tot[b] = tot.get(b, 0) + t.numel() * t.element_size()
            print(f"{n:<52} {b:<12} {str(t.shape):<18} {t.dtype} {t.numel() * t.element_size() / 1e6:.1f} MB")
print("totals:", {k: round(v / 1e6, 1) for k, v in tot.items()})
llm.exit()
