"""用真实 config 验证 Mistral/Gemma-2 模型类构造（CPU，无权重）：解析器 + 结构正确性。

用法: python benchmarks/_model_build.py <model_dir>...
"""
import os
import sys

import torch

from transformers import AutoConfig

from nanovllm.models.registry import get_model_class
from nanovllm.models.gemma2 import gemma2_layer_types


def main():
    for arg in sys.argv[1:]:
        path = os.path.expanduser(arg)
        hf = AutoConfig.from_pretrained(path)
        print(f"\n=== {path} model_type={hf.model_type} ===")
        if hf.model_type == "gemma2":
            lt = gemma2_layer_types(hf, hf.num_hidden_layers)
            print(f"layer_types: {lt[:14]}{'...' if len(lt) > 14 else ''} "
                  f"(global={lt.count('global')}, sliding={lt.count('sliding')})")
        cls = get_model_class(hf.model_type)
        torch.set_default_dtype(hf.dtype)
        m = cls(hf)
        attns = [x for x in m.modules() if hasattr(x, "window_size")]
        windows = {a.window_size for a in attns}
        softcaps = {a.logit_softcapping for a in attns}
        n_attn = len(attns)
        print(f"built {cls.__name__}: attn layers={n_attn} "
              f"windows={windows} softcaps={softcaps}")
        print(f"qkv merged: {hasattr(m.model.layers[0].self_attn, 'qkv_proj')} | "
              f"mlp gate_up merged: {hasattr(m.model.layers[0].mlp, 'gate_up_proj')}")
        params = sum(p.numel() for p in m.parameters())
        print(f"params: {params/1e9:.2f}B (fp16={params*2/1e9:.1f}GB)")


if __name__ == "__main__":
    main()
