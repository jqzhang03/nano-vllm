"""从 DeepSeek-V2-Lite 全量文件切出"embed + 前 4 层 + norm + lm_head"子集
（单文件 safetensors + 截断 config）→ 真实权重数值锚（16GB 卡可放 fp16 双份）。

用法: python benchmarks/_v2lite_build_4l.py [--src DIR] [--dst DIR]
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors import safe_open
import torch

SRC = os.path.expanduser("~/huggingface/DeepSeek-V2-Lite")
DST = os.path.expanduser("~/v2lite_4l")
N_LAYERS = 4


def main():
    src = os.environ.get("V2L_SRC", SRC)
    dst = os.environ.get("V2L_DST", DST)
    os.makedirs(dst, exist_ok=True)
    keep_prefixes = ("model.embed_tokens", "model.norm", "lm_head")
    keep = set()
    meta = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(src, fn), "pt", "cpu") as f:
            for k in f.keys():
                if k.startswith(keep_prefixes) or any(
                        k.startswith(f"model.layers.{i}.") for i in range(N_LAYERS)):
                    keep.add(k)
                    meta[k] = f.get_slice(k).get_shape()
    print(f"keep {len(keep)} tensors")
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg["num_hidden_layers"] = N_LAYERS
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    # 收集子集张量（CPU 内存峰值 ≈ 子集体积 3GB，可接受）
    tensors = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(src, fn), "pt", "cpu") as f:
            for k in keep:
                if k not in tensors and k in f.keys():
                    tensors[k] = f.get_tensor(k)
    from safetensors.torch import save_file
    save_file(tensors, os.path.join(dst, "model.safetensors"))
    for f in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
              "special_tokens_map.json", "generation_config.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dst, f))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    print(f"wrote {dst} ({len(tensors)} tensors, {total/1e9:.2f}GB)")


if __name__ == "__main__":
    main()
