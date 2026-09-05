"""MoE 真实模型边界核对：参数量（config 解析）与权重文件大小（hub 元数据）。

只拉 config.json + 文件清单（不下载权重）。目标：诚实回答"本机 16GB 能跑哪个 MoE、
下载要多大"。注意：官方 bf16 权重通常远大于 int4 后显存需求——磁盘/带宽是另一道门槛。

用法: python benchmarks/_moe_model_probe.py
"""
import sys

import torch

REPOS = {
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
    "Qwen1.5-MoE-A2.7B": "Qwen/Qwen1.5-MoE-A2.7B",
    "DeepSeek-V2-Lite": "deepseek-ai/DeepSeek-V2-Lite",
}


def hub_files(repo: str) -> dict:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings if s.size is not None}


def config_of(repo: str):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(repo)


def main():
    print("=== MoE 模型边界核对（config 解析 + hub 文件清单，不下载权重）===\n")
    for name, repo in REPOS.items():
        try:
            cfg = config_of(repo)
            files = hub_files(repo)
        except Exception as e:
            print(f"{name}: probe failed: {e}")
            continue
        st_bytes = sum(v for k, v in files.items() if k.endswith(".safetensors"))
        st_gb = st_bytes / 1e9
        mt = getattr(cfg, "model_type", "?")
        # 参数量：bf16 字节 / 2（safetensors 全 bf16 时近似；fp32 部分让估计偏小——
        # 多数新模型全 bf16）
        params = st_bytes / 2.0
        print(f"--- {name} ({repo}) ---")
        print(f"  model_type={mt} | hidden={getattr(cfg, 'hidden_size', '?')} "
              f"layers={getattr(cfg, 'num_hidden_layers', '?')}")
        print(f"  experts={getattr(cfg, 'num_experts', '?')} "
              f"top_k={getattr(cfg, 'num_experts_per_tok', getattr(cfg, 'moe_top_k', '?'))} "
              f"step={getattr(cfg, 'decoder_sparse_step', '?')} "
              f"shared={getattr(cfg, 'num_shared_experts', '?')}")
        print(f"  safetensors 合计 {st_gb:.1f} GB (bf16 存储) → 参数 ≈ {params / 1e9:.1f}B")
        print(f"  int4 权重估计 ≈ {params * 0.5 / 1e9:.1f} GB "
              f"（int4=0.5B/参数，不含 scale 与 KV）")
        if mt in ("qwen3_moe", "qwen2_moe", "mixtral", "deepseek_moe"):
            pass
        print()


if __name__ == "__main__":
    main()
