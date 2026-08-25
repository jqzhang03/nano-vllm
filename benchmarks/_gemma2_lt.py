"""安装版 transformers 对 gemma-2 各 config 的实际 layer_types 解析（parity 对照基准）。

用法: python benchmarks/_gemma2_lt.py
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from transformers import AutoConfig  # noqa: E402


def main():
    for repo in ("/home/zjq/huggingface/gemma-2-2b-it", "unsloth/gemma-2-9b-it"):
        try:
            cfg = AutoConfig.from_pretrained(repo)
        except Exception as e:
            print(f"{repo}: ERROR {e}")
            continue
        lt = getattr(cfg, "layer_types", "MISSING")
        at = getattr(cfg, "attention_types", "MISSING")
        print(f"{repo}: layers={cfg.num_hidden_layers}")
        print(f"  layer_types: {lt if isinstance(lt, str) else lt[:16]}{'...' if isinstance(lt, list) and len(lt) > 16 else ''}")
        print(f"  attention_types attr: {at}")
        if isinstance(lt, list):
            print(f"  count: full={lt.count('full_attention')} sliding={lt.count('sliding_attention')}")


if __name__ == "__main__":
    main()
