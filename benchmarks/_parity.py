"""HF 参考 parity：本引擎（fp16 eager）prefill logits vs transformers（GPU bf16）。

新架构端口（Mistral SWA / Gemma-2 softcap+交替注意力）正确性的金标准：top-1 100%
= 结构读对（qkv 打包、window 语义、softcap 位置、激活函数）；mean diff 0.05~0.1
为 bf16 + flash-attn vs SDPA 的正常偏差。端口错误（窗口 off-by-one / softcap 位置错
/ 激活错）会塌到 top-1 < 60%。

两阶段（WSL 仅 7GB RAM + 单卡 16GB）：引擎先跑（fp16，7B=14GB）→ 退出释放显存 →
HF 再在 GPU 上加载对照。

用法: python benchmarks/_parity.py <model_dir> [dump.pt]
"""
import os
import sys

import torch

from nanovllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
]


def main():
    model = os.path.expanduser(sys.argv[1])
    dump = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else "/tmp/_parity_logits.pt"

    # ---- 阶段1：本引擎（fp16，不量化——隔离架构正确性） ----
    # 7B fp16 权重 ~14.5GB 塞满 16GB 卡：提高显存利用率 + 缩小 warmup 峰值，
    # 给 KV cache 挤出空间（默认 0.9 利用率 + 16k token warmup 会算出 0 块）
    llm = LLM(model, max_model_len=1024, quantization="none", kv_swap=False,
              gpu_memory_utilization=0.95, max_num_batched_tokens=512)
    hf = llm.config.hf_config
    print(f"\n=== engine model_type={hf.model_type} ===")
    print(f"layers={hf.num_hidden_layers} hidden={hf.hidden_size} "
          f"heads={hf.num_attention_heads} kv_heads={hf.num_key_value_heads} "
          f"head_dim={getattr(hf, 'head_dim', None)} "
          f"sliding_window={getattr(hf, 'sliding_window', None)}")
    if hf.model_type == "gemma2":
        from nanovllm.models.gemma2 import gemma2_layer_types
        lt = gemma2_layer_types(hf, hf.num_hidden_layers)
        print(f"layer_types: {lt[:14]}{'...' if len(lt) > 14 else ''} "
              f"(global={lt.count('global')}, sliding={lt.count('sliding')})")
        print(f"attn_logit_softcapping={getattr(hf, 'attn_logit_softcapping', None)} "
              f"final_logit_softcapping={getattr(hf, 'final_logit_softcapping', None)} "
              f"query_pre_attn_scalar={getattr(hf, 'query_pre_attn_scalar', None)}")
    llm.generate(["warm up"] * 2, SamplingParams(temperature=0.6, max_tokens=4), use_tqdm=False)
    llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1),
                 use_tqdm=False, collect_logits=True)
    logits = None
    for kind, lg in llm.collected_logits:
        if kind == "prefill" and lg is not None:
            logits = lg  # [n_seqs, vocab] fp32
            break
    llm.exit()  # 释放显存（7B fp16 = 14GB，必须退出后才能加载 HF）
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    assert logits is not None, "no prefill logits collected"
    torch.save(logits, dump)
    print(f"saved engine logits -> {dump}")

    # ---- 阶段2：HF 参考（GPU bf16） ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    ref = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
    ref_logits = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = torch.tensor([tok.encode(p)], device="cuda")
            out = ref(input_ids=ids, use_cache=False)
            ref_logits.append(out.logits[0, -1].float())
    ref_logits = torch.stack(ref_logits)
    diff = (logits - ref_logits).abs()
    top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    flat = diff.flatten()
    q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99, 1.0], device=flat.device))
    print(f"\nHF reference logits: max diff {diff.max().item():.4f} | "
          f"mean diff {diff.mean().item():.6f} | top-1 agree {100*top1:.1f}%")
    print(f"diff percentiles p50/p90/p99/max: {[f'{x:.4f}' for x in q.tolist()]}")
    print("(判定：top-1 100% = 端口正确；mean diff ~0.05-0.1 为 bf16+flash vs SDPA 正常偏差)")
    assert top1 == 1.0, "top-1 mismatch vs HF reference!"
    assert diff.mean().item() < 0.5, "mean logit diff too large vs HF reference!"
    del ref
    torch.cuda.empty_cache()
    print("PARITY OK")


if __name__ == "__main__":
    main()
