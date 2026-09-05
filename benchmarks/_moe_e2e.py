"""Qwen3-MoE 端到端 parity：随机权重 toy 模型目录 → 本引擎 vs transformers 5.15。

覆盖：模型适配（qwen3_moe 注册 + DecoderLayer 混合 dense/MoE + loader 段匹配 +
3D 专家权重装载 + tie 词表）与 MoE 层语义（与 transformers 同权重同 dtype 对照）。

两阶段（同 _parity.py 惯例）：引擎先跑（bf16）→ exit → HF 在 GPU 对照。
判定：prefill logits top-1 100% + mean diff 在 bf16/flash vs SDPA 的正常偏差内
（MoE 另受路由离散性影响：同 dtype 同权重下 topk 应一致，除非概率贴边翻转）。

用法: python benchmarks/_moe_e2e.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/moe_toy")
PROMPTS = ["The capital of France is", "Once upon a time"]
SRC_TOK = os.path.expanduser("~/huggingface/Qwen3-0.6B")


def build_toy(segment: bool = False):
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    if os.path.exists(TOY):
        shutil.rmtree(TOY)
    os.makedirs(TOY)
    cfg = Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=512,
        num_hidden_layers=4,          # 层0,2 dense；层1,3 MoE（decoder_sparse_step=2 混合）
        num_attention_heads=8,
        num_key_value_heads=4,        # head_dim = 64
        intermediate_size=1024,       # dense 层
        moe_intermediate_size=768,
        num_experts=8,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        norm_topk_prob=False,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    if segment:
        cfg.moe_segment_backend = True  # Triton 真段式后端（引擎 parity 复验用）
    cfg.torch_dtype = torch.bfloat16
    model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(TOY)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(SRC_TOK, f), os.path.join(TOY, f))
    print(f"toy model -> {TOY} "
          f"({os.path.getsize(os.path.join(TOY, 'model.safetensors')) / 1e6:.0f} MB)")


def main():
    build_toy(segment=True)
    dump = "/tmp/_moe_e2e_logits.pt"

    # ---- 阶段1：本引擎（bf16，不量化，MoE 动态 gather → enforce_eager） ----
    if os.path.exists(dump) and os.environ.get("MOE_E2E_SKIP_ENGINE"):
        logits = torch.load(dump)
        print(f"engine logits loaded from {dump} (skip-engine)")
    else:
        from nanovllm import LLM, SamplingParams
        llm = LLM(TOY, max_model_len=128, quantization="none", kv_swap=False,
                  enforce_eager=True)
        hf = llm.config.hf_config
        n_moe = sum(1 for i in range(hf.num_hidden_layers)
                    if (i not in getattr(hf, "mlp_only_layers", ()))
                    and (i + 1) % getattr(hf, "decoder_sparse_step", 1) == 0)
        print(f"\n=== engine model_type={hf.model_type} layers={hf.num_hidden_layers} "
              f"(moe layers={n_moe}) ===")
        llm.generate(["warm up"] * 2, SamplingParams(temperature=0.6, max_tokens=2),
                     use_tqdm=False)
        llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=1),
                     use_tqdm=False, collect_logits=True)
        logits = None
        for kind, lg in llm.collected_logits:
            if kind == "prefill" and lg is not None:
                logits = lg
                break
        llm.exit()
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        assert logits is not None, "no prefill logits"
        torch.save(logits, dump)
        print(f"engine logits saved, shape {tuple(logits.shape)}")

    # ---- 阶段2：HF 参考（同目录同权重，bf16 GPU） ----
    # sm_120 无 torch._grouped_mm（sm_90 only）→ 强制 transformers 走 eager 逐专家
    # 循环实现（"eager" 不在注册表 → get_interface 回退原始 forward）
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)
    ref = AutoModelForCausalLM.from_pretrained(TOY, dtype=torch.bfloat16).to("cuda").eval()
    ref.config._experts_implementation = "eager"
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
    print(f"\nHF reference: max diff {diff.max().item():.4f} | "
          f"mean diff {diff.mean().item():.6f} | top-1 agree {100 * top1:.1f}%")
    print("diff percentiles p50/p90/p99/max: "
          + str([f"{x:.4f}" for x in q.tolist()]))
    assert top1 == 1.0, "top-1 mismatch vs HF!"
    assert diff.mean().item() < 0.5, "mean logit diff too large!"
    print("MOE PARITY OK")


if __name__ == "__main__":
    main()
