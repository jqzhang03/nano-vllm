"""DeepSeek-V2-Lite 4-layer real-weight parity and quantization drift probe.

The script checks each engine mode against the matching manual forward, then
compares INT4/NVFP4 logits with BF16 on identical fixed token prefixes. The
small prompt set is a smoke check, not a perplexity or task-quality evaluation.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL = os.path.expanduser("~/v2lite_4l")
V = 129280


def prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def _ids(t):
        return tok.encode(t, add_special_tokens=True)

    return [_ids("The Eiffel Tower was completed in 1889 in Paris as the entrance"
                 " to the World's Fair."),
            _ids("量子计算利用叠加与纠缠原理，有望在特定问题上超越经典计算机。")]


def build_nano_ref(quant: str):
    from transformers import AutoConfig
    from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(MODEL)
    model = DeepseekV2ForCausalLM(cfg).eval()
    load_model(model, MODEL)
    if cfg.tie_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
    model = model.to(torch.bfloat16).cuda()
    if quant in ("int4", "nvfp4"):
        from nanovllm.layers.linear import LinearBase
        for m in model.modules():
            if isinstance(m, LinearBase) and not getattr(m, "quantize_exclude", False):
                if quant == "int4":
                    # V2-Lite dense 中间维 10944 不能被 128 整除 → group 64
                    m.quantize_int4(dense_path=getattr(m, "is_mla_kv_b", False),
                                    group_size=64)
                else:
                    m.quantize_nvfp4(keep_dense=getattr(m, "is_mla_kv_b", False))
    return model


def nano_last_logits(model, tokens, T):
    """规范稠密参考（bf16 GPU，全手工/模块链）：前 T token 的末行 logits。"""
    import torch.nn.functional as F
    dev = "cuda"
    with torch.no_grad():
        ids = torch.tensor(tokens[:T], device=dev)
        x = model.model.embed_tokens(ids)
        pos = torch.arange(T, device=dev)
        r = None
        for n_l in model.model.layers:
            if r is None:
                s1, r = n_l.input_layernorm(x), x
            else:
                s1, r = n_l.input_layernorm(x, r)
            a = n_l.self_attn(pos, s1)
            s2, r = n_l.post_attention_layernorm(a, r)
            mlp = n_l.mlp
            if hasattr(mlp, "shared_experts"):
                y = mlp.reference(s2) + mlp.shared_experts(s2)
            else:
                y = mlp(s2)
            x = y
        fin, _ = model.model.norm(x, r)
        return F.linear(fin, model.lm_head.weight).float()[T - 1]


def fixed_input_logits(model, tokens):
    """Logits for identical one-token and full-prompt inputs, copied to CPU."""
    logits = {}
    for index, sequence in enumerate(tokens):
        for case, length in (("one_token", 1), ("full_prompt", len(sequence))):
            logits[f"seq{index}_{case}"] = nano_last_logits(
                model, sequence, length).detach().cpu()
    return logits


def quantization_drift(reference, candidate):
    """Aggregate same-input logit error, KL, and top-1 agreement."""
    import torch.nn.functional as F

    absolute_differences = []
    divergences = []
    top1_matches = 0
    for name, expected in reference.items():
        actual = candidate[name].float()
        expected = expected.float()
        absolute_differences.append((expected - actual).abs().flatten())
        logp = F.log_softmax(expected, dim=-1)
        logq = F.log_softmax(actual, dim=-1)
        divergences.append((logp.exp() * (logp - logq)).sum().item())
        top1_matches += int(expected.argmax() == actual.argmax())
    all_differences = torch.cat(absolute_differences)
    return {
        "input_cases": len(reference),
        "mean_abs_logit_diff": all_differences.mean().item(),
        "max_abs_logit_diff": all_differences.max().item(),
        "mean_kl_bf16_to_quantized": sum(divergences) / len(divergences),
        "top1_agreement_percent": 100.0 * top1_matches / len(reference),
    }


def main():
    import gc
    TOKENS = prompts()
    # ---- (1') 引擎 fp16 / int4 decode vs nano 手工参考（4 层真权重）----
    # 注：transformers 5.15 的 MoE grouped 路径调 torch._grouped_mm（仅 sm_90）——
    # 本机 sm_120 无法直接跑 HF-GPU 锚（CPU 慢路径另需加速库），如实记录；
    # 引擎数学链（含 MoE 路由/reference 同序）已在 toy 上对 HF 位级验证过。
    from nanovllm import LLM, SamplingParams
    engine = {}
    for quant in ("none", "int4", "nvfp4"):
        print(f"[engine quant={quant}]")
        llm = LLM(MODEL, max_model_len=512, quantization=quant, kv_swap=False,
                  enforce_eager=True, max_num_batched_tokens=1024,
                  gpu_memory_utilization=0.85,
                  int4_group_size=64)
        out = llm.generate(TOKENS, SamplingParams(temperature=0.7, max_tokens=24),
                           use_tqdm=False, collect_logits=True)
        dec = [lg for kind, lg in llm.collected_logits
               if kind == "decode" and lg is not None]
        comps = [o["token_ids"] for o in out]
        llm.exit()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        engine[quant] = (dec, comps)
    baseline_logits = None
    drift = {}
    for quant in ("none", "int4", "nvfp4"):
        ref = build_nano_ref(quant)
        fixed_logits = fixed_input_logits(ref, TOKENS)
        if quant == "none":
            baseline_logits = fixed_logits
        else:
            drift[quant] = quantization_drift(baseline_logits, fixed_logits)
            result = drift[quant]
            print(f"[{quant} vs BF16; identical fixed token inputs] "
                  f"mean_abs={result['mean_abs_logit_diff']:.6f} "
                  f"max_abs={result['max_abs_logit_diff']:.4f} "
                  f"KL={result['mean_kl_bf16_to_quantized']:.6f} "
                  f"top1={result['top1_agreement_percent']:.1f}%")

        dec, comps = engine[quant]
        bad = 0
        print(f"[{quant} engine decode vs same-mode manual reference]")
        for i, t in enumerate(TOKENS):
            so_far = t + comps[i]
            for step in range(min(len(comps[i]), len(dec))):
                lg = nano_last_logits(ref, so_far, len(t) + step + 1)
                e = dec[step][i].float()
                dd = (e - lg).abs()
                ok = int(e.argmax() == lg.argmax())
                bad += 0 if ok else 1
                if step % 6 == 0 or not ok:
                    print(f"  seq{i} step{step}: max {dd.max().item():.4f} "
                          f"mean {dd.mean().item():.5f} top1 {'Y' if ok else 'N'}")
        print(f"  mismatches {bad}")
        assert bad <= max(2, 48 // 10), f"{quant} 引擎 decode 对照失败"
        del ref
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    report = {
        "model": MODEL,
        "reference": "BF16, same fixed token prefixes",
        "input_cases": ["one-token prefix", "full prompt"],
        "quality_is_smoke_check": True,
        "quantization_drift": drift,
    }
    with open("results/v2lite_4l_quant_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("V2-LITE 4L REAL-WEIGHTS PARITY OK")


if __name__ == "__main__":
    main()
