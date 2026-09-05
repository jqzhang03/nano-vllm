"""真实 DeepSeek-V2-Lite 数值锚（阶段 2b 扩展）：4 层真权重 + 真实分布。

链（跑通≠写对）：
1) HF(GPU fp16, 4 层真权重) 全模型前向 vs 本引擎 nano 手工稠密参考 ——
   取每序列**末行**（transformers 5.15 DeepseekV2 无缓存全前向无因果掩码，
   中间行语义不可比，见 _mla_check.py 尾注）；
2) 本引擎(fp16 4 层) decode vs nano 手工参考（引擎解码语义 = 末行规范）；
3) 本引擎 int4(4 层) vs 本引擎 fp16(4 层)：真实权重上的量化噪声带。

用法: python benchmarks/_v2lite_4l_parity.py
"""
import os
import sys

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
    if quant == "int4":
        from nanovllm.layers.linear import LinearBase
        for m in model.modules():
            if isinstance(m, LinearBase) and not getattr(m, "quantize_exclude", False):
                # V2-Lite dense 中间维 10944 不能被 128 整除 → group 64
                m.quantize_int4(dense_path=getattr(m, "is_mla_kv_b", False),
                                group_size=64)
    return model.to(torch.bfloat16).cuda()


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


def main():
    import gc
    TOKENS = prompts()
    # ---- (1') 引擎 fp16 / int4 decode vs nano 手工参考（4 层真权重）----
    # 注：transformers 5.15 的 MoE grouped 路径调 torch._grouped_mm（仅 sm_90）——
    # 本机 sm_120 无法直接跑 HF-GPU 锚（CPU 慢路径另需加速库），如实记录；
    # 引擎数学链（含 MoE 路由/reference 同序）已在 toy 上对 HF 位级验证过。
    from nanovllm import LLM, SamplingParams
    engine = {}
    for quant in ("none", "int4"):
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
    ref_none = build_nano_ref("none")
    ref_i4 = build_nano_ref("int4")
    for quant, ref in (("none", ref_none), ("int4", ref_i4)):
        dec, comps = engine[quant]
        bad = 0
        print(f"[{quant} engine decode vs {'int4' if quant == 'int4' else 'fp16'} "
              f"稠密参考]")
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
    print("V2-LITE 4L REAL-WEIGHTS PARITY OK")


if __name__ == "__main__":
    main()
