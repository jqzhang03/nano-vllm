"""MLA 纯 int4/fp8 权重 decode 高效路径（阶段 2b 扩展 M5，GPU toy）。

用**全 dense**（无 MoE 动态路由）DeepSeek toy（first_k_dense_replace=层数），
使 decode CUDA graph 捕获真正可行——验证流式纯 int4（streaming_load=True，
dense_path 强制 False）下 kv_b 反量化副本保留（is_mla_kv_b → w_deq）后：
  - _mla_dense_decode = False（不落稠密兜底）；enforce_eager 不被强制；
  - decode 吸收式内核路径 + CUDA graph 复盖（int4 权重 + graph 并行）；
  - decode logits vs bf16 引擎在 int4 噪声带内一致。

用法: python benchmarks/_mla_int4_probe.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/ds_dense_toy")
SRC_TOK = os.path.expanduser("~/huggingface/Qwen3-0.6B")
V = 32000


def build_toy():
    from transformers import AutoTokenizer, DeepseekV2Config, DeepseekV2ForCausalLM
    if os.path.exists(TOY):
        shutil.rmtree(TOY)
    os.makedirs(TOY)
    cfg = DeepseekV2Config(
        vocab_size=V, hidden_size=512, intermediate_size=1024,
        num_hidden_layers=4, num_attention_heads=16,
        first_k_dense_replace=4,          # 全 dense（无 MoE → decode 可入图）
        kv_lora_rank=128, q_lora_rank=128,   # ≥int4 group(128)（K%128 断言）
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=8, n_shared_experts=2, num_experts_per_tok=2,
        moe_intermediate_size=256, routed_scaling_factor=1.0,
        max_position_embeddings=512, rms_norm_eps=1e-6,
        attention_bias=False, mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
    )
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(7)
    model = DeepseekV2ForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(TOY)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(SRC_TOK, f), os.path.join(TOY, f))
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOY, use_fast=True)


def cpl(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    tok = build_toy()

    def _ids(t):
        return [x % V for x in tok.encode(t)]

    TOKENS = [_ids("The capital of France is a city called Paris which lies"),
              _ids("Once upon a time, in a small village, a fox met a crow")]
    WARM = [_ids("warm")] * 2

    def run_engine(int4: bool, seed: int):
        from nanovllm import LLM, SamplingParams
        torch.manual_seed(seed)
        kwargs = dict(max_model_len=160, quantization="none", kv_swap=False,
                      enforce_eager=False, max_num_batched_tokens=4096)
        if int4:
            kwargs.update(quantization="int4", streaming_load=True)
        llm = LLM(TOY, **kwargs)
        mr = llm.model_runner
        state = (bool(getattr(mr, "_mla_dense_decode", None)),
                 bool(mr.enforce_eager),
                 getattr(mr.model.model.layers[0].self_attn.kv_b_proj, "w_deq",
                         None) is not None,
                 hasattr(mr, "graphs"))
        llm.generate(WARM, SamplingParams(temperature=0.6, max_tokens=2),
                     use_tqdm=False)
        out = llm.generate(TOKENS, SamplingParams(temperature=0.6, max_tokens=12),
                           use_tqdm=False, collect_logits=True)
        dec = [lg for kind, lg in llm.collected_logits
               if kind == "decode" and lg is not None]
        llm.exit()
        import gc
        del llm, mr
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return state, dec, [o["token_ids"] for o in out]

    s_i4, dec_i4, comp_i4 = run_engine(True, 55)
    print(f"[int4 状态] dense_decode={s_i4[0]} enforce_eager被强制={s_i4[1]} "
          f"kv_b w_deq={'有' if s_i4[2] else '无'} decode图={'有' if s_i4[3] else '无'}")
    assert not s_i4[0] and s_i4[2], "int4 MLA 仍落稠密兜底/缺 kv_b w_deq"
    assert not s_i4[1], "int4 MLA 被强制 eager（应保留 graph）"
    assert s_i4[3], "全 dense toy 应能捕获 decode 图"
    # 语义对照：随机权重模型上 int4-vs-bf16 无意义（混沌）；正确参考 = **同量化**
    # 稠密手工模型（与引擎量化参数一致的 int4 权重；吸收式内核读 w_deq =
    # int4 反量化 → 应与其 dense 路径在噪声带内一致）。
    from transformers import AutoConfig
    from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(TOY)
    ref = DeepseekV2ForCausalLM(cfg).eval()
    load_model(ref, TOY)
    if cfg.tie_word_embeddings:
        ref.lm_head.weight.data = ref.model.embed_tokens.weight.data
    from nanovllm.layers.linear import LinearBase
    for m in ref.modules():
        if isinstance(m, LinearBase) and not getattr(m, "quantize_exclude", False):
            m.quantize_int4(dense_path=getattr(m, "is_mla_kv_b", False))
    ref = ref.to(torch.bfloat16).cuda()

    def ref_logits(tokens, T):
        import torch.nn.functional as F
        with torch.no_grad():
            ids = torch.tensor(tokens[:T], device="cuda")
            pos = torch.arange(T, device="cuda")
            x = ref.model.embed_tokens(ids)
            r = None
            for n_l in ref.model.layers:
                if r is None:
                    s1, r = n_l.input_layernorm(x), x
                else:
                    s1, r = n_l.input_layernorm(x, r)
                a = n_l.self_attn(pos, s1)
                s2, r = n_l.post_attention_layernorm(a, r)
                y = n_l.mlp(s2)
                x = y
            fin, _ = ref.model.norm(x, r)
            return F.linear(fin, ref.lm_head.weight).float()[T - 1]

    bad = total = 0
    maxd = 0.0
    mean_sum = 0.0
    for i, t in enumerate(TOKENS):
        so_far = t + comp_i4[i]
        for step in range(min(len(comp_i4[i]), len(dec_i4))):
            T = len(t) + step + 1
            lg = ref_logits(so_far, T)
            e = dec_i4[step][i].float()
            dd = (e - lg).abs()
            ok = int(e.argmax() == lg.argmax())
            total += 1
            bad += 0 if ok else 1
            maxd = max(maxd, dd.max().item())
            mean_sum += dd.mean().item()
    print(f"[int4 引擎 decode vs int4 稠密参考] {total} 行: top-1 失配 {bad}"
          f" | max {maxd:.4f} mean {mean_sum/max(total,1):.5f}")
    assert bad <= max(2, total // 20), "int4 引擎 vs 同量化参考失配过多"
    print("MLA INT4 ABSORBED-DECODE (+GRAPH) OK")


if __name__ == "__main__":
    main()
