"""DeepSeek-V2 端到端 parity：toy 目录（HF 5.15 save_pretrained 存盘格式）
→ 本引擎（MLA：decode 吸收式内核 / prefill flash 稠密）vs transformers。

对照口径（重要，见 _mla_check.py 尾注）：transformers 5.15 的 DeepseekV2 无缓存
全模型前向不产生因果掩码（create_causal_mask→None），中间行语义与规范不同；
只有**各行末行**（无未来 key）与缓存无关路径一致。因此：
- prefill：只比每序列最后一行（= 首个生成 token 的 logits）；
- decode：HF 增量（past_key_values，每步单 token 无因果问题）逐 token 对照。
用法: python benchmarks/_ds_e2e.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/ds_toy")
SRC_TOK = os.path.expanduser("~/huggingface/Qwen3-0.6B")
DUMP = "/tmp/_ds_e2e_logits.pt"


def build_toy():
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    if os.path.exists(TOY):
        shutil.rmtree(TOY)
    os.makedirs(TOY)
    cfg = DeepseekV2Config(
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=4,
        num_attention_heads=16,
        first_k_dense_replace=1,       # 层0 dense；层1-3 MoE
        kv_lora_rank=128,
        q_lora_rank=96,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        routed_scaling_factor=1.0,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
    )
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(7)
    model = DeepseekV2ForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(TOY)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(SRC_TOK, f), os.path.join(TOY, f))
    from safetensors import safe_open
    with safe_open(os.path.join(TOY, "model.safetensors"), "pt", "cpu") as f:
        keys = list(f.keys())
    ex = [k for k in keys if "experts" in k]
    print(f"toy -> {TOY} | {len(keys)} keys | expert keys sample:")
    for k in ex[:3]:
        print("   ", k)
    assert any(".experts.0.gate_proj.weight" in k for k in ex), \
        "存盘专家不是 2D per-expert 布局？"


def build_ref_model():
    """GPU bf16 稠密参考：nano 模块 + 手工注意力（无 flash/无 paged/无组装）——
    与引擎路径独立；模型级数学已由 CPU-HF 规范链位级验证（_mla_check.py）。
    tie 语义与 HF from_pretrained 一致：文件里 embed/lm_head 两份独立随机值，
    tie=True 时必须取 embed（先 tie 再 .to——.to 会破共享但两份副本同源同值）。"""
    from transformers import AutoConfig
    from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(TOY)
    model = DeepseekV2ForCausalLM(cfg).eval()
    load_model(model, TOY)
    if cfg.tie_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
    return model.to(torch.bfloat16).to("cuda")


def nano_dense_logits(model, tokens: list[int], T: int) -> torch.Tensor:
    """规范稠密参考（bf16 GPU）：tokens 前 T 个 → 末行 logits [V]。"""
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
            a = n_l.self_attn(pos, s1)          # 手工注意力（无 cache/无内核）
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
    build_toy()
    # toy 词表 32000 << Qwen 分词器词表 → 取模映射（随机权重 toy，语义无关）
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)
    _V = 32000

    def _ids(text: str) -> list[int]:
        return [t % _V for t in _tok.encode(text)]

    PROMPTS = ["The capital of France is", "Once upon a time in a faraway"]
    TOKENS = [_ids(p) for p in PROMPTS]
    WARM = [_ids("warm up")] * 2

    # ---------- 阶段1：本引擎 ----------
    if os.path.exists(DUMP) and os.environ.get("DS_E2E_SKIP_ENGINE"):
        engine = torch.load(DUMP)
        print("engine logits loaded (skip-engine)")
    else:
        from nanovllm import LLM, SamplingParams
        # MoE 动态路由不能进 CUDA graph（与 Qwen3-MoE 同款边界）→ eager
        llm = LLM(TOY, max_model_len=128, quantization="none", kv_swap=False,
                  enforce_eager=True)
        llm.generate(WARM,
                     SamplingParams(temperature=0.6, max_tokens=2),
                     use_tqdm=False)
        out = llm.generate([TOKENS[0], TOKENS[1]],
                           SamplingParams(temperature=0.6, max_tokens=3),
                           use_tqdm=False, collect_logits=True)
        pref = []
        dec = []
        for kind, lg in llm.collected_logits:
            if kind == "prefill" and lg is not None:
                pref.append(lg)
            elif kind == "decode" and lg is not None:
                dec.append(lg)
        completions = [o["token_ids"] for o in out]
        llm.exit()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        assert pref and dec, "缺 prefill/decode logits"
        engine = {"prefill": pref, "decode": dec,
                  "lens": [len(TOKENS[0]), len(TOKENS[1])],
                  "completions": completions}
        torch.save(engine, DUMP)
        print("engine done:", len(dec), "decode steps, completions",
              completions)

    # ---------- 阶段2：稠密参考（nano 模块手工注意力，GPU bf16） ----------
    # 模型级语义已由 _mla_check.py 的 CPU-HF 规范链位级验证；这里参考路径与
    # 引擎路径相互独立（无 flash / 无 paged 内核 / 无 cache 组装 / MoE 走
    # 全行掩码 reference）——专测引擎的缓存/内核/组装集成。
    ref = build_ref_model()
    comps = engine["completions"]
    n_step = len(engine["decode"])

    # (a) prefill：每序列末行
    ref_last = [nano_dense_logits(ref, TOKENS[i], len(TOKENS[i]))
                for i in range(len(TOKENS))]
    ref_last = torch.stack(ref_last)          # [n_seq, V]
    eng_pref = [lg for lg in engine["prefill"] if lg is not None]
    eng_cat = torch.cat(eng_pref).float()     # 引擎 prefill logits = 每序列末行
    print("[prefill events]", [tuple(x.shape) for x in eng_pref],
          "cat", tuple(eng_cat.shape), "lens", engine["lens"])
    eng_last = eng_cat
    if eng_last.shape[0] != len(ref_last):
        # 事件内多序列（顺序 = 输入序）；多于一轮事件则取末事件
        eng_last = eng_cat[-len(ref_last):]
    d = (eng_last - ref_last).abs()
    dX = (eng_last.flip(0) - ref_last).abs()
    top1 = (eng_last.argmax(-1) == ref_last.argmax(-1)).float().mean().item()
    topX = (eng_last.argmax(-1) == ref_last.flip(0).argmax(-1)).float().mean().item()
    print(f"[prefill last-row] max {d.max().item():.4f} | "
          f"mean {d.mean().item():.6f} | top-1 {100*top1:.1f}% | "
          f"cross-swap top-1 {100*topX:.1f}% mean {dX.mean().item():.6f}")
    assert top1 == 1.0, "prefill last-row top-1 mismatch"
    assert d.mean().item() < 0.5, "prefill 均值差过大"

    # (b) decode：用引擎真实 completion 流逐行重算末行对照
    print("[decode dense-ref]")
    bad = 0
    for i, base in enumerate(TOKENS):
        so_far = base + comps[i]
        for step in range(min(len(comps[i]), n_step)):
            T = len(base) + step + 1
            lg = nano_dense_logits(ref, so_far, T)
            e = engine["decode"][step][i].float()
            dd = (e - lg).abs()
            agree = int(e.argmax() == lg.argmax())
            print(f"  seq{i} step{step}: max {dd.max().item():.4f} "
                  f"mean {dd.mean().item():.6f} top1 {'Y' if agree else 'N'}")
            if not agree or dd.mean().item() > 0.5:
                bad += 1
    assert bad == 0, f"{bad} decode 步异常"
    print("DEEPSEEK-V2 E2E PARITY OK")


if __name__ == "__main__":
    main()
