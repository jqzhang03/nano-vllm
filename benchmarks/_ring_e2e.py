"""SWA 滚动缓冲端到端（阶段 2b，GPU）：mistral toy（窗口 W=512 < 解码长度，
多轮跨窗）+ rolling_cache=True。

对照：逐（采样）步稠密掩码参考——nano 模块手工注意力（Q/K/V 全展开 +
逐行因果 + 窗口掩码），bf16 GPU；与引擎的 flash/paged 无关，专测引擎的
环驱逐/表维护/内核位置还原。

判定：引擎 decode logits vs 参考 top-1 一致、mean diff 小；跨窗边界附近
加密采样（N ≈ W、W+B、2W、2W+B ± 2）。

用法: python benchmarks/_ring_e2e.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/mistral_toy")
SRC_TOK = os.path.expanduser("~/huggingface/Qwen3-0.6B")
DUMP = "/tmp/_ring_e2e_logits.pt"
W = 512          # 滑动窗口
B = 256          # 块大小（引擎断言 256 倍数）
VOCAB = 2048


def build_toy():
    from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM
    if os.path.exists(TOY):
        shutil.rmtree(TOY)
    os.makedirs(TOY)
    cfg = MistralConfig(
        vocab_size=VOCAB,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        sliding_window=W,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
        torch_dtype=torch.bfloat16,
    )
    torch.manual_seed(11)
    model = MistralForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(TOY)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(SRC_TOK, f), os.path.join(TOY, f))
    tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)

    def _ids(text):
        return [t % VOCAB for t in tok.encode(text)]

    return [_ids("The capital of France is a city"), _ids("Once upon a time")]


def masked_ref_logits(model, tokens: list[int], T: int) -> torch.Tensor:
    """稠密掩码参考：ctx 前 T token，SWA 因果窗口注意力，末行 logits。"""
    import torch.nn.functional as F
    dev = "cuda"
    h = model.config.num_attention_heads
    kvh = model.config.num_key_value_heads
    hd = model.config.head_dim
    with torch.no_grad():
        ids = torch.tensor(tokens[:T], device=dev)
        pos = torch.arange(T, device=dev)
        x = model.model.embed_tokens(ids)
        r = None
        for n_l in model.model.layers:
            if r is None:
                s1, r = n_l.input_layernorm(x), x
            else:
                s1, r = n_l.input_layernorm(x, r)
            # ---- 手工窗口注意力（与引擎实现无关）----
            qkv = n_l.self_attn.qkv_proj(s1)
            q, k, v = qkv.split([h * hd, kvh * hd, kvh * hd], dim=-1)
            q = q.view(T, h, hd)
            k = k.view(T, kvh, hd)
            v = v.view(T, kvh, hd)
            q, k = n_l.self_attn.rotary_emb(pos, q, k)
            v = v.repeat_interleave(h // kvh, dim=1)
            k = k.repeat_interleave(h // kvh, dim=1)
            sc = torch.einsum("phd,khd->phk", q.float(), k.float())
            sc = sc * (hd ** -0.5)
            causal = pos[:, None] >= pos[None, :]                 # [p,k]
            if W:
                causal = causal & (pos[:, None] < pos[None, :] + W)
            p = sc.masked_fill(~causal.unsqueeze(1), float("-inf")).softmax(-1)
            a = torch.einsum("phk,khd->phd", p, v.float()).to(s1.dtype)
            a = n_l.self_attn.o_proj(a.reshape(T, -1))
            s2, r = n_l.post_attention_layernorm(a, r)
            y = n_l.mlp(s2)
            x = y
        fin, _ = model.model.norm(x, r)
        return F.linear(fin, model.lm_head.weight).float()[T - 1]


def main():
    PROMPTS = build_toy()
    comps = None
    if os.path.exists(DUMP) and os.environ.get("RING_SKIP_ENGINE"):
        engine = torch.load(DUMP)
    else:
        from nanovllm import LLM, SamplingParams
        llm = LLM(TOY, max_model_len=1500, quantization="none", kv_swap=False,
                  rolling_cache=True, max_num_seqs=8, max_num_batched_tokens=4096)
        assert llm.scheduler.block_manager.rolling
        out = llm.generate(PROMPTS, SamplingParams(temperature=0.6,
                                                   max_tokens=1100),
                           use_tqdm=False, collect_logits=True)
        dec = [lg for kind, lg in llm.collected_logits
               if kind == "decode" and lg is not None]
        llm.exit()
        comps = [o["token_ids"] for o in out]
        print("decode steps:", len(dec), "| completions lens",
              [len(c) for c in comps])
        engine = {"decode": dec, "completions": comps,
                  "plen": [len(p) for p in PROMPTS]}
        torch.save(engine, DUMP)
    comps = engine["completions"]
    plen = engine["plen"]

    # 参考模型（独立实例，loader 同源权重）
    from transformers import AutoConfig
    from nanovllm.models.mistral import MistralForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(TOY)
    ref = MistralForCausalLM(cfg).eval()
    load_model(ref, TOY)
    ref = ref.to(torch.bfloat16).cuda()
    if cfg.tie_word_embeddings:
        ref.lm_head.weight.data = ref.model.embed_tokens.weight.data

    # 采样步：常规稀疏 + 跨窗边界加密
    bad = 0
    total = 0
    for si in range(len(plen)):
        so_far = PROMPTS[si] + comps[si]
        base_len = plen[si]
        n_comp = len(comps[si])
        steps = set(range(0, n_comp, 37))
        last_ok = min(n_comp - 1, len(engine["decode"]) - 1)
        for n_w in (W - 2, W - 1, W, W + 1, W + 2,
                    W + B - 1, W + B, W + B + 1,
                    2 * W - 1, 2 * W, 2 * W + 1):
            s = n_w - base_len - 1
            if 0 <= s <= last_ok:
                steps.add(s)
        if last_ok >= 0:
            steps.add(last_ok)
        for s in sorted(steps):
            ctx_len = base_len + s + 1
            lg = masked_ref_logits(ref, so_far, ctx_len)
            e = engine["decode"][s][si].float()
            dd = (e - lg).abs()
            agree = int(e.argmax() == lg.argmax())
            total += 1
            if not agree:
                bad += 1
            flag = "Y" if agree else "N"
            if s % 111 == 0 or (not agree):
                print(f"seq{si} step{s:4d} L={ctx_len:4d}: "
                      f"max {dd.max().item():.4f} mean {dd.mean().item():.5f} "
                      f"top1 {flag}")
    print(f"checked {total} steps, top1 mismatches {bad}")
    assert bad <= max(2, total // 100), "滚动 decode 对照失败过多"
    print("RING E2E OK")


if __name__ == "__main__":
    main()
