"""gemma-2 交替 local/global 环滚动（split 模式，阶段 2b 扩展 M4，GPU）。

toy：6 层交替（sliding 开头的默认层型 → local/global 3+3），W=512，
attn_logit_softcapping=50（local 环层 decode 走自研 paged 内核 + 内核内
cap·tanh softcap；global 层走 full 池 flash 分页 + 原生 softcap）。

对照：
1) split 引擎（rolling_cache=True）vs 掩码引擎（False，flash 全 KV 窗口）：
   同种子同温度，共同前缀内逐 decode 行 top-1 / logits 差；
2) 采样步 vs 模块级手工稠密参考（含窗口掩码 + softcap，与引擎无关）。

用法: python benchmarks/_gemma2_ring_e2e.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

TOY = os.path.expanduser("~/gemma2_toy")
VOCAB = 8192
W = 512
B = 256
SOFTCAP = 50.0


def build_toy():
    from transformers import AutoTokenizer, Gemma2Config, Gemma2ForCausalLM
    if os.path.exists(TOY):
        shutil.rmtree(TOY)
    os.makedirs(TOY)
    cfg = Gemma2Config(
        vocab_size=VOCAB,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        sliding_window=W,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=64,
        attn_logit_softcapping=SOFTCAP,
        final_logit_softcapping=30.0,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(17)
    model = Gemma2ForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(TOY)
    src = os.path.expanduser("~/huggingface/Qwen3-0.6B")
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy2(os.path.join(src, f), os.path.join(TOY, f))
    tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)

    def _ids(text):
        return [t % VOCAB for t in tok.encode(text)]

    return [_ids("The capital of France is a city called Paris and it is famous"),
            _ids("Once upon a time, a small fox lived near a big river in a forest")]


def build_ref():
    """模块级稠密参考：gemma-2 手工注意力（窗口+softcap），无 flash/无缓存。"""
    from transformers import AutoConfig
    from nanovllm.models.gemma2 import Gemma2ForCausalLM
    from nanovllm.utils.loader import load_model
    cfg = AutoConfig.from_pretrained(TOY)
    model = Gemma2ForCausalLM(cfg).eval()
    load_model(model, TOY)
    if cfg.tie_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
    return model.to(torch.bfloat16).cuda()


def ref_logits(model, tokens, T):
    """tokens 前 T 个 → 末行 logits（手工因果+窗口+softcap）。"""
    import torch.nn.functional as F
    dev = "cuda"
    with torch.no_grad():
        ids = torch.tensor(tokens[:T], device=dev)
        pos = torch.arange(T, device=dev)
        x = model.model.embed_tokens(ids) * (model.config.hidden_size ** 0.5)
        for n_l in model.model.layers:
            attn = n_l.self_attn
            h = n_l.input_layernorm(x)
            # ---- 手工窗口注意力（含 softcap，与引擎无关）----
            qkv = attn.qkv_proj(h)
            hd = attn.head_dim
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q = q.view(T, attn.num_heads, hd)
            k = k.view(T, attn.num_kv_heads, hd)
            v = v.view(T, attn.num_kv_heads, hd)
            q, k = attn.rotary_emb(pos, q, k)
            g = attn.num_heads // attn.num_kv_heads
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
            s = torch.einsum("phd,khd->phk", q.float(), k.float()) * attn.scaling
            if attn.attn.logit_softcapping:
                cap = attn.attn.logit_softcapping
                s = cap * torch.tanh(s / cap)
            causal = pos[:, None] >= pos[None, :]
            if attn.attn.window_size:
                causal = causal & (pos[:, None] < pos[None, :] + attn.attn.window_size)
            p = s.masked_fill(~causal.unsqueeze(1), float("-inf")).softmax(-1)
            a = torch.einsum("phk,khd->phd", p, v.float()).to(h.dtype)
            o = attn.o_proj(a.reshape(T, -1))
            x1 = n_l.post_attention_layernorm(o) + x
            x = x1 + n_l.post_feedforward_layernorm(
                n_l.mlp(n_l.pre_feedforward_layernorm(x1)))
        fin = model.model.norm(x)
        logits = model.lm_head(fin).float()
        cap = model.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits[T - 1]


def run_engine(rolling: bool, prompts, seed):
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(seed)
    llm = LLM(TOY, max_model_len=1200, quantization="none", kv_swap=False,
              rolling_cache=rolling, enforce_eager=True,
              max_num_seqs=8, max_num_batched_tokens=4096)
    out = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=800),
                       use_tqdm=False, collect_logits=True)
    dec = [lg for kind, lg in llm.collected_logits
           if kind == "decode" and lg is not None]
    llm.exit()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return dec, [o["token_ids"] for o in out]


def cpl(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    PROMPTS = build_toy()
    plen = [len(p) for p in PROMPTS]
    print("[engine split(ring)]")
    dec_r, comp_r = run_engine(True, PROMPTS, 777)
    print("[engine masked]")
    dec_m, comp_m = run_engine(False, PROMPTS, 777)
    n = min(len(dec_r), len(dec_m))
    cpls = [cpl(comp_r[i], comp_m[i]) for i in range(len(comp_r))]
    agree_tot = agree_ok = flip = 0
    maxd = 0.0
    mean_sum = 0.0
    for s in range(n):
        a = dec_r[s].float()
        b = dec_m[s].float()
        for i in range(a.shape[0]):
            if s > cpls[i]:
                continue
            agree_tot += 1
            ok = int(a[i].argmax() == b[i].argmax())
            agree_ok += ok
            flip += 0 if ok else 1
            dd = (a[i] - b[i]).abs()
            maxd = max(maxd, dd.max().item())
            mean_sum += dd.mean().item()
    mean_d = mean_sum / max(agree_tot, 1)
    print(f"[split vs masked] 可比行 {agree_tot}: top-1 一致 "
          f"{100*agree_ok/max(agree_tot,1):.2f}% (翻转 {flip}) | "
          f"max {maxd:.4f} mean {mean_d:.5f}")
    assert flip <= max(2, agree_tot // 80), "split 与掩码对照翻转过多"

    # 与模块级手工参考对照（跨窗口边界加密采样）
    ref = build_ref()
    bad = total = 0
    for si in range(len(plen)):
        so_far = PROMPTS[si] + comp_r[si]
        last_ok = min(len(comp_r[si]) - 1, len(dec_r) - 1)
        steps = set(range(0, last_ok + 1, 53))
        for n_w in (W - 1, W, W + B - 1, W + B, 2 * W, 2 * W + B):
            s = n_w - plen[si] - 1
            if 0 <= s <= last_ok:
                steps.add(s)
        for s in sorted(steps):
            lg = ref_logits(ref, so_far, plen[si] + s + 1)
            e = dec_r[s][si].float()
            dd = (e - lg).abs()
            ok = int(e.argmax() == lg.argmax())
            total += 1
            bad += 0 if ok else 1
            print(f"seq{si} step{s:4d} L={plen[si]+s+1:4d}: max {dd.max().item():.3f} "
                  f"mean {dd.mean().item():.4f} top1 {'Y' if ok else 'N'}")
    print(f"[split vs manual ref] checked {total} steps, top1 mismatches {bad}")
    assert bad <= max(2, total // 60), "split 对照手工参考失败过多"
    print("GEMMA2 SPLIT-RING E2E OK")


if __name__ == "__main__":
    main()
