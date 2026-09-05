"""阶段 2a 验证脚本（跑通≠写对）：
(1) GPU：MLA decode 吸收式内核 vs 稠密展开参考（同权重同缓存，含 rope）
(2) CPU：本实现 DeepseekV2ForCausalLM vs transformers 5.15 同款类——**逐层
    显式因果掩码的规范参考链**（HF 5.15 该模块的无缓存全模型前向不传因果掩码，
    见文件尾注；逐层直调 + 标准下三角掩码才是规范语义）
用法: python benchmarks/_mla_check.py [--stage kernel|cpu|both]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

torch.manual_seed(0)


def _toy_cfg():
    from transformers import DeepseekV2Config
    return DeepseekV2Config(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,           # 层0 dense；层1 MoE（first_k_dense_replace=1）
        num_attention_heads=16,
        first_k_dense_replace=1,
        kv_lora_rank=128,
        q_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        n_routed_experts=8,
        n_shared_experts=2,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        routed_scaling_factor=1.0,
        topk_method="greedy",
        norm_topk_prob=False,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=True,
        torch_dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# (1) decode 内核：构造随机 MLA 层 → S token 缓存 → 末 token decode
# ---------------------------------------------------------------------------
def stage_kernel():
    from nanovllm.layers.attention_mla import MLAAttention, mla_store, \
        mla_decode_attention

    torch.manual_seed(1)
    dev = "cuda"
    B = 256
    H = 256
    cfg = dict(hidden_size=H, num_heads=16, q_lora_rank=64, kv_lora_rank=128,
               qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
               max_position=2048, rope_theta=10000.0, rms_norm_eps=1e-6)
    attn = MLAAttention(**cfg).to(torch.bfloat16).to(dev).eval()
    with torch.no_grad():
        for p in attn.parameters():
            if p.ndim >= 2:
                p.uniform_(-0.1, 0.1)

    for S in (1, 130, 600):      # 跨 0/1/3 块
        nb = (S + B - 1) // B
        cache = torch.zeros(nb, B, 128 + 16, device=dev,
                            dtype=torch.bfloat16)
        x = torch.randn(S, H, device=dev, dtype=torch.bfloat16) * 0.5
        positions = torch.arange(S, device=dev)
        with torch.no_grad():
            q_nope, q_pe_raw, c, k_raw = attn._project(x)
            q_pe_r, k_pe_r = attn.rotary_emb(positions, q_pe_raw, k_raw)
        slot_map = torch.arange(S, dtype=torch.int32, device=dev)
        mla_store(c, k_pe_r, cache, slot_map)
        # ---- 稠密参考（缓存内容 → 逐头展开 → 末行注意力）----
        flat = cache.reshape(-1, 128 + 16)[:S]
        gc, gk = flat.split([128, 16], dim=-1)
        with torch.no_grad():
            k_head, v = attn._expand(gc, gk, S)
            q_full = torch.cat([q_nope, q_pe_r], dim=-1)      # [S, h, 48]
            s = torch.einsum("phd,khd->phk", q_full.float(),
                             k_head.float())
            s = s * attn.scaling
            causal = torch.tril(torch.ones(S, S, dtype=torch.bool,
                                           device=dev)).unsqueeze(1)
            s = s.masked_fill(~causal, float("-inf"))
            p_row = s[-1].softmax(dim=-1)                     # [h, S]
            ref = torch.einsum("hk,khd->hd", p_row.float(),
                               v.float())                    # [h, v]
        # ---- 内核路径 ----
        q_nope_d = q_nope[-1:].contiguous()
        q_pe_d = q_pe_r[-1:].contiguous()
        w3 = attn.kv_b_proj.weight.detach().reshape(16, 32 + 32, 128)
        wuk = w3[:, :32, :].contiguous()
        wuv = w3[:, 32:, :].contiguous()
        q_abs = torch.einsum("bhj,hjk->bhk", q_nope_d.to(wuk.dtype), wuk) \
            .to(torch.float16)
        o_c = mla_decode_attention(q_abs, q_pe_d.to(torch.float16), cache,
                                   torch.tensor([list(range(nb))],
                                                dtype=torch.int32,
                                                device=dev),
                                   torch.tensor([S], dtype=torch.int32,
                                                device=dev),
                                   attn.scaling)
        out = torch.einsum("bhk,hvk->bhv", o_c.to(torch.float32),
                           wuv.float())[0]
        err = (out.float() - ref).abs()
        rel = err.max() / (ref.abs().max() + 1e-6)
        print(f"[kernel] S={S:4d}: abs_max={err.max().item():.4f} "
              f"rel_max={rel.item():.5f}")
        assert rel.item() < 0.02, "MLA decode kernel vs dense reference 超差!"
    print("(1) decode kernel OK")


# ---------------------------------------------------------------------------
# (2) CPU 全模型 parity vs transformers 5.15
# ---------------------------------------------------------------------------
def _map_hf_sd(sd: dict, cfg) -> dict:
    """transformers 5.15 state_dict（experts 内存 3D 直挂 Parameter：键无 .weight
    后缀）→ 本实现 2D per-expert 键（gate/experts 段名本身直配）。"""
    I = cfg.moe_intermediate_size
    out = {}
    gu3 = {}
    dn3 = {}
    for k, v in sd.items():
        if k.endswith(".experts.gate_up_proj"):
            gu3[k] = v
        elif k.endswith(".experts.down_proj"):
            dn3[k] = v
        else:
            out[k] = v
    for k, gu in gu3.items():          # [E, 2I, H]
        pre = k[: -len(".experts.gate_up_proj")]
        for e in range(gu.shape[0]):
            out[f"{pre}.experts.{e}.gate_proj.weight"] = gu[e, :I].contiguous()
            out[f"{pre}.experts.{e}.up_proj.weight"] = gu[e, I:].contiguous()
    for k, w in dn3.items():           # [E, H, I]
        pre = k[: -len(".experts.down_proj")]
        for e in range(w.shape[0]):
            out[f"{pre}.experts.{e}.down_proj.weight"] = w[e].contiguous()
    return out


def hf_canonical_logits(hf, ids: torch.Tensor) -> torch.Tensor:
    """HF 模块的规范逐层参考：显式下三角因果掩码 + 逐层调用（其无缓存全模型
    前向不传掩码，rows≥1 语义与规范不同——见文件尾注；这里逐层直调复原）。

    返回 [T, V] logits（fp32 计算链与 nano 同构：embed → 层循环 → norm → head）。
    """
    T = ids.numel()
    pos = torch.arange(T, device=ids.device)[None]
    am = torch.zeros(1, 1, T, T, device=ids.device)
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool, device=ids.device))
    am.masked_fill_(~tril.unsqueeze(0).unsqueeze(0), float("-inf"))
    with torch.no_grad():
        emb = hf.model.embed_tokens(ids[None])
        freqs = hf.model.rotary_emb(emb, pos)
        h = emb
        for layer in hf.model.layers:
            h_ln = layer.input_layernorm(h)
            a, _ = layer.self_attn(hidden_states=h_ln, attention_mask=am,
                                   position_embeddings=freqs)
            h = h + a
            h_ln2 = layer.post_attention_layernorm(h)
            y = layer.mlp(h_ln2)
            h = h + y
        hn = hf.model.norm(h)
        return hf.lm_head(hn)[0].float()


def _canonical_debug(hf, nano, ids):
    """规范链 vs nano 逐层对照（定位首发散层/子层）。"""
    T = ids.numel()
    pos = torch.arange(T)
    am = torch.zeros(1, 1, T, T)
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
    am.masked_fill_(~tril.unsqueeze(0).unsqueeze(0), float("-inf"))
    with torch.no_grad():
        emb = hf.model.embed_tokens(ids[None])
        freqs = hf.model.rotary_emb(emb, pos[None])
        h_hf = emb
        x_n, r_n = None, None
        h_in = nano.model.embed_tokens(ids)
        for i, layer in enumerate(hf.model.layers):
            # hf canonical 层
            h_ln = layer.input_layernorm(h_hf)
            a_hf, _ = layer.self_attn(hidden_states=h_ln, attention_mask=am,
                                      position_embeddings=freqs)
            h_hf = h_hf + a_hf
            m_in = layer.post_attention_layernorm(h_hf)
            y_hf = layer.mlp(m_in)
            h_hf = h_hf + y_hf
            # nano 层
            n_l = nano.model.layers[i]
            if x_n is None:
                s1, r_n = n_l.input_layernorm(h_in), h_in
            else:
                s1, r_n = n_l.input_layernorm(x_n, r_n)
            if i == 0:
                print("[ln diff]", (h_ln[0] - s1).abs().max().item())
                print("[emb diff]", (emb[0] - h_in).abs().max().item())
                import torch.nn.functional as F
                d0 = F.embedding(ids, nano.model.embed_tokens.weight)
                h_in2 = nano.model.embed_tokens(ids)
                print("[module-vs-fresh-module]", (h_in - h_in2).abs().max().item(),
                      "| fresh-module-vs-F]", (h_in2 - d0).abs().max().item(),
                      "| fresh-vs-hf]", (h_in2 - emb[0]).abs().max().item())
                print("[ids]", ids[:4].tolist())
                print("[emb0 hf]", emb[0, 0, :4].tolist(),
                      "| nano", h_in[0, :4].tolist())
                print("[w hf/nano]", hf.model.embed_tokens.weight[ids[0],
                                                                   :4].tolist(),
                      nano.model.embed_tokens.weight[ids[0], :4].tolist())
            a_n = n_l.self_attn(pos, s1)
            if i == 0:
                f1 = hf.model.rotary_emb(h_ln, pos[None])
                print("[freqs diff]", (f1 - freqs).abs().max().item())
                a_x, _ = layer.self_attn(hidden_states=s1[None],
                                         attention_mask=am,
                                         position_embeddings=freqs)
                print("[a_hf vs a_hf(s1)]", (a_hf - a_x).abs().max().item())
                print("[a_hf(s1) vs a_n]",
                      (a_x[0] - a_n).abs().max().item())
            da = (a_hf[0] - a_n).abs().max().item()   # post 前对比（post 会原位加）
            s2, r_n = n_l.post_attention_layernorm(a_n, r_n)
            y_n = n_l.mlp(s2)
            x_n = y_n
            net_n = x_n + r_n
            dnet = (h_hf[0] - net_n).abs().max().item()
            dy = (y_hf[0] - y_n).abs().max().item()
            if i == 0:
                print("[m_in diff]", (m_in[0] - s2).abs().max().item(),
                      "| r diff", (m_in[0] - r_n).abs().max().item())
            print(f"layer{i} attn {da:.6f} | mlp {dy:.6f} | net {dnet:.6f}")
        hn = hf.model.norm(h_hf)
        nn_, rr = nano.model.norm(x_n, r_n)
        dlog = (hf.lm_head(hn)[0] - nano.compute_logits(nn_)).abs()
        print("final norm/head max diff:", dlog.max().item())


def stage_cpu():
    import transformers
    print("transformers", transformers.__version__)
    from transformers import DeepseekV2ForCausalLM
    from nanovllm.models.deepseek_v2 import DeepseekV2ForCausalLM as NanoV2

    cfg = _toy_cfg()
    torch.manual_seed(3)
    hf = DeepseekV2ForCausalLM(cfg).to(torch.float32).eval()
    cfg.torch_dtype = torch.float32
    nano = NanoV2(cfg).to(torch.float32).eval()
    sd = _map_hf_sd(hf.state_dict(), cfg)
    missing = nano.load_state_dict(sd, strict=False)
    print("missing(unexpected):", len(missing.missing_keys),
          len(missing.unexpected_keys))
    assert not missing.unexpected_keys, missing.unexpected_keys
    print("[weight checks] embed",
          (nano.model.embed_tokens.weight
           - hf.model.embed_tokens.weight).abs().max().item(),
          "| lm_head weight tied:", nano.lm_head.weight.data_ptr()
          == nano.model.embed_tokens.weight.data_ptr())

    for trial in range(3):
        torch.manual_seed(5 + trial)
        ids = torch.randint(0, cfg.vocab_size, (17 + trial * 3,))
        if os.environ.get("DS_CANDEBUG"):
            _canonical_debug(hf, nano, ids)
            return
        with torch.no_grad():
            h1 = hf_canonical_logits(hf, ids)
            h2 = nano.compute_logits(
                nano(ids, torch.arange(ids.numel()))).float()
        d = (h1 - h2).abs()
        top1 = (h1.argmax(-1) == h2.argmax(-1)).float().mean().item()
        print(f"[cpu parity] trial{trial} T={ids.numel()}: "
              f"max {d.max().item():.6f} | mean {d.mean().item():.8f} | "
              f"top-1 {100*top1:.2f}%")
        assert d.mean().item() < 1e-4, "CPU parity mean diff 过大"
        assert top1 == 1.0, "top-1 一致性不足"
    print("(2) CPU model parity vs transformers OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both", choices=["kernel", "cpu", "both"])
    a = ap.parse_args()
    if a.stage in ("kernel", "both"):
        stage_kernel()
    if a.stage in ("cpu", "both"):
        stage_cpu()
    print("MLA CHECK DONE")


if __name__ == "__main__":
    main()
