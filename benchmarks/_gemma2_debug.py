"""Gemma-2 parity 失败的逐层定位：本引擎 vs HF（GPU bf16）逐层 hidden 对比。

流程：引擎跑 prefill（fp16）抓 embed/层0/层1/层N/最终 hidden/logits → 退出 →
HF 加载跑同一 prompt → 逐层对比 max diff，找发散起点。
"""
import os
import sys

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

PROMPT = "The capital of France is"


def main():
    model = os.path.expanduser(sys.argv[1])
    llm = LLM(model, max_model_len=256, quantization="none", kv_swap=False)
    tok = llm.tokenizer
    ids = tok.encode(PROMPT)

    # ---- 引擎前向（直连，不经调度器） ----
    bm = llm.scheduler.block_manager
    seq = Sequence(ids)
    seq.num_scheduled_tokens = len(ids)
    bm.allocate(seq, bm.can_allocate(seq))
    ours = {}
    hooks = []
    # 引擎：embed 输出 [T, H]（直接抓）；decoder layer 返回 (hidden, residual) tuple → o[0]
    hooks.append(llm.model_runner.model.model.embed_tokens.register_forward_hook(
        lambda _m, _i, o: ours.__setitem__("embed", o.detach().clone())))
    for name, idx in (("layer0", 0), ("layer1", 1)):
        hooks.append(llm.model_runner.model.model.layers[idx].register_forward_hook(
            lambda _m, _i, o, name=name: ours.__setitem__(name, o[0].detach().clone())))
    # 层0 子模块二分：attn 输出 / post_attention_layernorm 输出 / mlp 输出
    l0 = llm.model_runner.model.model.layers[0]
    hooks.append(l0.self_attn.register_forward_hook(
        lambda _m, _i, o: ours.__setitem__("l0_attn", o.detach().clone())))
    hooks.append(l0.post_attention_layernorm.register_forward_hook(
        lambda _m, _i, o: ours.__setitem__("l0_pa", o.detach().clone())))
    hooks.append(l0.mlp.register_forward_hook(
        lambda _m, _i, o: ours.__setitem__("l0_mlp", o.detach().clone())))
    # 层0 注意力内部：q/k/v（rope 后）与 o_proj 输入
    attn0 = l0.self_attn
    rec = {}

    def rec_attn_fwd(positions, hidden_states):
        rec["ln1_out"] = hidden_states.detach().clone()  # 注意力输入 = ln1(x0)
        qkv = attn0.qkv_proj(hidden_states)
        q, k, v = qkv.split([attn0.q_size, attn0.kv_size, attn0.kv_size], dim=-1)
        q = q.view(-1, attn0.num_heads, attn0.head_dim)
        k = k.view(-1, attn0.num_kv_heads, attn0.head_dim)
        v = v.view(-1, attn0.num_kv_heads, attn0.head_dim)
        rec["qkv_pre"] = qkv.detach().clone()
        q, k = attn0.rotary_emb(positions, q, k)
        rec["q_rope"] = q.detach().clone()
        rec["k_rope"] = k.detach().clone()
        o = attn0.attn(q, k, v)
        rec["o_attn"] = o.detach().clone()
        return attn0.o_proj(o.flatten(1, -1))

    orig_attn_fwd = attn0.forward
    attn0.forward = rec_attn_fwd
    llm.model_runner.call("run", [seq], "prefill", True, False)
    attn0.forward = orig_attn_fwd
    ours.update({f"l0_{k}": v for k, v in rec.items()})
    for h in hooks:
        h.remove()
    for s in [seq]:
        bm.deallocate(s)
    # 最终 hidden + logits：走引擎 run（return_hidden=True），context 由 runner 设置
    bm.allocate(seq, bm.can_allocate(seq))
    _tokens, logits, hidden = llm.model_runner.call("run", [seq], "prefill", True, True)
    for s in [seq]:
        bm.deallocate(s)
    ours["hidden"] = hidden[0].detach().clone()
    ours["logits"] = logits[0].detach().clone().float()
    ours["_w_embed"] = llm.model_runner.model.model.embed_tokens.weight.data.detach().clone()
    llm.exit()
    torch.cuda.empty_cache()

    # ---- HF 参考 ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok2 = AutoTokenizer.from_pretrained(model, use_fast=True)
    ref_m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to("cuda").eval()
    ref = {}
    hooks = []
    # HF：embed/层输出都是 [1, T, H] → o[0]
    hooks.append(ref_m.model.embed_tokens.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("embed", o[0].detach().clone())))
    for name, idx in (("layer0", 0), ("layer1", 1), ("layer_last", -1)):
        hooks.append(ref_m.model.layers[idx].register_forward_hook(
            lambda _m, _i, o, name=name: ref.__setitem__(name, o[0].detach().clone())))
    rl0 = ref_m.model.layers[0]
    hooks.append(rl0.self_attn.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("l0_attn", o[0][0].detach().clone())))
    hooks.append(rl0.post_attention_layernorm.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("l0_pa", o[0].detach().clone())))
    hooks.append(rl0.mlp.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("l0_mlp", o[0].detach().clone())))
    # HF 注意力内部：q_proj/k_proj/v_proj 输出（rope 前）与 o_proj 输入
    ra = rl0.self_attn
    for pname in ("q_proj", "k_proj", "v_proj"):
        hooks.append(getattr(ra, pname).register_forward_hook(
            lambda _m, _i, o, pname=pname: ref.__setitem__(
                f"l0_{pname}", o[0].detach().clone())))
    hooks.append(ra.o_proj.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("l0_o_proj_out", o[0].detach().clone())))
    hooks.append(rl0.input_layernorm.register_forward_hook(
        lambda _m, _i, o: ref.__setitem__("l0_ln1", o[0].detach().clone())))
    with torch.no_grad():
        out = ref_m(input_ids=torch.tensor([tok2.encode(PROMPT)], device="cuda"),
                    output_hidden_states=True)
    for h in hooks:
        h.remove()
    ref["hidden"] = ref_m.model.norm(out.hidden_states[-1][0]).detach().clone()
    ref["logits"] = out.logits[0].detach().clone().float()
    ref["_w_embed"] = ref_m.model.embed_tokens.weight.data.detach().clone()
    del ref_m
    torch.cuda.empty_cache()

    print(f"\n逐层 max |diff| (ours vs HF, prompt={len(ids)} tokens):")
    print(f"  ids: {ids}")
    # embed 层我们侧的 ×√d 缩放发生在 Gemma2Model.forward（embed_tokens 之后），
    # hook 抓到的是未缩放值 → 对比时补乘 √hidden
    import math
    ours_embed_scaled = ours["embed"] * math.sqrt(llm.config.hf_config.hidden_size)
    d = (ours_embed_scaled.float() - ref["embed"].float()).abs().max().item()
    print(f"  embed (ours×√d)  max_diff={d:.4f}")
    for name in ("layer0", "layer1", "layer_last", "hidden"):
        if name in ours and name in ref:
            d = (ours[name].float() - ref[name].float()).abs().max().item()
            print(f"  {name:<12} max_diff={d:.4f}")
    print("  --- 层0 内部二分 ---")
    for name in ("l0_attn", "l0_pa", "l0_mlp"):
        if name in ours and name in ref:
            d = (ours[name].float() - ref[name].float()).abs().max().item()
            print(f"  {name:<12} max_diff={d:.4f}")
    print("  --- 层0 注意力内部（qkv 按 q/k/v 顺序展开比较） ---")
    if "l0_ln1_out" in ours and "l0_ln1" in ref:
        d = (ours["l0_ln1_out"].float() - ref["l0_ln1"].float()).abs().max().item()
        print(f"  l0_ln1 (注意力输入)  max_diff={d:.4f}")
        print(f"  ours l0_ln1 row0 head: {ours['l0_ln1_out'][0, :6].tolist()}")
        print(f"  ref  l0_ln1 row0 head: {ref['l0_ln1'][0, :6].tolist()}")
        # 手算 RMSNorm：输入 = embed×√d（已核对 0 diff），权重 = 我们的 ln1 权重
        x = ours["embed"] * math.sqrt(llm.config.hf_config.hidden_size)
        w = llm.model_runner.model.model.layers[0].input_layernorm.weight.data
        var = x.float().pow(2).mean(-1, keepdim=True)
        manual = (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype) * w
        d_man = (manual.float() - ours["l0_ln1_out"].float()).abs().max().item()
        print(f"  手算 RMSNorm(embed×√d) vs 捕获 ln1: {d_man:.2e}")
        d_man2 = (manual.float() - ref["l0_ln1"].float()).abs().max().item()
        print(f"  手算 RMSNorm(embed×√d) vs HF  ln1: {d_man2:.4f}")
    if "l0_qkv_pre" in ours and "l0_q_proj" in ref:
        qkv = ours["l0_qkv_pre"]  # [T, 4096] = [q|k|v]
        qs, ks, vs = qkv.split([2048, 1024, 1024], dim=-1)
        for nm, a, b in (("q", qs, ref["l0_q_proj"]), ("k", ks, ref["l0_k_proj"]),
                         ("v", vs, ref["l0_v_proj"])):
            d = (a.float() - b.float()).abs().max().item()
            print(f"  l0_{nm}_proj   max_diff={d:.4f}")
    # F.linear 直算核对：我们的 ln1 输出 × 我们的 qkv 权重 == 捕获的 qkv_pre？
    if "l0_ln1_out" in ours and "l0_qkv_pre" in ours:
        import torch.nn.functional as F
        qkv_lin = F.linear(ours["l0_ln1_out"], attn0.qkv_proj.weight.data, attn0.qkv_proj.bias)
        d = (qkv_lin.float() - ours["l0_qkv_pre"].float()).abs().max().item()
        print(f"  F.linear(ln1, qkv_w) vs 捕获 qkv_pre: {d:.2e}")
        # HF 侧同样核对
        if "l0_ln1" in ref and "l0_q_proj" in ref:
            q_lin = F.linear(ref["l0_ln1"], ref_m.model.layers[0].self_attn.q_proj.weight.data,
                             ref_m.model.layers[0].self_attn.q_proj.bias)
            d2 = (q_lin.float() - ref["l0_q_proj"].float()).abs().max().item()
            print(f"  HF F.linear(ln1, q_w) vs 捕获 q_proj: {d2:.2e}")
            # 终极：同一输入×同一权重（双方）——直接比较 ln1 输出与权重逐元素
            d3 = (ours["l0_ln1_out"].float() - ref["l0_ln1"].float()).abs().max().item()
            print(f"  ln1 输出直接对比: {d3:.4f}")
    if "l0_q_rope" in ours:
        print(f"  (rope 后 q/k 已抓取，ours l0_q_rope[:2,:2,:4]: "
              f"{ours['l0_q_rope'][:2, :2, :4].flatten().tolist()})")
    # embed 直算核对：两侧权重 + 已知 ids
    import torch.nn.functional as F
    ours_e = F.embedding(torch.tensor(ids, device="cuda"), ours["_w_embed"])
    ref_e = F.embedding(torch.tensor(ids, device="cuda"), ref["_w_embed"])
    print(f"  F.embedding(ids, 各自权重): ours-vs-captured diff="
          f"{(ours_e - ours['embed']).abs().max().item():.2e}  "
          f"ref-vs-captured diff={(ref_e - ref['embed']).abs().max().item():.2e}")
    print(f"  ours row0 head: {ours['embed'][0, :6].tolist()}")
    print(f"  ref  row0 head: {ref['embed'][0, :6].tolist()}")
    d = (ours["logits"] - ref["logits"]).abs()
    top1 = (ours["logits"].argmax(-1) == ref["logits"].argmax(-1)).float().mean().item()
    print(f"  logits      max_diff={d.max().item():.4f} top-1={100*top1:.1f}%")
    print(f"  ours argmax (末行): {ours['logits'][-1].argmax(-1).item()}  "
          f"ref argmax (末行): {ref['logits'][-1].argmax(-1).item()}")


if __name__ == "__main__":
    main()
