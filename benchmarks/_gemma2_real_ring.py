"""真实 gemma-2-2b-it 交替窗口滚动（split，阶段 2b 扩展 M4，GPU）——最终对照。

引擎解码 4800 token（ignore_eos，跨 W=4096 + local 层环驱逐/full 池长历史）：
- 每 decode 步经 collect 容器只留 argmax（轨迹证据）；**每 SPARSE 步** 额外驻留
  全 logits（[行, 256K] fp16 cpu ≈1MB/步）；
- 引擎退出后载 fp16 模块做手工稠密参考（窗口+softcap，独立实现），按稀疏步
  （context = plen + d，d = 1-based decode 步号）重算两序列末行 → 逐稀疏步
  top-1/logits 差；
- split 与 masked 各跑一遍对照同一参考：失配同量级 ⇒ 环/双池布局正确。

用法: python benchmarks/_gemma2_real_ring.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL = os.path.expanduser("~/huggingface/gemma-2-2b-it")
W = 4096
B = 256
MAX_TOK = 4800
SPARSE = 300


def build_prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def _ids(text):
        return tok.encode(text, add_special_tokens=True)

    return [_ids("Explain how a neural network learns from data, step by step,"
                 " including backpropagation and gradient descent."),
            _ids("Write a short story about a robot that discovers poetry and"
                 " decides to become a gardener in a quiet valley.")]


class CollectSparse(list):
    """收集容器：常规步只留 argmax；每 SPARSE 个 decode 步驻留全 logits fp16。"""

    def __init__(self, sparse):
        super().__init__()
        self._dec = 0
        self._sparse = sparse

    def append(self, item):  # noqa: D102
        kind, lg = item
        if lg is None:
            return
        if kind == "decode":
            self._dec += 1
            if self._dec % self._sparse == 0:
                list.append(self, ("sparse", self._dec, lg.float().cpu()))
                return
        list.append(self, ("am", lg.argmax(dim=-1).cpu()))


def run_engine(rolling: bool, prompts, seed, cache_pt):
    """引擎跑完存盘（稀疏 logits ~几十 MB），供 ref 阶段重复装载。"""
    if os.path.exists(cache_pt):
        import pickle
        with open(cache_pt, "rb") as f:
            comps, collected, trace, cap = pickle.load(f)
        print(f"  (loaded {cache_pt}) completions {[len(c) for c in comps]}")
        return comps, collected, trace, cap
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(seed)
    llm = LLM(MODEL, max_model_len=5400, quantization="none", kv_swap=False,
              rolling_cache=rolling, enforce_eager=True,
              max_num_seqs=8, max_num_batched_tokens=8192)
    sp = SamplingParams(temperature=0.8, max_tokens=MAX_TOK, ignore_eos=True)
    for p in prompts:
        llm.add_request(p, sp)
    llm._collect_logits = True
    llm.collected_logits = CollectSparse(SPARSE)
    done = {}
    trace = []
    cap = getattr(llm.scheduler.block_manager, "ring_cap", -1)
    while not llm.is_finished():
        out, kind, _, _ = llm.step()
        if kind == "decode":
            run = list(llm.scheduler.running)
            if run and llm.collected_logits._dec % 250 == 0:
                if getattr(llm.scheduler, "rolling_split", False):
                    trace.append((llm.collected_logits._dec,
                                  [len(s.block_table) for s in run],
                                  [len(s.kv_table) for s in run]))
                else:
                    trace.append((llm.collected_logits._dec,
                                  [len(s.block_table) for s in run]))
        for seq_id, toks in out:
            done[seq_id] = toks
    comps = [done[sid] for sid in sorted(done)]
    collected = list(llm.collected_logits)
    llm.exit()
    import gc
    import pickle
    gc.collect()
    torch.cuda.empty_cache()
    with open(cache_pt, "wb") as f:
        pickle.dump((comps, collected, trace, cap), f)
    print(f"  {MODEL.split('/')[-1]} rolling={rolling} | completions "
          f"{[len(c) for c in comps]} | cap {cap} | trace {trace[:3]} …")
    return comps, collected, trace, cap


def ref_logits_chunked(model, tokens, T):
    """手工稠密参考（窗口+softcap），query 分块 → O(T·C) 内存（T 大时 O(T²) fp32
    会 OOM 16GB，见 OOM 记录）。与 _gemma2_ring_e2e.ref_logits 数学一致。"""
    import torch.nn.functional as F
    dev = "cuda"
    C = 512  # query 分块
    with torch.no_grad():
        ids = torch.tensor(tokens[:T], device=dev)
        pos = torch.arange(T, device=dev)
        x = model.model.embed_tokens(ids) * (model.config.hidden_size ** 0.5)
        for n_l in model.model.layers:
            attn = n_l.self_attn
            h = n_l.input_layernorm(x)
            hd = attn.head_dim
            qkv = attn.qkv_proj(h)
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q = q.view(T, attn.num_heads, hd)
            k = k.view(T, attn.num_kv_heads, hd)
            v = v.view(T, attn.num_kv_heads, hd)
            q, k = attn.rotary_emb(pos, q, k)
            g = attn.num_heads // attn.num_kv_heads
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
            kf = k.float()
            vf = v.float()
            cap = attn.attn.logit_softcapping
            win = attn.attn.window_size
            out_parts = []
            for c0 in range(0, T, C):
                c1 = min(T, c0 + C)
                qc = q[c0:c1]
                s = torch.einsum("phd,khd->phk", qc.float(), kf) * attn.scaling
                if cap:
                    s = cap * torch.tanh(s / cap)
                causal = torch.arange(c0, c1, device=dev)[:, None] >= \
                    torch.arange(T, device=dev)[None, :]
                if win:
                    causal = causal & (torch.arange(c0, c1, device=dev)[:, None]
                                       < torch.arange(T, device=dev)[None, :] + win)
                p = s.masked_fill(~causal.unsqueeze(1), float("-inf")).softmax(-1)
                a = torch.einsum("phk,khd->phd", p, vf).to(qc.dtype)
                del s, p, causal
                out_parts.append(a)
            o = torch.cat(out_parts, 0)
            o = attn.o_proj(o.reshape(T, -1))
            del kf, vf
            x1 = n_l.post_attention_layernorm(o) + x
            x = x1 + n_l.post_feedforward_layernorm(
                n_l.mlp(n_l.pre_feedforward_layernorm(x1)))
            del o, x1
        fin = model.model.norm(x)
        import torch.nn.functional as F
        logits = F.linear(fin[-1:], model.lm_head.weight).float()[0]
        cap = model.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        return logits


def check_vs_ref(comps, collected, plen, tag):
    """稀疏步全 logits vs 手工稠密参考（fp16 模块，query 分块）。"""
    import _gemma2_ring_e2e as g2
    g2.TOY = MODEL
    ref = g2.build_ref()
    dec_sparse = []
    for item in collected:
        if item[0] == "sparse":
            dec_sparse.append((item[1], item[2]))
    # d 是 1-based decode 步号 → context = plen + d（completions-1 惯例见头部）
    total = bad = 0
    maxd = 0.0
    mean_sum = 0.0
    details = []
    for d, lg in dec_sparse:
        for i in range(len(plen)):
            T = plen[i] + d
            tokens = comps[i]
            so_far = _prefix_tokens(i, plen) + tokens
            if T > len(so_far):
                continue
            rl = ref_logits_chunked(ref, so_far, T).cpu()
            e = lg[i].float()
            dd = (e - rl).abs()
            ok = int(e.argmax() == rl.argmax())
            total += 1
            bad += 0 if ok else 1
            maxd = max(maxd, dd.max().item())
            mean_sum += dd.mean().item()
            details.append((d, T, ok, dd.mean().item()))
    mean_d = mean_sum / max(total, 1)
    print(f"[{tag} vs 手工参考] 稀疏步行 {total}: top-1 失配 {bad}"
          f" | max {maxd:.3f} mean {mean_d:.6f}")
    for d, T, ok, m in details[::3]:
        print(f"   步{d} L={T}: top1 {'Y' if ok else 'N'} mean {m:.5f}")
    return bad, total


def _prefix_tokens(si, plen):
    # 与 build_prompts 相同（避免重复分词）
    return PROMPTS[si]


def main():
    global PROMPTS
    PROMPTS = build_prompts()
    plen = [len(p) for p in PROMPTS]
    print("[engine split(ring)]")
    comp_r, col_r, trace_r, cap = run_engine(True, PROMPTS, 2024,
                                             "/tmp/_g2ring_r.pt")
    print(f"  completions {[len(c) for c in comp_r]} | ring cap {cap}")
    print("[engine masked]")
    comp_m, col_m, trace_m, _ = run_engine(False, PROMPTS, 2024,
                                           "/tmp/_g2ring_m.pt")
    print(f"  completions {[len(c) for c in comp_m]}")
    del comp_m, col_m, trace_m
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    b1, t1 = check_vs_ref(comp_r, col_r, plen, "split")
    assert b1 <= max(2, t1 // 10), "split 对照手工参考失配过多"
    print("REAL GEMMA2-2B SPLIT-RING E2E OK")


if __name__ == "__main__":
    main()
