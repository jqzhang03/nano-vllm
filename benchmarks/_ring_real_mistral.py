"""真实 Mistral-7B-v0.1 长解码 环 vs 掩码（阶段 2b 扩展，GPU）。

权重 int4（streaming 纯 int4，16GB 卡装 7B）；解码 ~5050 token > W=4096
（跨窗 + 多次环驱逐）。对照（同 dtype 内）：
  ring(rolling_cache=True) vs masked(False)：同种子同温度，共同前缀内逐
  decode 行 top-1 / logits 差；末态序列块表长度对比（环内存有界 vs 掩码
  线性增长——解码内存账本的实模型证据）。
每配置打印 KV 显存证据：完成时 per-seq 块表长 & 环上界 ring_cap。

用法: python benchmarks/_ring_real_mistral.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL = os.path.expanduser("~/huggingface/Mistral-7B-v0.1")
W = 4096
B = 256
MAX_TOK = 5050  # 每序列生成 token 数（> W，覆盖跨窗与多次驱逐）


def build_prompts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

    def _ids(text):
        return tok.encode(text, add_special_tokens=True)

    return [_ids("The history of the Roman Empire spans more than a thousand years"
                 " and shaped Europe deeply."),
            _ids("A mathematical proof of Fermat's last theorem was completed by"
                 " Andrew Wiles in 1995 after seven years of work.")]


def run_one(rolling: bool, fp8kv: bool, prompts, seed):
    """引擎循环（手工驱动）：decode 步数每 250 步采样各序列块表长度。"""
    from nanovllm import LLM, SamplingParams
    torch.manual_seed(seed)
    llm = LLM(MODEL, max_model_len=5600, quantization="int4", kv_swap=False,
              rolling_cache=rolling,
              kv_cache_dtype="fp8_e4m3" if fp8kv else "auto",
              max_num_seqs=8, max_num_batched_tokens=8192)
    assert llm.model_runner.streaming, "期望流式加载（16GB 卡装 7B int4）"
    sp = SamplingParams(temperature=0.8, max_tokens=MAX_TOK)
    for p in prompts:
        llm.add_request(p, sp)
    llm._collect_logits = True
    llm.collected_logits = []
    done = {}
    lens_trace = []          # (decode步数, per-seq 块表长) 运行中采样
    cap = llm.scheduler.block_manager.ring_cap
    step_no = 0
    while not llm.is_finished():
        out, kind, _, _ = llm.step()
        if kind in ("decode", "spec", "mixed"):
            step_no += 1
            run = list(llm.scheduler.running)
            if run and step_no % 250 == 0:
                lens_trace.append((step_no, [len(s.block_table) for s in run]))
        for seq_id, toks in out:
            done[seq_id] = toks
    dec = [lg for k, lg in llm.collected_logits
           if k == "decode" and lg is not None]
    comps = [done[sid] for sid in sorted(done)]
    llm.exit()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return dec, comps, lens_trace, cap


def cpl(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    PROMPTS = build_prompts()
    for fp8kv in (False, True):
        tag = "fp8KV" if fp8kv else "bf16KV"
        print(f"=== {tag} 权重 int4-streaming ===")
        print("[ring]")
        dec_r, comp_r, trace_r, cap = run_one(True, fp8kv, PROMPTS, 31415)
        print(f"  ring 块表采样: {trace_r[:6]} … (cap {cap})")
        print("[masked]")
        dec_m, comp_m, trace_m, cap_m = run_one(False, fp8kv, PROMPTS, 31415)
        print(f"  masked 块表采样: {trace_m[:6]} …")
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
        print(f"[{tag} ring vs masked] decode 步 {n}，可比行 {agree_tot}: "
              f"top-1 一致 {100*agree_ok/max(agree_tot,1):.2f}% (翻转 {flip}) | "
              f"max {maxd:.4f} mean {mean_d:.5f}")
        print(f"[{tag} 完成token] ring {comp_r} masked {comp_m}"
              f" | 共同前缀长 {min(cpls)}")
        assert flip <= max(2, agree_tot // 100), f"{tag} 环 vs 掩码翻转过多"
    print("REAL MISTRAL-7B RING E2E OK")


if __name__ == "__main__":
    main()
