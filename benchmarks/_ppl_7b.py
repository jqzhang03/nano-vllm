"""7B 级模型真实文本 ppl（fp16 / int4(RTN) / awq(α搜索) / fp8）——_quant_ppl.py 的 7B 版。

用法（WSL，GPU，单卡空闲时跑）：
    python benchmarks/_ppl_7b.py [model]          # model 缺省 = Mistral-7B-v0.1

与 0.6B 版的方法学差异（诚实标注）：
  1. 评测语料由 **量化引擎**生成（12 条 × 256 token，temp=0.8）——7B 的 fp16 引擎在
     16GB 卡/11GB WSL RAM 上装不下（14.5GB 权重 + KV + 图）。**语料生成器不能与被评
     模型同量化级**：首轮用 int4 引擎生成语料，测得 int4 ppl(5.195) < fp16(5.285) 的
     伪优势（自生成文本对生成器 in-distribution）——该批数据弃用。现用 fp8 引擎生成
     （fp8≈fp16 输出分布，引擎侧流式可装载），跨模式偏差可忽略。
  2. fp16 裸模型**直接在 cuda 上构造**（0.6B 版 CPU 构造再 .to(cuda) 需要 ~14.5GB CPU
     RAM，WSL 11GB 会 OOM）；loader 仍逐 key 拷入，峰值 ≈ 权重 14.5GB + 上下文 ~1GB。
  3. ppl 与 AWQ 校准共用同一批自生成语料 + 20 条真实短 prompt 的激活（0.6B 版校准另有
     一轮生成；此处为省 GPU 时间共用，awq 数字可能略偏乐观，报告时注明）。
  4. AWQ 校准（α∈{0..1} 网格、llm-awq 同款 ||(Q(W·s)/s−W)·X^T|| 归一化误差）在 fp16
     裸模型前向上做（引擎 fp16 不可用），每层保留 ~512 行激活样本（0.6B 版 1024 行，
     其校准用引擎 prefill 激活、样本更多更均匀——7B 版校准统计量较弱，awq 结论保守读）。
  5. ppl 口径与 0.6B 版一致：12 条续写拼接为一条长文本前向，跨序列边界标签剔除
     （0.6B 版为 3060 token；此版 ≈3060 token，长度依生成而定）。

输出：控制台表格 + results/ppl_7b.json + results/awq_scales_<model名>.pt（校准文件，
引擎侧 awq_scales_path 可直接使用）。

仅实测过 Mistral-7B-v0.1（fp16 14.5GB 在 16GB 卡上可装载）；Llama-3.1-8B(16.1GB)/Qwen2.5-7B
的 fp16 超载，本脚本会如实报 fp16 不可用并继续 int4/fp8（awq 无校准则回落 RTN）。
"""
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Mistral-7B-v0.1/")
REAL_PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
    "The three laws of robotics are",
    "A summary of the water cycle:",
    "Machine learning is",
    "The best way to learn programming is",
    "Photosynthesis happens when",
    "In 1969, humans",
    "The history of the steam engine begins with",
    "A good morning routine starts with",
    "The solar system consists of",
    "To improve your sleep, you should",
    "Quantum computing works by",
    "The main difference between TCP and UDP is",
    "A healthy diet should include",
    "The rules of chess are",
    "Deep learning models are trained by",
    "The Amazon rainforest is",
    "Cooking pasta correctly requires",
    "The invention of the telephone is credited to",
]

GROUP = 128
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SAMPLE = 512  # 每层保留的校准激活行数（评分用；0.6B 版为 1024）
N_GEN = 12       # ppl 语料序列数
GEN_LEN = 256


GEN_QUANT = "fp8"  # 语料生成引擎的量化（须 ≠ 被评模型；fp8≈fp16，见文件头说明）


def log(msg):
    print(f"[ppl7b] {msg}", flush=True)


def quant_error(w: torch.Tensor, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """给定缩放 s，量化输出误差 ||(Q(W·s)/s − W)·X^T||_F / ||W·X^T||_F（AWQ 论文方向）。"""
    N, K = w.shape
    ws = w * s.clamp(min=1e-8)[None, :]
    g = ws.view(N, K // GROUP, GROUP)
    scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(g / scale), -7, 7)
    deq = (q * scale).view(N, K) / s.clamp(min=1e-8)[None, :]
    denom = (w.float() @ x.float().t()).norm().clamp(min=1e-8)
    return ((deq - w).float() @ x.float().t()).norm() / denom


def make_scale(mean: torch.Tensor, w_col: torch.Tensor, alpha: float) -> torch.Tensor:
    s = (mean.clamp(min=1e-8) / w_col.clamp(min=1e-8)) ** alpha
    return s / s.mean()


def quant_mods(model):
    from nanovllm.layers.linear import LinearBase
    from nanovllm.layers.embed_head import ParallelLMHead
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, LinearBase) and not isinstance(m, ParallelLMHead)]


def build_fp16(path):
    """直接在 cuda 上以 bf16 构造裸模型 + eager 加载（CPU RAM 装不下 14.5GB fp16）。"""
    hf = AutoConfig.from_pretrained(path)
    from nanovllm.models.registry import get_model_class
    from nanovllm.utils.loader import load_model
    ddtype = torch.get_default_dtype()
    ddev = torch.get_default_device()
    torch.set_default_dtype(hf.dtype)
    torch.set_default_device("cuda")
    model = get_model_class(hf.model_type)(hf)
    torch.set_default_device(ddev)
    torch.set_default_dtype(ddtype)
    load_model(model, path)  # eager：逐 key CPU→GPU
    if getattr(hf, "tie_word_embeddings", False):
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
    model.eval()
    return model, hf


def run_forward(model, ids, positions):
    """一次 varlen prefill 前向（context 契约与 _quant_ppl 同款）。返回 hidden [T, H]。"""
    from nanovllm.utils.context import set_context, reset_context
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, T, T,
                torch.tensor([], dtype=torch.int32, device="cuda"), None, None)
    with torch.inference_mode():
        hidden = model(ids, positions)
    reset_context()
    return hidden


def eval_ppl(model, seqs, starts):
    """拼接长文本单前向；跨序列边界标签剔除（0.6B 版同口径）。返回 (ppl, nll, cnt)。"""
    ids = torch.tensor([t for s in seqs for t in s], device="cuda")
    T = ids.numel()
    bounds = set(starts)  # 每 seq 起点的全局下标
    positions = torch.arange(T, device="cuda")
    hidden = run_forward(model, ids, positions)
    nll = 0.0
    cnt = 0
    with torch.inference_mode():
        for c0 in range(0, T, 512):
            logits = model.compute_logits(hidden[c0:c0 + 512])
            logp = F.log_softmax(logits.float(), dim=-1)
            nxt = torch.arange(c0, min(c0 + 512, T), device="cuda") + 1
            ok = (nxt < T) & ~torch.tensor([int(n.item()) in bounds for n in nxt], device="cuda")
            lab = ids[nxt[ok]]
            sel = logp[ok]
            nll += -sel[torch.arange(lab.numel(), device="cuda"), lab].sum().item()
            cnt += lab.numel()
    del hidden, ids
    gc.collect()
    torch.cuda.empty_cache()
    return math.exp(nll / max(cnt, 1)), nll, cnt


def attach_awq_hooks(mods):
    """每层收集逐通道 mean|X| 与 ≤N_SAMPLE 行激活样本（bf16 存 CPU，容量有界）。"""
    for _, m in mods:
        m.ax_sum = None
        m.ax_n = 0
        m.ax_samp = None

    def make_hook(mm):
        def hook(_mod, args):
            x = args[0].detach()
            K = x.shape[-1]
            s = x.float().abs().sum(dim=0).cpu()  # [K]
            if mm.ax_sum is None:
                mm.ax_sum = s
            else:
                mm.ax_sum += s
            mm.ax_n += x.shape[0]
            rows = x.reshape(-1, K).to("cpu", dtype=torch.bfloat16)
            if mm.ax_samp is None:
                mm.ax_samp = rows
            else:
                mm.ax_samp = torch.cat([mm.ax_samp, rows], dim=0)
            while mm.ax_samp.size(0) > N_SAMPLE:  # 均匀降采样保持容量有界
                mm.ax_samp = mm.ax_samp[::2]
        return hook
    return [m.register_forward_pre_hook(make_hook(m)) for _, m in mods]


def calibrate_awq(model, seqs, starts, prompts):
    """fp16 裸模型前向上收集激活 → 按层 α 搜索 → 返回 state {name: s(bf16)}。"""
    mods = quant_mods(model)
    hooks = attach_awq_hooks(mods)
    # pass 1：ppl 语料（12 条续写拼接）
    T = sum(len(s) for s in seqs)
    ids = torch.tensor([t for s in seqs for t in s], device="cuda")
    run_forward(model, ids, torch.arange(T, device="cuda"))
    del ids
    # pass 2：20 条真实短 prompt 拼接（补足激活分布，0.6B 校准同源文本）
    pids = torch.tensor([t for p in prompts for t in p], device="cuda")
    run_forward(model, pids, torch.arange(pids.numel(), device="cuda"))
    del pids
    for h in hooks:
        h.remove()
    gc.collect()
    torch.cuda.empty_cache()

    state = {}
    n_cand = 0
    # α 搜索放 CPU：fp16 模型驻留时 GPU 空闲显存 <1.5GB，逐层 float 副本会 OOM
    for name, m in mods:
        mean = (m.ax_sum / max(m.ax_n, 1))                        # [K] CPU fp32
        x_cal = m.ax_samp[-N_SAMPLE:].float()                     # [n, K] CPU
        w = m.weight.detach().float().cpu()                       # [N, K] CPU
        w_col = w.abs().amax(dim=0)                               # [K]
        best = None
        for alpha in ALPHAS:
            s = make_scale(mean, w_col, alpha)
            err = quant_error(w, x_cal, s)
            n_cand += 1
            if best is None or err < best[0]:
                best = (err, s, alpha)
        state[name] = best[1].to(torch.bfloat16)
        log(f"  awq {name:<48} α={best[2]:.1f}  s[min={best[1].min().item():.3f} "
            f"max={best[1].max().item():.3f}]  rel_err={best[0].item():.4f}")
        del m.ax_sum, m.ax_n, m.ax_samp, w, x_cal, mean
    log(f"awq α 搜索完成：{len(state)} 层 × {len(ALPHAS)} α = {n_cand} 候选（CPU）")
    return state


def quantize_model(model, mode, state=None):
    """原位量化（与引擎语义一致：不量化 lm_head/embed；int4 纯模式 = streaming 7B 语义）。"""
    for name, m in quant_mods(model):
        if mode == "rtn":
            m.quantize_int4(dense_path=False)
        elif mode == "awq":
            s = state.get(name) if state else None
            if s is not None and s.numel() == m.weight.shape[1]:
                m.quantize_int4(s, dense_path=False)
            else:
                log(f"  [awq] {name} 缺 scale → 回落 RTN")
                m.quantize_int4(dense_path=False)
        elif mode == "fp8":
            m.quantize_fp8()


def main():
    model_dir = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else MODEL
    model_dir = model_dir.rstrip("/")
    tag = os.path.basename(model_dir)
    log(f"model = {model_dir}")
    t0 = time.time()

    # 1) 语料：fp8 流式引擎生成（fp16 引擎超载；生成器与被评量化级错开，见文件头）
    llm = LLM(model_dir, quantization=GEN_QUANT, max_model_len=4096)
    tok = llm.tokenizer
    llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    out = llm.generate([tok.encode(p) for p in REAL_PROMPTS[:N_GEN]],
                       SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=GEN_LEN),
                       use_tqdm=False)
    seqs = [o["token_ids"] for o in out]
    prompts = [tok.encode(p) for p in REAL_PROMPTS]
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    starts = [0]
    for s in seqs:
        starts.append(starts[-1] + len(s))
    log(f"语料：{len(seqs)} seqs × ~{GEN_LEN} token（{GEN_QUANT} 引擎自生成）；引擎已退出")

    # 2) fp16：校准（含 α 搜索）+ fp16 ppl
    model, _ = build_fp16(model_dir)
    log(f"fp16 裸模型就绪（{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B 参数）")
    state = calibrate_awq(model, seqs, starts, prompts)  # fp16 前向 + α 搜索
    results = {"model": model_dir, "date": time.strftime("%Y-%m-%d"),
               "corpus": f"{len(seqs)} {GEN_QUANT}-generated seqs x ~{GEN_LEN}", "ppl": {}}
    ppl, nll, cnt = eval_ppl(model, seqs, starts)
    results["ppl"]["fp16"] = {"ppl": round(ppl, 4), "tokens": cnt}
    log(f"ppl fp16      = {ppl:.4f}  (nll={nll:.1f} over {cnt} tokens)")
    torch.cuda.empty_cache()

    scales_path = f"results/awq_scales_{tag}.pt"
    os.makedirs("results", exist_ok=True)
    torch.save(state, scales_path)
    log(f"saved awq scales ({len(state)} layers) -> {scales_path}")

    # 3) 原位量化评测：RTN
    quantize_model(model, "rtn")
    ppl, nll, cnt = eval_ppl(model, seqs, starts)
    results["ppl"]["int4_rtn"] = {"ppl": round(ppl, 4), "tokens": cnt}
    log(f"ppl int4(RTN) = {ppl:.4f}  (nll={nll:.1f} over {cnt} tokens)")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 4) AWQ（重新 fp16 → 带 scales 量化）
    model, _ = build_fp16(model_dir)
    quantize_model(model, "awq", state)
    ppl, nll, cnt = eval_ppl(model, seqs, starts)
    results["ppl"]["awq"] = {"ppl": round(ppl, 4), "tokens": cnt}
    log(f"ppl awq       = {ppl:.4f}  (nll={nll:.1f} over {cnt} tokens)")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 5) fp8（同流程；fp16 不可用时该行为基线参考）
    model, _ = build_fp16(model_dir)
    quantize_model(model, "fp8")
    ppl, nll, cnt = eval_ppl(model, seqs, starts)
    results["ppl"]["fp8"] = {"ppl": round(ppl, 4), "tokens": cnt}
    log(f"ppl fp8       = {ppl:.4f}  (nll={nll:.1f} over {cnt} tokens)")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    with open("results/ppl_7b.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"结果 -> results/ppl_7b.json（wall {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
