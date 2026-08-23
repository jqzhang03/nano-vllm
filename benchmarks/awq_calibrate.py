"""AWQ 激活感知缩放校准（真实文本 + 按层 α 搜索）：收集每层输入逐通道 mean|X|
与一小批激活样本，对每层在 α ∈ {0, 0.1, ..., 1.0} 网格上搜索使量化输出误差
||Q(W/s)·s·X − W·X||_F 最小的缩放 s = mean|X|^α（归一化），保存 scales 文件。

用法（WSL，GPU）：
    python benchmarks/awq_calibrate.py [--out results/awq_scales.pt] [--n-gen 12]
引擎侧：
    LLM(..., quantization="awq", awq_scales_path="results/awq_scales.pt")

为什么按层搜索：固定 α=0.5 时 s 的动态范围可达 0.008~7.3——除以极小 s 的通道
会把该 128 组的 amax 撑爆，整组其他通道量化塌缩（实测 KL 1.08 → 12.4 灾难）。
α→0 时 s→1（等价于 RTN），搜索自动在"不缩放"与"缩放"之间按层取舍。
"""
import argparse
import os

import torch
import torch.nn.functional as F

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
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
N_SAMPLE = 1024  # 每层保留的校准激活行数（评分用）


def quant_error(w: torch.Tensor, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """给定缩放 s，返回量化后的输出误差 ||(Q(W·s)/s − W)·X^T||_F（AWQ论文方向：权重乘s、激活除s）。"""
    N, K = w.shape
    ws = w * s.clamp(min=1e-8)[None, :]
    g = ws.view(N, K // GROUP, GROUP)
    scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(g / scale), -7, 7)
    deq = (q * scale).view(N, K) / s.clamp(min=1e-8)[None, :]  # 还原到原始尺度（含缩放误差）
    err = (deq - w).float() @ x.float().t()  # 输出误差 [N, n_sample]
    return err.norm() / (w.float() @ x.float().t()).norm().clamp(min=1e-8)


def make_scale(mean: torch.Tensor, w_col: torch.Tensor, alpha: float) -> torch.Tensor:
    """llm-awq 同款：s = (act_scale / w_col)^α 归一化。α=0 → s=1（RTN 基线）；
    α=0.5 → 权重均衡化（act^0.5/w^0.5）；α=1 → 完全均衡化。"""
    s = (mean.clamp(min=1e-8) / w_col.clamp(min=1e-8)) ** alpha
    return s / s.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="模型目录（默认 Qwen3-0.6B）")
    ap.add_argument("--out", default="results/awq_scales.pt")
    ap.add_argument("--n-gen", type=int, default=12, help="模型自生成的真实文本序列数")
    ap.add_argument("--gen-len", type=int, default=256)
    args = ap.parse_args()
    args.model = os.path.expanduser(args.model)  # bash argv 不展开 ~ → 手动展开

    llm = LLM(args.model, quantization="none", max_model_len=4096)
    print("tie_word_embeddings:", llm.model_runner.config.hf_config.tie_word_embeddings)
    llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

    from nanovllm.layers.linear import LinearBase
    from nanovllm.layers.embed_head import ParallelLMHead
    # 与引擎 _quant_mods 默认一致：不量化 lm_head（logits 直接由它决定，见 BENCHMARKS.md §10）
    mods = [(name, m) for name, m in llm.model_runner.model.named_modules()
            if isinstance(m, LinearBase) and not isinstance(m, ParallelLMHead)]
    for _, m in mods:
        m.ax_sum = None
        m.ax_n = 0
        m.ax_sample = []

    hooks = []
    for _, m in mods:
        def make_hook(mm):
            def hook(_mod, args):
                x = args[0].float()
                s = x.abs().sum(dim=0)
                if mm.ax_sum is None:
                    mm.ax_sum = s.cpu()
                else:
                    mm.ax_sum += s.cpu()
                mm.ax_n += x.shape[0]
                mm.ax_sample.append(x.detach().to(torch.bfloat16))
                if len(mm.ax_sample) > 8:
                    mm.ax_sample = mm.ax_sample[-8:]  # 只保留最近几次调用
            return hook
        hooks.append(m.register_forward_pre_hook(make_hook(m)))

    # 校准数据：真实 prompt + 模型自生成续写（真实激活分布）
    tokenizer = llm.tokenizer
    prompts = [tokenizer.encode(p) for p in REAL_PROMPTS]
    gen = llm.generate(prompts[:args.n_gen],
                       SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=args.gen_len),
                       use_tqdm=False)
    prompts += [o["token_ids"] for o in gen]
    sps = [SamplingParams(temperature=0.6, max_tokens=8)] * len(prompts)
    llm.generate(prompts, sps, use_tqdm=False)  # 校准前向（hooks收集统计+样本）
    for h in hooks:
        h.remove()

    state = {}
    n_search = 0
    for name, m in mods:
        mean = (m.ax_sum / max(m.ax_n, 1)).to("cuda")
        x_cal = torch.cat(m.ax_sample, dim=0)[-N_SAMPLE:].float()  # [n, K]
        w = m.weight.detach().float()
        w_col = w.abs().amax(dim=0)  # [K] 每输入通道权重max（均衡化用）
        # α 网格搜索：使量化输出误差最小（llm-awq 同款逐层搜索）
        best_s, best_err, best_alpha = None, None, None
        for alpha in ALPHAS:
            s = make_scale(mean, w_col, alpha)
            err = quant_error(w, x_cal, s)
            if best_err is None or err < best_err:
                best_err, best_s, best_alpha = err, s, alpha
            n_search += 1
        state[name] = best_s.to(torch.bfloat16)
        print(f"{name:<52} α={best_alpha:.1f}  s[min={best_s.min().item():.3f} "
              f"max={best_s.max().item():.3f}]  rel_err={best_err.item():.4f}")
        del m.ax_sum, m.ax_n, m.ax_sample

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(state, args.out)
    print(f"saved {len(state)} scales (α search over {n_search} candidates) -> {args.out}")
    llm.exit()


if __name__ == "__main__":
    main()
