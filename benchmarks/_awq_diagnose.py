"""AWQ 决定性诊断：在引擎真实 prefill 激活上，逐层对比 RTN(s=1) vs AWQ(α搜索) 的输出误差。

判定：若 AWQ 在引擎真实激活上仍更差 → 缩放对该模型无益（结论：RTN int4 胜出）；
若 AWQ 更好 → α 搜索过拟合了校准样本（结论：需要更大/更贴合的校准集）。
"""
import os
import sys

import torch

from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
SCALES = os.path.join("results", "awq_scales.pt")
PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
    "The three laws of robotics are",
    "A summary of the water cycle:",
    "Machine learning is",
    "The best way to learn programming is",
    "Photosynthesis happens when",
    "In 1969, humans",
]
GROUP = 128


def quant_out_err(w, x, s):
    N, K = w.shape
    ws = w * s.clamp(min=1e-8)[None, :]
    g = ws.view(N, K // GROUP, GROUP)
    scale = g.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 7.0
    q = torch.clamp(torch.round(g / scale), -7, 7)
    deq = (q * scale).view(N, K) / s.clamp(min=1e-8)[None, :]
    return ((deq - w).float() @ x.float().t()).norm() / (w.float() @ x.float().t()).norm().clamp(min=1e-8)


llm = LLM(PATH, quantization="none", max_model_len=4096)
llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

from nanovllm.layers.linear import LinearBase
from nanovllm.layers.embed_head import ParallelLMHead
mods = [(name, m) for name, m in llm.model_runner.model.named_modules()
        if isinstance(m, LinearBase) and not isinstance(m, ParallelLMHead)]
for _, m in mods:
    m.x_samp = []

hooks = []
for _, m in mods:
    def make_hook(mm):
        def hook(_mod, args):
            mm.x_samp.append(args[0].float().detach().to(torch.bfloat16))
        return hook
    hooks.append(m.register_forward_pre_hook(make_hook(m)))

torch.manual_seed(123)
llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=64), use_tqdm=False)
for h in hooks:
    h.remove()

state = torch.load(SCALES, map_location="cpu")
win_awq = win_rtn = 0
print(f"{'layer':<52}  err_RTN   err_AWQ   赢家")
for name, m in mods:
    x = torch.cat(m.x_samp, dim=0).float()
    w = m.weight.detach().float()
    e_rtn = quant_out_err(w, x, torch.ones(w.shape[1], device=w.device))
    e_awq = quant_out_err(w, x, state[name].to(w.device))
    win = "AWQ" if e_awq < e_rtn else "RTN"
    if e_awq < e_rtn:
        win_awq += 1
    else:
        win_rtn += 1
    print(f"{name:<52}  {e_rtn.item():.4f}  {e_awq.item():.4f}   {win}")
print(f"\n引擎真实激活上：AWQ 赢 {win_awq} 层 / RTN 赢 {win_rtn} 层")
llm.exit()
