"""sparse24 引擎级调试：eager vs CUDA-graph 路径定位（首个 decode 步 logits）。"""
import os

import torch

from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
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


def run_once(quant: str, eager: bool):
    llm = LLM(PATH, quantization=quant, enforce_eager=eager, max_model_len=4096)
    llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    torch.manual_seed(123)
    llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=64),
                 use_tqdm=False, collect_logits=True)
    steps = llm.collected_logits
    llm.exit()
    torch.cuda.empty_cache()
    for kind, logits in steps:
        if kind == "decode" and logits is not None:
            return logits
    raise RuntimeError("no decode step logged")


l16 = run_once("none", True)
print("== fp16 eager 基线 OK ==")

for eager in (True, False):
    ls = run_once("sparse24", eager)
    d = (l16.float() - ls.float()).abs()
    p16 = torch.softmax(l16.float(), dim=-1)
    ps = torch.softmax(ls.float(), dim=-1)
    kl = (p16 * (p16.log() - ps.log())).sum(dim=-1).mean().item()
    agree = (l16.argmax(-1) == ls.argmax(-1)).float().mean().item()
    print(f"sparse24 eager={eager}: max|Δ|={d.max().item():.4f} mean={d.mean().item():.5f} "
          f"KL={kl:.4f} top1={100 * agree:.1f}%")
