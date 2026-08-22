"""int4/awq 端到端困惑度对比（真实文本，比 8-prompt KL 更稳健）。

流程：fp16 引擎生成 12 条真实文本续写 → 裸模型上分别以 fp16 / RTN int4 / AWQ int4
前向（分块 logits），计算 next-token 困惑度。
"""
import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

PATH = "/home/zjq/huggingface/Qwen3-0.6B/"
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
]
DEV = "cuda"

# ---- 1) 真实文本数据（引擎自己初始化 NCCL 进程组） ----
llm = LLM(PATH, quantization="none", max_model_len=4096)
llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
tokenizer = llm.tokenizer
prompts = [tokenizer.encode(p) for p in REAL_PROMPTS]
out = llm.generate(prompts, SamplingParams(temperature=0.8, ignore_eos=True, max_tokens=256),
                   use_tqdm=False)
seqs = [o["token_ids"] for o in out]
llm.exit()
torch.cuda.empty_cache()

ids = torch.tensor([t for s in seqs for t in s], device=DEV)
bounds = set(torch.cumsum(torch.tensor([len(s) for s in seqs]), 0).tolist())  # 每seq末尾的全局下标
positions = torch.arange(ids.numel(), device=DEV)
print(f"data: {len(seqs)} seqs, {ids.numel()} tokens")

# ---- 2) 裸模型评估 ----
torch.cuda.set_device(0)
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)


def build_model():
    hf = AutoConfig.from_pretrained(PATH)
    m = Qwen3ForCausalLM(hf).to(DEV, dtype=hf.dtype)
    load_model(m, PATH)
    m.eval()
    return m


def eval_ce(model) -> tuple[float, float, int]:
    from nanovllm.utils.context import set_context, reset_context
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEV)
    set_context(True, cu, cu, T, T, torch.tensor([], dtype=torch.int32, device=DEV), None, None)
    with torch.inference_mode():
        hidden = model(ids, positions)
    reset_context()  # 之后分块调 lm_head（无 context 时直接 F.linear，不再按 cu_seqlens 取行）
    with torch.inference_mode():
        nll = 0.0
        cnt = 0
        for c0 in range(0, T, 512):
            logits = model.compute_logits(hidden[c0:c0 + 512])
            logp = F.log_softmax(logits.float(), dim=-1)
            nxt = torch.arange(c0, min(c0 + 512, T), device=DEV) + 1
            ok = (nxt < T) & ~torch.tensor([int(n.item()) in bounds for n in nxt], device=DEV)
            lab = ids[nxt[ok]]
            sel = logp[ok]  # [m, V]
            nll += -sel[torch.arange(lab.numel(), device=DEV), lab].sum().item()
            cnt += lab.numel()
    reset_context()
    return math.exp(nll / cnt), nll, cnt


from nanovllm.layers.linear import LinearBase
from nanovllm.layers.embed_head import ParallelLMHead

state = torch.load(os.path.join("results", "awq_scales.pt"), map_location="cpu")

for mode, setup in [
    ("fp16", None),
    ("int4(RTN)", "rtn"),
    ("awq(α搜索)", "awq"),
]:
    model = build_model()
    if setup == "rtn":
        for m in model.modules():
            if isinstance(m, LinearBase) and not isinstance(m, ParallelLMHead):
                m.quantize_int4()
    elif setup == "awq":
        for name, m in model.named_modules():
            if isinstance(m, LinearBase) and not isinstance(m, ParallelLMHead):
                m.quantize_int4(state[name])
    ppl, nll, cnt = eval_ce(model)
    print(f"{mode:<14} ppl={ppl:.4f}  (nll={nll:.1f} over {cnt} tokens)")
    del model
    torch.cuda.empty_cache()
