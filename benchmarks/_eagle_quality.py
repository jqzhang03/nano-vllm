"""EAGLE 草稿层推理质量检查：teacher-forced 输入（真 hidden + 真下一 token）下，
草稿分布 top-1/argmax 与目标模型真实下一 token 的命中率。"""
import os
import sys

import torch
import torch.nn.functional as F

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

PATH = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B")
EAGLE = os.path.expanduser(sys.argv[2]) if len(sys.argv) > 2 else "results/eagle_layer.pt"

llm = LLM(PATH, speculative="eagle", eagle_path=EAGLE, max_model_len=2048)
llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)

tokenizer = llm.tokenizer
prompts = ["Repeat exactly, continuing the same digit forever: " + " ".join(["5"] * 40),
           "The capital of France is", "To bake a chocolate cake, you need",
           "Machine learning is", "Quantum computing works by"]
seqs = [Sequence(tokenizer.encode(p)) for p in prompts]
for s in seqs:
    s.num_scheduled_tokens = len(s)

bm = llm.scheduler.block_manager
for s in seqs:
    bm.allocate(s, bm.can_allocate(s))
_, _, hidden = llm.model_runner.call("run", seqs, "prefill", True, True)  # [T, H]
for s in seqs:
    bm.deallocate(s)

# teacher-forced：位置 t 输入 (h_t, e(w_{t+1}))，草稿分布预测 w_{t+2}
layer = llm.model_runner.eagle_layer
embed_w = llm.model_runner.model.model.embed_tokens.weight
lm_head_w = llm.model_runner.model.lm_head.weight
# 对照：fp32 版（训练口径）vs bf16 版（引擎口径）
for tag, dtype in (("bf16(引擎)", torch.bfloat16), ("fp32(训练)", torch.float32)):
    layer_f = layer.to(dtype)
    n_match = n = 0
    for s, hh in zip(seqs, torch.split(hidden, [len(s) for s in seqs])):
        hh = hh.to(dtype)
        toks = s.token_ids
        for t in range(len(toks) - 2):
            h_t = hh[t:t + 1]
            emb = F.embedding(torch.tensor([toks[t + 1]], device=hh.device),
                              embed_w.to(dtype))
            with torch.inference_mode():
                h_pred = layer_f(h_t, emb)
                logits = F.linear(h_pred.to(dtype), lm_head_w.to(dtype))[0]
            n += 1
            n_match += logits.argmax().item() == toks[t + 2]
    print(f"{tag}: {n} 位置 | 草稿argmax==真实下一token: {100 * n_match / n:.1f}%")
llm.exit()
print("（模型自身 top-1 可预测性 ~35% 是上限；>35% 说明草稿层学到了目标分布）")
