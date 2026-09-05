"""真实 DeepSeek-V2-Lite 流式 int4 全模型（阶段 2b 扩展，GPU，跑通≠写对）。

完整性验证策略（模型级数学已由 toy 位级链验证；真实权重的数值锚在
_v2lite_4l_parity.py 用 4 层真权重组装）：
- 本文档只做"真实权重 + 真实分布的引擎级证据"：生成质量（多 prompt 文本
  人工可读性）、吞吐、显存、量化内存账本（int4 + kv_b 反量化副本占比）；
- 不做 HF 对照（31GB fp16 全模型在 16GB 卡/7GB RAM 上无参考物）。

用法: python benchmarks/_v2lite_full_probe.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

MODEL = os.path.expanduser("~/huggingface/DeepSeek-V2-Lite")


def main():
    from nanovllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(MODEL, max_model_len=1024, quantization="int4",
              kv_swap=False, enforce_eager=True,
              max_num_seqs=8, max_num_batched_tokens=4096)
    mr = llm.model_runner
    print(f"[engine] streaming={mr.streaming} mla={mr._mla_model} "
          f"mla_dense_decode={mr._mla_dense_decode} enforce_eager={mr.enforce_eager}")
    free0, total0 = torch.cuda.mem_get_info()
    print(f"[engine] 启动完成 {time.time()-t0:.0f}s | 空闲显存 {free0/1e9:.2f}/{total0/1e9:.2f}GB")
    # 权重量化内存账本（真实模型证据）
    from nanovllm.layers.linear import LinearBase
    b = sum(p.numel() * p.element_size() for p in mr.model.parameters()
            if not p.is_meta and p.dtype not in
            (torch.float8_e4m3fn, torch.uint8, torch.int8))
    f = sum(p.numel() for p in mr.model.parameters()
            if p.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int8))
    wdeq = sum(v.numel() * v.element_size()
               for m in mr.model.modules() if isinstance(m, LinearBase)
               for name, v in m.named_buffers() if name == "w_deq")
    print(f"[weights] bf16/fl32 显存 {b/1e9:.2f}GB | 打包(int4/fp8等)参数 {f/1e6:.0f}M "
          f"| kv_b w_deq 总量 {wdeq/1e6:.1f}MB")
    sp = SamplingParams(temperature=0.6, max_tokens=96)
    prompts = [
        "深度学习模型训练的核心挑战之一是计算资源的分配与优化。请解释分布式训练中数据并行的基本思想。",
        "The capital of Australia is Canberra, not Sydney. Write three sentences about kangaroos.",
        "列出三个计算机操作系统，并分别说明它们的主要设计目标。",
    ]
    t1 = time.time()
    out = llm.generate(prompts, sp, use_tqdm=True)
    met = llm.collect_metrics()
    ss = met["step_stats"]
    dt = time.time() - t1
    ncomp = sum(len(o["token_ids"]) for o in out)
    print(f"[gen] {dt:.0f}s | prefill tok {ss['prefill_tokens']} | decode tok {ncomp} "
          f"| decode tok/s ≈ {ncomp / max(ss['decode_time'], 1e-9):.1f}")
    for o in out:
        txt = o["text"].replace("\n", " ")
        print("---", txt[:300])
    llm.exit()
    print("V2-LITE REAL INT4 PROBE OK")


if __name__ == "__main__":
    main()
