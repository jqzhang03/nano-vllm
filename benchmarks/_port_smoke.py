"""新模型端口冒烟：Mistral-7B / Gemma-2（2B/9B）int4 流式加载 + 生成 + 显存账本。

用法: python benchmarks/_port_smoke.py <model_dir> [quant] [--long]
  quant    int4|fp8|none（默认 int4）
  --long   追加 5000-token 长上下文 prompt（Mistral/Gemma-2-9B 会越过 sliding_window
           =4096，压 SWA 窗口路径；gemma-2-2b 全 global 不触发窗口，仅验证长上下文）
输出：模型结构关键字段（sliding_window / layer_types / softcap）、权重与 KV 显存、
生成结果、长上下文 logits 有限性检查。
"""
import os
import sys
import time

import torch

from nanovllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "To bake a chocolate cake, you need",
    "Machine learning is",
    "The best way to learn programming is",
]


def main():
    model = os.path.expanduser(sys.argv[1])
    quant = sys.argv[2] if len(sys.argv) > 2 else "int4"
    do_long = "--long" in sys.argv
    max_len = 2048 if not do_long else 8192

    t0 = time.perf_counter()
    llm = LLM(model, max_model_len=max_len, quantization=quant)
    runner = llm.model_runner
    hf = llm.config.hf_config
    print(f"\n=== model [{model}] model_type={hf.model_type} quant={quant} "
          f"streaming={runner.streaming} ===")
    print(f"layers={hf.num_hidden_layers} hidden={hf.hidden_size} "
          f"heads={hf.num_attention_heads} kv_heads={hf.num_key_value_heads} "
          f"head_dim={getattr(hf, 'head_dim', None)} intermediate={hf.intermediate_size} "
          f"vocab={hf.vocab_size}")
    print(f"sliding_window={getattr(hf, 'sliding_window', None)} "
          f"tie_word_embeddings={getattr(hf, 'tie_word_embeddings', None)}")
    if hf.model_type == "gemma2":
        from nanovllm.models.gemma2 import gemma2_layer_types
        lt = gemma2_layer_types(hf, hf.num_hidden_layers)
        print(f"layer_types: {lt[:14]}{'...' if len(lt) > 14 else ''} "
              f"(global={lt.count('global')}, sliding={lt.count('sliding')})")
        print(f"attn_logit_softcapping={getattr(hf, 'attn_logit_softcapping', None)} "
              f"final_logit_softcapping={getattr(hf, 'final_logit_softcapping', None)} "
              f"query_pre_attn_scalar={getattr(hf, 'query_pre_attn_scalar', None)}")

    tensors = [t for t in list(runner.model.parameters()) + list(runner.model.buffers())
               if t.is_cuda]
    w_bytes = sum(t.numel() * t.element_size() for t in tensors)
    print(f"weights+buffers on GPU: {w_bytes/1e9:.3f} GB  "
          f"| KV blocks: {llm.config.num_kvcache_blocks} "
          f"({llm.config.num_kvcache_blocks * llm.config.kvcache_block_size/1000:.0f}k tokens)")

    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=32), use_tqdm=False)
    print(f"init+generate wall: {time.perf_counter()-t0:.1f}s")
    for o in outs:
        print(f"  -> {o['text'][:80]!r}")
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1e9
    print(f"peak allocated: {peak:.2f} GB")

    if do_long:
        # 长上下文：5000 token（Mistral / Gemma-2-9B local 层越过窗口 4096）
        tok = llm.tokenizer
        words = " machine learning inference optimization quantization attention"
        long_text = (words * 700)[:5000]
        ids = tok.encode(long_text)
        print(f"\nlong-context: prompt={len(ids)} tokens (window={getattr(hf, 'sliding_window', None)})")
        t1 = time.perf_counter()
        o = llm.generate([long_text], SamplingParams(temperature=0.6, max_tokens=16), use_tqdm=False)
        dt = time.perf_counter() - t1
        print(f"  -> {o[0]['text'][:60]!r}  ({dt:.1f}s, {len(ids)/dt:.0f} tok/s prefill)")
        # 窗口掩码出错（如全掩列 NaN）会在生成长度/内容上暴露：token 数必须 >= 1
        assert o[0]["token_ids"], "long-context generation produced no tokens!"
        print(f"  KV blocks used: {max(len(s.block_table) for s in llm.scheduler.running) if llm.scheduler.running else 'n/a'}")

    llm.exit()
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
