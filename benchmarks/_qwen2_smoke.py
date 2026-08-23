"""Qwen2/Qwen3 多模型适配冒烟测试 + HF 参考 logits 对比。

用法:
  python benchmarks/_qwen2_smoke.py [model] [quant] [streaming] [kv] [--check-hf]
    model    模型目录（默认 ~/huggingface/Qwen2.5-0.5B）
    quant    none|int4|w8a8|sparse24|awq（默认 none）
    streaming 1|0（默认自动：大模型+量化时自动开启）
    kv       auto|fp8_e4m3（默认 auto）
    --check-hf  加载 HF transformers 参考模型对比 prefill logits（仅 fp16 有意义）

输出：模型结构关键字段（验证 qwen2 端口读对 config）、生成结果、
显存（权重/峰值/KV容量）。--check-hf 时额外打印 logits max diff / top-1 一致率。
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
    model = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen2.5-0.5B")
    quant = sys.argv[2] if len(sys.argv) > 2 else "none"
    streaming_arg = sys.argv[3] if len(sys.argv) > 3 else ""
    kv = sys.argv[4] if len(sys.argv) > 4 else "auto"
    check_hf = "--check-hf" in sys.argv
    enforce_eager = "--enforce-eager" in sys.argv
    dump_path = None
    if "--dump-logits" in sys.argv:
        dump_path = sys.argv[sys.argv.index("--dump-logits") + 1]

    streaming = {"1": True, "0": False}.get(streaming_arg, None)
    kw = dict(kv_cache_dtype=kv, quantization=quant, enforce_eager=enforce_eager)
    if streaming is not None:
        kw["streaming_load"] = streaming

    t0 = time.perf_counter()
    llm = LLM(model, max_model_len=2048, **kw)
    runner = llm.model_runner
    hf = llm.config.hf_config
    print(f"\n=== model [{model}] model_type={hf.model_type} "
          f"streaming={runner.streaming} ===")
    print(f"layers={hf.num_hidden_layers} hidden={hf.hidden_size} "
          f"heads={hf.num_attention_heads} kv_heads={hf.num_key_value_heads} "
          f"head_dim={getattr(hf, 'head_dim', None)} "
          f"intermediate={hf.intermediate_size} vocab={hf.vocab_size}")
    print(f"attention_bias={getattr(hf, 'attention_bias', None)} "
          f"tie_word_embeddings={getattr(hf, 'tie_word_embeddings', None)} "
          f"rope_theta={getattr(hf, 'rope_theta', None)}")
    # 偏置张量存在性：config 期望的 attention_bias 必须与 checkpoint 里的 bias 张量一致
    from safetensors import safe_open
    st_files = [os.path.join(model, f) for f in os.listdir(model) if f.endswith(".safetensors")]
    bias_keys = []
    with safe_open(st_files[0], "pt", "cpu") as f:
        bias_keys = [k for k in f.keys() if k.endswith(".bias")]
    exp_bias = getattr(hf, "attention_bias", True)
    has_qkv_bias = any("q_proj.bias" in k for k in bias_keys)
    print(f"checkpoint bias tensors: {len(bias_keys)} (qkv={has_qkv_bias}) | "
          f"config attention_bias={exp_bias} -> {'一致' if has_qkv_bias == exp_bias else '不一致!'}")

    # 权重/缓存显存
    tensors = [t for t in list(runner.model.parameters()) + list(runner.model.buffers())
               if t.is_cuda]
    w_bytes = sum(t.numel() * t.element_size() for t in tensors)
    print(f"weights+buffers on GPU: {w_bytes/1e9:.3f} GB  "
          f"| KV blocks: {llm.config.num_kvcache_blocks} "
          f"({llm.config.num_kvcache_blocks * llm.config.kvcache_block_size/1000:.0f}k tokens)")

    # warmup + 生成
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    logits_first = None
    if check_hf or dump_path:
        llm.generate(PROMPTS[:2], SamplingParams(temperature=0.6, max_tokens=1),
                     use_tqdm=False, collect_logits=True)
        for kind, lg in llm.collected_logits:
            if kind == "prefill" and lg is not None:
                logits_first = lg  # [n_seqs, vocab] fp32
                break
        if dump_path is not None:
            torch.save(logits_first, dump_path)
            print(f"dumped prefill logits to {dump_path}")
    if "--dump-hidden" in sys.argv:
        # 直接前向：不经调度器，dump 最后一层 hidden（lm_head 输入）供对比
        from nanovllm.engine.sequence import Sequence
        hpath = sys.argv[sys.argv.index("--dump-hidden") + 1]
        bm = llm.scheduler.block_manager
        seqs = [Sequence(llm.tokenizer.encode(p)) for p in PROMPTS[:2]]
        for s in seqs:
            s.num_scheduled_tokens = len(s)
            bm.allocate(s, bm.can_allocate(s))
        # 同时抓 layer-0 输出与 embed 输出：定位发散发生在哪一层/哪一步
        l0_out, emb_out, rec = {}, {}, {}
        attn = llm.model_runner.model.model.layers[0].self_attn
        orig_fwd = attn.forward

        def rec_fwd(positions, hidden_states):
            qkv = attn.qkv_proj(hidden_states)
            rec["qkv"] = qkv.clone()
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            q = q.view(-1, attn.num_heads, attn.head_dim)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            rec["q_pre"] = q.clone()
            rec["k_pre"] = k.clone()
            q, k = attn.rotary_emb(positions, q, k)
            rec["q_post"] = q.clone()
            rec["k_post"] = k.clone()
            o = attn.attn(q, k, v)
            rec["attn_out"] = o.clone()
            return attn.o_proj(o.flatten(1, -1))

        h1 = llm.model_runner.model.model.layers[0].register_forward_hook(
            lambda m, i, o: l0_out.__setitem__("v", o[0].clone()))
        h2 = llm.model_runner.model.model.embed_tokens.register_forward_hook(
            lambda m, i, o: emb_out.__setitem__("v", o[0].clone()))
        attn.forward = rec_fwd
        try:
            _, _, hidden = llm.model_runner.call("run", seqs, "prefill", True, True)
        finally:
            attn.forward = orig_fwd
            h1.remove()
            h2.remove()
        for s in seqs:
            bm.deallocate(s)
        torch.save({"hidden": hidden, "layer0": l0_out["v"], "embed": emb_out["v"], **rec}, hpath)
        print(f"dumped prefill hidden + layer0 + embed + layer0-attn internals to {hpath}")
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=32), use_tqdm=False)
    print(f"init+generate wall: {time.perf_counter()-t0:.1f}s")
    for o in outs:
        print(f"  -> {o['text'][:80]!r}")
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1e9
    print(f"peak allocated: {peak:.2f} GB")

    if check_hf:
        if logits_first is None:
            print("WARN: no prefill logits collected; skipped HF comparison")
            llm.exit()
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, use_fast=True)
        # 不用 device_map（需要 accelerate 包）：CPU 加载后整体搬到 cuda
        ref = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16).to("cuda").eval()
        ref_logits = []
        with torch.no_grad():
            for p in PROMPTS[:2]:
                ids = torch.tensor([tok.encode(p)], device="cuda")
                out = ref(input_ids=ids, use_cache=False)
                ref_logits.append(out.logits[0, -1].float())
        ref_logits = torch.stack(ref_logits)
        diff = (logits_first - ref_logits).abs()
        top1 = (logits_first.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
        # 分布诊断：diff 是否集中在少数尾部 token（bf16 双实现 + 不同注意力内核的正常量级）
        flat = diff.flatten()
        q = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99, 1.0], device=flat.device))
        print(f"\nHF reference logits: max diff {diff.max().item():.4f} | "
              f"mean diff {diff.mean().item():.6f} | top-1 agree {100*top1:.1f}%")
        print(f"diff percentiles p50/p90/p99/max: {q.tolist()}")
        print("(判定：top-1 100% = 端口正确；mean diff 0.05~0.1 为 bf16 + flash-attn vs SDPA 的正常偏差，"
              "端口错误（缺偏置/错打包/错head_dim）会塌到 top-1 < 30% + mean diff > 1)")
        assert top1 == 1.0, "top-1 mismatch vs HF reference!"
        assert diff.mean().item() < 0.5, "mean logit diff too large vs HF reference!"
        del ref
        torch.cuda.empty_cache()
    llm.exit()
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
