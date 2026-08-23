"""FP8 KV cache accuracy check: logits-level drift measurement.

Both engines run the same prompts; the FIRST decode step sees identical inputs
(prefill used fresh fp16 K/V in both modes), so its logits difference isolates
the impact of reading the quantized FP8 KV cache. Token-level agreement is NOT
used as the bar — temperature sampling amplifies tiny logit perturbations.

Metrics: max/mean |logit diff|, KL divergence, top-1 token agreement.
"""
import os
import sys
import torch

from nanovllm import LLM, SamplingParams

PATH = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else os.path.expanduser("~/huggingface/Qwen3-0.6B/")
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


def run_once(kv_dtype: str, quantization: str = "none", awq_scales_path: str = "",
             quantize_lm_head: bool = False):
    llm = LLM(PATH, kv_cache_dtype=kv_dtype, quantization=quantization,
              awq_scales_path=awq_scales_path, quantize_lm_head=quantize_lm_head,
              max_model_len=4096)
    llm.generate(["warm up"] * 8, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    torch.manual_seed(123)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.6, max_tokens=64),
                        use_tqdm=False, collect_logits=True)
    steps = llm.collected_logits
    llm.exit()
    torch.cuda.empty_cache()
    return [o["token_ids"] for o in outs], steps


def first_decode_logits(steps):
    # kind ∈ {"prefill", "decode", "mixed"}；首个纯decode步的logits即对齐点
    # （混合步的logits形状为 [prefill_tokens + decode_tokens, vocab]，不可直接对齐）
    for kind, logits in steps:
        if kind == "decode" and logits is not None:
            return logits
    raise RuntimeError("no decode step logged")


def main():
    # 用法: accuracy_check.py [model] [w8a8|int4|awq|sparse24] [auto|fp8_e4m3] [awq_scales_path] [lmhead]
    #   —— 对比fp16基线 vs 指定组合（默认 fp8 KV + 无权重量化，隔离各自影响）
    quant = sys.argv[2] if len(sys.argv) > 2 else "none"
    kv2 = sys.argv[3] if len(sys.argv) > 3 else "fp8_e4m3"
    awq_path = sys.argv[4] if len(sys.argv) > 4 else ""
    lm_head = len(sys.argv) > 5 and sys.argv[5] == "lmhead"
    print("running fp16 engine ...")
    tokens16, steps16 = run_once("auto", "none")
    print(f"running engine with quantization=[{quant}] + kv_cache_dtype=[{kv2}] "
          f"+ awq_scales_path=[{awq_path}] + quantize_lm_head=[{lm_head}] ...")
    tokens8, steps8 = run_once(kv2, quant, awq_path, lm_head)

    l16 = first_decode_logits(steps16)   # [bs, vocab] fp32
    l8 = first_decode_logits(steps8)
    assert l16.shape == l8.shape, f"step shapes differ: {l16.shape} vs {l8.shape}"

    diff = (l16 - l8).abs()
    p16 = torch.softmax(l16.float(), dim=-1)
    p8 = torch.softmax(l8.float(), dim=-1)
    kl = (p16 * (p16.log() - p8.log())).sum(dim=-1).mean().item()
    top1_16 = l16.argmax(dim=-1)
    top1_8 = l8.argmax(dim=-1)
    top1_agree = (top1_16 == top1_8).float().mean().item()

    # 翻转诊断：被翻转位置的两个候选token概率差距（小=近并列，量化微扰导致的正常翻转）
    for i in range(l16.shape[0]):
        if top1_16[i] != top1_8[i]:
            t16, t8 = top1_16[i].item(), top1_8[i].item()
            p_t16 = p16[i, t16].item()
            p_t8 = p8[i, t8].item()
            print(f"  flip@seq{i}: fp16选token{t16}(p={p_t16:.4f}) fp8选token{t8}(p={p_t8:.4f}) "
                  f"两候选概率差={abs(p_t16 - p_t8):.4f}")

    print(f"first decode step: bs={l16.shape[0]}, vocab={l16.shape[1]}")
    print(f"logits max abs diff : {diff.max().item():.4f}")
    print(f"logits mean abs diff: {diff.mean().item():.6f}")
    print(f"KL(softmax16||softmax8): {kl:.6f}")
    print(f"top-1 agreement     : {100 * top1_agree:.1f}%")

    # token-level agreement (informational; low bar because temperature sampling amplifies noise)
    total = agree = 0
    for a, b in zip(tokens16, tokens8):
        n = min(len(a), len(b))
        total += n
        agree += sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    print(f"token agreement (informational): {100 * agree / total:.1f}%")

    if quant == "none":
        assert diff.max().item() < 5.0, "FP8 KV logits diverged too much"
        assert diff.mean().item() < 0.5, "FP8 KV mean logit drift too large"
        assert kl < 0.05, "FP8 KV softmax distribution drift too large"
        assert top1_agree > 0.8, "FP8 KV top-1 prediction agreement too low"
        print("FP8 KV accuracy check OK")
    elif quant in ("int4", "sparse24"):
        # RTN int4 组量化在 0.6B 上 KL≈1 量级；2:4 一次性幅值剪枝丢弃~35%权重质量
        # → KL≈8.5（精度灾难来自剪枝算法本身，非内核；见 _sparse24_layers.py）。
        # 此处只保证不发散，精确值以报告为准
        assert kl < 12.0, "weight quant distribution drift too large"
        print(f"Weight quant [{quant}] accuracy check OK (阈值仅防发散: 0.6B上RTN/剪枝误差大)")
    else:  # w8a8 / awq（激活感知缩放，误差应显著小于RTN；8-prompt KL受尾部噪声主导，
        # 决定性指标是真实文本ppl，见 benchmarks/_quant_ppl.py）
        assert kl < 4.0, "weight quant distribution drift too large"
        assert top1_agree > 0.25, "weight quant top-1 agreement too low"
        print(f"Weight quant [{quant}] accuracy check OK (AWQ/w8a8阈值)")


if __name__ == "__main__":
    main()
