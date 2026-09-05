"""MLA + fp8 KV（阶段 2b 扩展 M2，GPU）：toy DeepSeek-V2。

验证链：
1) 行 gather+反量化内核 vs 手工 dequant（同 ops 顺序 → 逐位一致）；
2) 引擎 fp8 KV vs 引擎 bf16 KV：同种子低温（≈argmax）轨迹，逐 decode 步
   top-1 一致率 + logits 差（量化噪声带内）；分歧步只计 top-1 翻转；
3) prefill 末行 vs 稠密参考（fp8 不参与 fresh 行注意力 → 应保持 2a 的
   100% top-1 parity）。

用法: python benchmarks/_mla_fp8_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from _ds_e2e import build_toy, build_ref_model, nano_dense_logits

TOY = os.path.expanduser("~/ds_toy")
V = 32000


def unit_gather_dequant():
    """gather+dequant 内核 vs 手工（同一 ops 序列 → 逐位一致）。"""
    from nanovllm.layers.attention_mla import mla_gather_dequant
    kv_lora, rope, nb, B = 128, 16, 3, 256
    D = kv_lora + rope
    g = torch.Generator(device="cuda").manual_seed(3)
    cache = (torch.randn(nb * B, D, device="cuda", generator=g) * 0.5)
    c_scale, r_scale = 0.0013, 0.02
    qc = (cache[:, :kv_lora] * (1 / c_scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
    qr = (cache[:, kv_lora:] * (1 / r_scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
    fused = torch.cat([qc.view(torch.uint8), qr.view(torch.uint8)], -1) \
        .view(torch.float8_e4m3fn).view(nb, B, D).contiguous()
    idx = torch.tensor([0, 1, B + 3, 2 * B + 9, 5], device="cuda")
    out = mla_gather_dequant(fused, idx, c_scale, r_scale, kv_lora, rope,
                             torch.bfloat16)
    ref = torch.cat([(qc.float() * c_scale).to(torch.bfloat16),
                     (qr.float() * r_scale).to(torch.bfloat16)], -1)[idx.long()]
    assert torch.equal(out, ref), "gather+dequant 内核与手工不一致"
    print(f"unit gather/dequant OK (n={out.shape[0]}, D={D})")


def main():
    unit_gather_dequant()
    build_toy()
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOY, use_fast=True)

    def _ids(text: str):
        return [t % V for t in _tok.encode(text)]

    TOKENS = [_ids("The capital of France is a"), _ids("Once upon a time")]
    WARM = [_ids("warm up")] * 2
    TEMP = 1e-9  # ≈argmax：两引擎轨迹只在 top-1 翻转处分歧

    def run_engine(fp8: bool):
        from nanovllm import LLM, SamplingParams
        torch.manual_seed(99)
        llm = LLM(TOY, max_model_len=128, quantization="none", kv_swap=False,
                  enforce_eager=True,
                  kv_cache_dtype="fp8_e4m3" if fp8 else "auto")
        llm.generate(WARM, SamplingParams(temperature=TEMP, max_tokens=2),
                     use_tqdm=False)
        out = llm.generate(TOKENS, SamplingParams(temperature=TEMP,
                                                  max_tokens=25),
                           use_tqdm=False, collect_logits=True)
        pref = [lg for kind, lg in llm.collected_logits
                if kind == "prefill" and lg is not None]
        dec = [lg for kind, lg in llm.collected_logits
               if kind == "decode" and lg is not None]
        llm.exit()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return pref, dec, [o["token_ids"] for o in out]

    print("[engine fp8 KV]")
    pf8, df8, comp_f8 = run_engine(True)
    print("[engine bf16 KV]")
    p16, d16, comp_16 = run_engine(False)

    # (3) prefill 末行 vs 稠密参考（fp8 引擎不读 fp8 缓存 → 100% top-1）
    ref = build_ref_model()
    ref_last = torch.stack([nano_dense_logits(ref, t, len(t)) for t in TOKENS])
    eng = torch.cat(pf8).float()
    if eng.shape[0] != len(ref_last):
        eng = eng[-len(ref_last):]
    d = (eng - ref_last).abs()
    top1 = (eng.argmax(-1) == ref_last.argmax(-1)).float().mean().item()
    print(f"[prefill last-row fp8] max {d.max().item():.4f} mean "
          f"{d.mean().item():.6f} top-1 {100*top1:.1f}%")
    assert top1 == 1.0, "fp8 引擎 prefill 末行 top-1 失真"

    # (2) decode：fp8 vs bf16。步骤上下文 = 双方各自已生成 token；分歧后上下文
    # 不再可比 → 只统计共同前缀内的行（seq i 的步 s 有效 ⇔ s ≤ 共同前缀长）。
    def cpl(a, b):
        k = 0
        while k < min(len(a), len(b)) and a[k] == b[k]:
            k += 1
        return k

    cpls = [cpl(comp_f8[i], comp_16[i]) for i in range(len(comp_f8))]
    n = min(len(df8), len(d16))
    agree_tot = agree_ok = flip = 0
    maxd = 0.0
    mean_sum = 0.0
    for s in range(n):
        a = df8[s].float()
        b = d16[s].float()
        for i in range(a.shape[0]):
            if s > cpls[i]:
                continue  # 该 seq 已分歧（上下文不可比）
            agree_tot += 1
            ok = int(a[i].argmax() == b[i].argmax())
            agree_ok += ok
            flip += 0 if ok else 1
            dd = (a[i] - b[i]).abs()
            maxd = max(maxd, dd.max().item())
            mean_sum += dd.mean().item()
    mean_d = mean_sum / max(agree_tot, 1)
    print(f"[decode fp8-vs-bf16] steps {n} 可比行 {agree_tot}: top-1 一致 "
          f"{100*agree_ok/max(agree_tot,1):.2f}% (翻转 {flip}) | "
          f"max {maxd:.4f} mean {mean_d:.5f}")
    assert flip <= max(2, agree_tot // 50), "fp8 KV decode 翻转过多"
    print("MLA FP8 KV E2E OK")


if __name__ == "__main__":
    main()
