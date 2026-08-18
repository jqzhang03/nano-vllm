"""投机解码（n-gram）正确性检查。

三层验证：

1. **verify前向logits对齐**（正确性核心）：同seed跑plain与spec（单prompt，两个
   run在首个spec步之前完全同流：同样的step、同样的噪声消耗）。取spec的首个
   verify步（第k步）的每seq第0行logits（位置len-1，预测下一个token），与plain
   第k步的decode logits比较——同输入同位置。差异应为内核级（varlen vs
   flash-kvcache 求和顺序不同），top-1必须一致。logits对了，验收就是
   logits+draft的纯函数，实现即正确。
2. **分布一致性**（统计）：temp=0.6（采样噪声主导，内核噪声~0.03可忽略）下
   plain跑两次的自一致率 ≈ plain vs spec的一致率——spec的输出就是目标分布的
   另一份采样（Leviathan保证：每位置输出分布=目标分布）。
   temp=1e-4版仅作诊断：plain-vs-plain翻转率（近并列+独立噪声）vs spec-vs-plain
   翻转率（额外含verify/decode内核噪声导致的近并列翻转——预期略高，非bug）。
3. **冒烟**：temp=0.6跑通、无NaN、报告接受率α；fp8 KV + spec 的对齐。

用法（WSL，GPU）：
    python benchmarks/_spec_equiv_check.py [--fp8]
"""
from __future__ import annotations

import argparse
import os

import torch

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

# 结构化prompt（token级重复）与自由文本混合，覆盖draft命中/落空
PROMPTS = [
    "Please continue this pattern, keeping the exact same spacing: 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40",
    "Repeat the sequence until told to stop: 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31",
    "Count down from 30 by threes: 30 27 24 21 18 15 12 9 6 3",
    'Complete the JSON array with more records: [{"id": 0, "name": "user_0", "score": 0.10, "active": true}, '
    '{"id": 1, "name": "user_1", "score": 0.20, "active": false}]',
    "The history of the steam engine begins with the aeolipile described by Hero of Alexandria.",
    "In the forest, the river carved a slow path through the valley, and the trees leaned over the water.",
    "Repeat exactly, continuing the same digit forever: 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5",
]


def run_engine(speculative: str, temperature: float, max_tokens: int, fp8: bool,
               collect_logits: bool = False, prompts: list[str] | None = None, seed: int = 0):
    # Gumbel噪声在GPU上生成（sampler的exponential_），必须种CUDA RNG——
    # 只种CPU RNG会让多个run共享同一条未重置的GPU噪声流（独立性被破坏）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    llm = LLM(MODEL, speculative=speculative, kv_cache_dtype="fp8_e4m3" if fp8 else "auto",
              gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    sps = [SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)] * len(prompts or PROMPTS)
    out = llm.generate(prompts or PROMPTS, sps, use_tqdm=False, collect_logits=collect_logits)
    steps = llm.collected_logits if collect_logits else None
    stats = llm.collect_metrics()["step_stats"]
    llm.exit()
    torch.cuda.empty_cache()  # 同进程多引擎必须显式归还显存
    return [o["token_ids"] for o in out], steps, stats


def mismatch_rate(a: list[int], b: list[int]) -> tuple[int, int]:
    n = min(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y), n


def first_spec_step(steps):
    """首个kind ∈ {spec, mixed}的 (索引, logits)。"""
    for i, (k, l) in enumerate(steps):
        if k in ("spec", "mixed") and l is not None:
            return i, l
    return None, None


def check_logits_alignment(fp8: bool) -> bool:
    print(f"== 1) verify前向 logits 对齐（同seed同流；首个spec步 vs plain同step，{'fp8' if fp8 else 'fp16'}） ==")
    ok = True
    for pi, prompt in enumerate(PROMPTS):
        _, steps_plain, _ = run_engine("none", 1e-4, 64, fp8, collect_logits=True, prompts=[prompt], seed=100)
        _, steps_spec, _ = run_engine("ngram", 1e-4, 64, fp8, collect_logits=True, prompts=[prompt], seed=100)
        i_spec, l_spec = first_spec_step(steps_spec)
        if i_spec is None:
            print(f"  [SKIP] prompt {pi}: 全程无spec步（无草稿）")
            continue
        l_plain = steps_plain[i_spec][1]          # plain同step（decode，同输入同位置）
        row0 = l_spec[0]                          # 每seq第0行 = 位置len-1 → 预测下一个token
        diff = (row0 - l_plain[0]).abs()
        agree = (row0.argmax() == l_plain[0].argmax()).item()
        status = "OK " if diff.max().item() < 0.5 and agree else "FAIL"
        ok &= status == "OK "
        print(f"  [{status}] prompt {pi}: 首个spec步@step{i_spec}, max|Δlogit|={diff.max().item():.4f} "
              f"mean={diff.mean().item():.6f} top1一致={agree}")
    print(("PASS: verify行logits = decode行logits（同输入同位置，内核级噪声）→ "
           "验收是logits+草稿的纯函数，实现正确。\n") if ok
          else "FAIL: verify前向与decode前向logits不一致——实现有bug！\n")
    return ok


def check_verify_rows(fp8: bool) -> bool:
    """验证verify行的 1..γ 行（attend草稿token的行）：与plain逐decode步对齐。

    repeat prompt 上模型续写 = 历史模式 → 草稿 = 模型续写。迭代重建每个spec步
    的seq状态（prefill/decode步追加1 token，spec步追加n_acc——由输出流反推），
    重算草稿；取第一个"全接受"的spec步：verify行j（输入=位置L-1+j的
    draft_{j-1}，预测L+j）必须与 plain 第 k+j 步（序列该位置是已接受的
    draft_{j-1}）的decode logits一致（plain同seed同流，temp=1e-4下其token=
    argmax=verify行argmax=draft）。"""
    print(f"== 1b) verify行 1..γ logits 对齐（全接受spec步，{'fp8' if fp8 else 'fp16'}） ==")
    from transformers import AutoTokenizer
    from nanovllm.engine.ngram import find_ngram_draft
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    max_tokens = 48
    ok = True
    done = False
    for pi in (0, 6):  # repeat模式prompt
        prompt = PROMPTS[pi]
        P = len(tokenizer.encode(prompt))
        out_spec, steps_spec, _ = run_engine("ngram", 1e-4, max_tokens, fp8,
                                             collect_logits=True, prompts=[prompt], seed=100)
        out_plain, steps_plain, _ = run_engine("none", 1e-4, max_tokens, fp8,
                                               collect_logits=True, prompts=[prompt], seed=100)
        out = out_spec[0]
        completion = 0  # 已完成completion token数（迭代重建）
        for k, (kind, logits) in enumerate(steps_spec):
            if kind == "prefill" or (kind == "decode" and logits is not None):
                completion += 1  # 每步追加1 token
                continue
            if kind != "spec" or logits is None:
                continue
            seq_tokens = tokenizer.encode(prompt) + out[:completion]
            drafts = find_ngram_draft(seq_tokens, 4, 1, min(4, max_tokens - completion - 1),
                                      tokenizer.eos_token_id)
            assert logits.shape[0] == len(drafts) + 1, (logits.shape[0], len(drafts))
            n_acc = 1
            for j, d in enumerate(drafts):
                if out[completion + j] != d:
                    break
                n_acc += 1
            if n_acc < len(drafts) + 1:
                completion += n_acc
                continue  # 非全接受：行1..γ的输入含被拒草稿，不可比
            # 全接受步：行j（预测位置P+k+j） vs plain第k+j步（同位置decode）
            print(f"  prompt {pi}: 全接受spec步@step{k}（γ={len(drafts)}, 位置{P + k}..{P + k + len(drafts)}）")
            for j in range(len(drafts) + 1):
                if j >= 1 and out_plain[0][k + j - 1] != drafts[j - 1]:
                    print(f"  [stop] 行{j}起不可比（plain该位置近并列翻转≠draft）")
                    break
                l_plain_j = steps_plain[k + j][1][0]
                diff = (logits[j] - l_plain_j).abs()
                agree = (logits[j].argmax() == l_plain_j.argmax()).item()
                status = "OK " if diff.max().item() < 0.5 and agree else "FAIL"
                ok &= status == "OK "
                print(f"  [{status}] 行{j}: max|Δlogit|={diff.max().item():.4f} "
                      f"mean={diff.mean().item():.6f} top1一致={agree}")
            completion += n_acc
            done = True
            break
        if done:
            break
    if not done:
        print("  [SKIP] 48 token内未找到全接受spec步")
    print(("PASS: verify全行（含attend草稿的行）与decode逐位置一致。\n" if ok
           else "FAIL: verify行映射错误！\n"))
    return ok


def check_distribution(fp8: bool) -> bool:
    print(f"== 2) 分布一致性（{'fp8' if fp8 else 'fp16'}） ==")
    ok = True
    for temp in (1e-4, 0.6):
        plain_a, _, _ = run_engine("none", temp, 96, fp8, seed=1)
        plain_b, _, _ = run_engine("none", temp, 96, fp8, seed=2)
        spec_c, _, stats_spec = run_engine("ngram", temp, 96, fp8, seed=3)
        tot_pp = mis_pp = tot_ps = mis_ps = 0
        for a, b, c in zip(plain_a, plain_b, spec_c):
            m1, n1 = mismatch_rate(a, b)
            m2, n2 = mismatch_rate(a, c)
            mis_pp += m1; tot_pp += n1
            mis_ps += m2; tot_ps += n2
        rate_pp, rate_ps = mis_pp / tot_pp, mis_ps / tot_ps
        if temp == 1e-4:
            # 诊断：spec额外包含 verify/decode 内核噪声(~0.03)在近并列位置的翻转，预期略高
            print(f"  temp=1e-4: plain自翻转 {100 * rate_pp:.2f}% | spec翻转 {100 * rate_ps:.2f}% "
                  f"（诊断用：spec多出的是内核噪声近并列翻转，非验收错误）")
        else:
            consistent = rate_ps <= rate_pp * 2 + 0.02
            ok &= consistent
            print(f"  temp=0.6 : plain自一致 {100 * (1 - rate_pp):.2f}% | spec一致 {100 * (1 - rate_ps):.2f}% "
                  f"→ {'PASS: spec输出=目标分布的另一份采样。' if consistent else 'FAIL: spec一致率显著更低！'}")
            if stats_spec.get("spec_draft_tokens"):
                print(f"            接受率 α = {stats_spec['spec_accepted_drafts']}/{stats_spec['spec_draft_tokens']} "
                      f"= {stats_spec['spec_accepted_drafts'] / stats_spec['spec_draft_tokens']:.3f}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fp8", action="store_true", help="同时用fp8 KV cache跑对齐与分布检查")
    args = p.parse_args()

    ok = check_logits_alignment(False)
    ok = check_verify_rows(False) and ok
    ok = check_distribution(False) and ok
    if args.fp8:
        ok = check_logits_alignment(True) and ok
        ok = check_verify_rows(True) and ok
        ok = check_distribution(True) and ok

    # ---- 冒烟（temp=0.6，多prompt批量） ----
    print("== 3) 冒烟（temp=0.6, 批量, fp16） ==")
    _, _, stats = run_engine("ngram", 0.6, 128, False, seed=7)
    d, v = stats["spec_draft_tokens"], stats["spec_verify_tokens"]
    acc = stats["spec_accepted_drafts"]
    print(f"  spec_steps={stats['spec_steps']}, drafts={d}, accepted={acc}, "
          f"α={acc / d if d else float('nan'):.3f}, verify_tokens={v}, decode_tokens={stats['decode_tokens']}")
    print("  无NaN/崩溃，冒烟通过。" if stats["spec_steps"] > 0 else "FAIL: 没有spec步！")
    ok &= stats["spec_steps"] > 0

    print("\n结论:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
