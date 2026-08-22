"""Medusa集成检查：引擎实际产生的draft与模型LM head argmax的重合率。

直接驱动引擎循环，每步schedule后记录draft_tokens，同时用该步verify的hidden
（engine返回）离线算模型argmax对照。若draft≈模型argmax重合率高（>30%）→
集成正确，α低是模型可预测性；若~0 → 集成bug（行索引/时机/dtype）。

用法（WSL，GPU）：
    python benchmarks/_medusa_integration.py [--medusa-path results/medusa_heads.pt]
"""
from __future__ import annotations

import argparse
import os

import torch

from nanovllm import LLM, SamplingParams

MODEL = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
TEXT = ("Repeat exactly, continuing the same digit forever: " + " ".join(["5"] * 40))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medusa-path", default="results/medusa_heads.pt")
    args = p.parse_args()

    llm = LLM(MODEL, speculative="medusa", medusa_path=args.medusa_path, gpu_memory_utilization=0.9)
    llm.generate(["warm up"] * 4, SamplingParams(temperature=0.6, max_tokens=8), use_tqdm=False)
    tokenizer = llm.tokenizer
    prompts = [tokenizer.encode(TEXT), tokenizer.encode("The capital of France is")]
    sps = [SamplingParams(temperature=0.6, max_tokens=48, ignore_eos=True)] * len(prompts)
    for pr, sp in zip(prompts, sps):
        llm.add_request(pr, sp)

    gamma = llm.config.max_draft_len
    heads = llm.model_runner.medusa_heads
    n_draft = n_match_model = n_match_true = 0
    n_steps = 0
    while not llm.is_finished():
        seqs, kind = llm.scheduler.schedule()
        for old_id, new_id in llm.scheduler.cow_pairs:
            llm.model_runner.call("cow_block", old_id, new_id)
        return_hidden = kind == "spec"
        result = llm.model_runner.call("run", seqs, kind, False, return_hidden)
        token_ids, hidden = result if return_hidden else (result, None)
        # 离线对照：用verify hidden第0行（t_last）算模型argmax（预测下一个token）
        model_top1 = None
        if hidden is not None and hidden.size(0) > 0:
            with torch.inference_mode():
                lg = llm.model_runner.model.compute_logits(hidden[0].unsqueeze(0).to(torch.bfloat16))
                model_top1 = lg.argmax(-1).item()
        if any(s.draft_tokens is not None for s in seqs):
            token_lists, n_dec, n_draft_s, n_acc_s, _, n_acc_list = llm._verify(seqs, token_ids)
            # 第一个verify行的draft_1（验收前的）
            d1 = next((s.draft_tokens[0] for s in seqs if s.draft_tokens), None)
            n_rows_list = [s.num_scheduled_tokens if s.draft_tokens is not None else 0 for s in seqs]
            llm.scheduler.postprocess_spec(seqs, token_lists)
            if hidden is not None and d1 is not None:
                n_draft += 1
                if model_top1 is not None and d1 == model_top1:
                    n_match_model += 1
                # 验收后新t_last的hidden（第n_acc行）的模型argmax 应 == 下轮draft_1
                idx0 = n_rows_list[0]  # 第一个seq（bs=2时这里近似；仅诊断用）
                if model_top1 is not None and n_acc_list[0] > 0 and idx0 > n_acc_list[0]:
                    with torch.inference_mode():
                        lg = llm.model_runner.model.compute_logits(
                            hidden[n_acc_list[0]].unsqueeze(0).to(torch.bfloat16))
                        nxt = lg.argmax(-1).item()
                    print(f"step{len(seqs)}: draft_1={d1} 模型argmax(t_last)={model_top1} "
                          f"n_acc={n_acc_list[0]} | 验收后模型argmax={nxt}（下轮draft应≈此）")
            if kind == "spec":
                llm._medusa_drafts(seqs, hidden, n_acc_list, n_rows_list)
            n_steps += 1
        else:
            llm.scheduler.postprocess(seqs, token_ids)
    print(f"\ndraft_1 vs 模型argmax: {n_match_model}/{n_draft} = "
          f"{n_match_model / n_draft if n_draft else 0:.2f}")
    llm.exit()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
