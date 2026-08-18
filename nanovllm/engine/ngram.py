"""n-gram（prompt-lookup）草稿生成与验收：投机解码第一阶段。

草稿源 = 序列自身历史（vLLM 的 PromptLookupWorker 同款思路，
`--speculative-model "[ngram]"`）：取末尾 w 个 token 作为窗口，在历史中找
"最近一次出现"（必须严格位于当前窗口之前），把上次出现之后紧跟着的 token
抄出来当草稿候选。无模型、零显存、零训练——草稿的唯一成本是 CPU 搜索。

验收（Leviathan et al. 2023 的分布保持性质）：因为草稿是确定性点质量分布，
"接受 iff 目标分布采样值 == 草稿，拒绝则输出该采样值"严格保持目标分布——
输出分布与不做投机时逐 token 相同（不是近似，是精确相等）。
"""
from __future__ import annotations


def find_ngram_draft(
    token_ids: list[int],
    window: int,
    min_window: int,
    max_len: int,
    eos_id: int,
) -> list[int]:
    """返回草稿 token 列表（可能为空）。

    - 窗口从 window 递减到 min_window：先用长窗口匹配（更可靠），找不到再退
      短窗口（覆盖更广，vLLM 的 ngram-prompt-lookup-max/min 同款策略）；
    - 前一次出现必须严格位于当前窗口之前（不重叠）：p + w <= L - w；
    - 草稿截断在 EOS 前：被接受的草稿不该含 EOS（否则序列会被草稿意外终止）；
    - max_len 由调用方按剩余输出预算封顶（含 1 个 bonus 位）。
    """
    L = len(token_ids)
    if max_len <= 0 or L < 2 * min_window:
        return []
    for w in range(window, min_window - 1, -1):
        if L <= w:
            continue
        pat = token_ids[L - w:]
        # 从末尾向前找最近一次出现：p 从 L-2w 递减到 0（p+w <= L-w）。
        # 先比首 token 再比整窗口：纯索引比较，避免反复切片。
        for p in range(L - 2 * w, -1, -1):
            if token_ids[p] != pat[0]:
                continue
            if token_ids[p + 1:p + w] != pat[1:]:
                continue
            end = min(p + w + max_len, L)
            drafts = token_ids[p + w:end]
            for i, t in enumerate(drafts):
                if t == eos_id:
                    drafts = drafts[:i]
                    break
            return drafts
    return []


def verify_drafts(drafts: list[int], samples: list[int]) -> tuple[list[int], int]:
    """验收一行的草稿：samples[i] 是位置 len-1+i 的目标采样，验证草稿 drafts[i]；
    samples[-1] 是全接受时的 bonus。

    返回 (接受的 token 列表, 接受数 n_acc)。n_acc = 连续匹配数 + 1（bonus 恒接受）：
    - 全部匹配: drafts + [bonus]（γ+1 个）
    - 第 j 个草稿被拒: drafts[:j] + [samples[j]]（j+1 个，被拒处输出目标采样）
    - 无草稿（γ=0）: [samples[0]]（普通 decode 一行）
    """
    n_acc = 1  # bonus 恒接受
    for i, d in enumerate(drafts):
        if samples[i] != d:
            break
        n_acc += 1
    return drafts[:n_acc - 1] + [samples[n_acc - 1]], n_acc
