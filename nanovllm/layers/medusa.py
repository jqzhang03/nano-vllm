"""Medusa 多头：投机解码第二阶段的草稿源（vLLM Medusa 同款思路）。

结构：γ+1 个小型 MLP 头，共享同一个 backbone 最后一层 hidden（norm 之后、
LM head 之前的 h_t），head_k(h_t) 预测位置 t+k 的 token。draft 阶段 = 一次
批量前向（无顺序依赖），成本 µs~ms 级；之后走既有的线性 verify 路径。

语义约定（训练与推理一致）：
  - 训练标签：head_k 的标签 = token_{t+k}（自蒸馏：模型自己生成的数据）。
  - 推理：验收后新 t_last 在位置 L'-1。
      * 非全接受（n_acc ≤ γ）：t_last 的 hidden = 当前 verify 输入的第 n_acc 行
        （输入行 i 在位置 len-1+i）→ head_1..head_γ 预测位置 L'..L'+γ-1 ✓
      * 全接受（n_acc = γ+1）：bonus 是采样产物、无 hidden → 用第 γ 行（位置
        len+γ-1）+ head_2..head_{γ+1}（预测位置 len+γ+1..，正好是 L'..）✓
    因此头数 = max_draft_len + 1，全接受时 head_1 的输出丢弃（预测的是已采样的
    bonus 位置，无损失——头很小）。

每个头：hidden → medusa_hidden（SiLU）→ vocab。输出层 256×vocab 是主要参数
（~39M/头）；0.6B 模型上用 256 维瓶颈控制总参数（4 头 ≈ 120M ≈ 模型 20%）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MedusaHead(nn.Module):

    def __init__(self, hidden: int, medusa_hidden: int, vocab: int):
        super().__init__()
        self.down = nn.Linear(hidden, medusa_hidden, bias=False)
        self.up = nn.Linear(medusa_hidden, vocab, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, hidden] → logits [B, vocab]（预测 h 位置之后第 k 个 token）。"""
        return self.up(F.silu(self.down(h)))


class MedusaHeads(nn.Module):
    """γ+1 个头（全接受偏移用）；draft 取 head[shift:shift+γ] 的 argmax。"""

    def __init__(self, n_heads: int, hidden: int, medusa_hidden: int, vocab: int):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([MedusaHead(hidden, medusa_hidden, vocab)
                                    for _ in range(n_heads)])

    def forward(self, h: torch.Tensor) -> list[torch.Tensor]:
        """h: [B, hidden] → 每头 logits [B, vocab]（保持 batch 维）。"""
        return [head(h) for head in self.heads]
