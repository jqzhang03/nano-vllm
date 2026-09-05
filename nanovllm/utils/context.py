from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    # 注意：字段顺序与set_context的位置传参约定一致——新字段必须加在末尾
    is_prefill: bool = False            # 纯prefill批次
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None   # 全批次（prefill槽位 + decode槽位）
    context_lens: torch.Tensor | None = None   # decode组（混合批次）或全部（纯decode）
    block_tables: torch.Tensor | None = None   # decode组（混合批次）或全部（纯decode）
    is_mixed: bool = False              # 混合批次（prefill行在前 + decode行在后）
    prefill_block_tables: torch.Tensor | None = None  # prefill组中需读缓存的行（前缀复用）
    n_prefill_tokens: int = 0                     # prefill组token数（decode组在q/k中的起点）
    is_spec: bool = False               # 投机verify步（LM head保留全行；mixed+spec时全批次varlen）
    n_prefill_rows: int = 0             # 投机混合步中prefill组的行数（spec组在cu_seqlens_q中的起点）
    # ---- 阶段2：滚动缓冲 / MLA 行信息（新字段一律加在末尾） ----
    chunk_starts: torch.Tensor | None = None      # [decode组行数] ring行首块序号j0（非ring=0）
    mla_pre_starts: torch.Tensor | None = None    # [varlen行] MLA cache-shaped行缓存起点（行query start）
    mla_pre_idx: torch.Tensor | None = None       # [Σstart] MLA前缀行槽位索引（行主序）
    mla_dec_starts: torch.Tensor | None = None    # [decode行] MLA稠密兜底行key长度（=context_lens）
    mla_dec_idx: torch.Tensor | None = None       # [Σlens] MLA decode兜底行槽位索引（行主序）
    # ---- 阶段 2b 扩展：环 spec（投机 verify 行的环内装配，MHA 滚动模型） ----
    # verify/prefill varlen 行的缓存段不再按 [0, start)（flash 分页语义，环表会错位），
    # 而按"环内窗口相关行 [max(j0·B, start-W+1), start)"装配成稠密 K/V——
    # ring_starts[r] = 该行装配的缓存行数；ring_idx = 槽位索引（行主序，Σring_starts）。
    # 装配行段起点 = 窗口语义起点（start-W+1 截齐）→ flash varlen 的 window_size
    # 掩码在段内下标上仍精确（装配外行本来就无资格被 attend）。
    ring_starts: torch.Tensor | None = None
    ring_idx: torch.Tensor | None = None
    # ---- 阶段 2b 扩展：split（交替窗口）滚动 —— full 池（global 层）镜像 ----
    # block_tables/…= 环池侧（local 层）；full_* = full 池侧（global 层）：
    # full_block_tables（decode/纯prefill分页读）、full_slot_mapping（全批次写槽）、
    # full_prefill_block_tables（混合批次 prefill 组）。split 之外为 None。
    full_block_tables: torch.Tensor | None = None
    full_slot_mapping: torch.Tensor | None = None
    full_prefill_block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None,
                is_mixed=False, prefill_block_tables=None, n_prefill_tokens=0,
                is_spec=False, n_prefill_rows=0,
                chunk_starts=None, mla_pre_starts=None, mla_pre_idx=None,
                mla_dec_starts=None, mla_dec_idx=None,
                ring_starts=None, ring_idx=None,
                full_block_tables=None, full_slot_mapping=None,
                full_prefill_block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables,
                       is_mixed, prefill_block_tables, n_prefill_tokens,
                       is_spec, n_prefill_rows,
                       chunk_starts, mla_pre_starts, mla_pre_idx,
                       mla_dec_starts, mla_dec_idx,
                       ring_starts, ring_idx,
                       full_block_tables, full_slot_mapping,
                       full_prefill_block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
