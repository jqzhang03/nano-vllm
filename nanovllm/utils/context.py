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

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None,
                is_mixed=False, prefill_block_tables=None, n_prefill_tokens=0,
                is_spec=False, n_prefill_rows=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables,
                       is_mixed, prefill_block_tables, n_prefill_tokens,
                       is_spec, n_prefill_rows)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
