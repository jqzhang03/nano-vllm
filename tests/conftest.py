from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def _simulate_prefill(bm: BlockManager, seq: Sequence, num_tokens: int):
    num_cached = bm.can_allocate(seq)
    bm.allocate(seq, num_cached)
    seq.num_scheduled_tokens = num_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += num_tokens