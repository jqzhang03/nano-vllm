from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from conftest import _simulate_prefill


def test_compute_hash_is_deterministic():
    token_ids_a = [1, 2, 3, 4]
    token_ids_b = [1, 2, 3, 5]
    h1 = BlockManager.compute_hash(token_ids_a)
    h2 = BlockManager.compute_hash(token_ids_a)
    h3 = BlockManager.compute_hash(token_ids_b)
    assert h1 == h2
    assert h1 != h3


def test_empty_cache():
    bm = BlockManager(64, 256)
    seq = Sequence([0] * 1024)
    x = bm.can_allocate(seq)
    assert x == 0


def test_reuses_all_full_blocks():
    bm = BlockManager(64, 256)
    seq1 = Sequence([0] * 1024)
    _simulate_prefill(bm, seq1, seq1.num_tokens)
    seq2 = Sequence([0] * 1024)
    x = bm.can_allocate(seq2)
    assert x == 4


def test_partial_last_block_is_cached():
    bm = BlockManager(64, 256)
    seq1 = Sequence([0] * 300)
    _simulate_prefill(bm, seq1, seq1.num_tokens)
    seq2 = Sequence([0] * 300)
    x = bm.can_allocate(seq2)
    assert x == 2


def test_refcount():
    bm = BlockManager(64, 256)
    seq1 = Sequence([0] * 300)
    _simulate_prefill(bm, seq1, 300)
    seq2 = Sequence([0] * 300)
    _simulate_prefill(bm, seq2, 0)
    shared_block = seq1.block_table[0]
    assert bm.blocks[shared_block].ref_count == 2
    bm.deallocate(seq2)
    assert bm.blocks[shared_block].ref_count == 1
    bm.deallocate(seq1)
    assert bm.blocks[shared_block].ref_count == 0
    assert len(bm.free_block_ids) == 64


def test_copy_on_write():
    