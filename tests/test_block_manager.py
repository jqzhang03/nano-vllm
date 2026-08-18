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
    bm = BlockManager(64, 256)
    seq1 = Sequence([0] * 300)
    _simulate_prefill(bm, seq1, 300)
    seq2 = Sequence([0] * 300)
    _simulate_prefill(bm, seq2, 0)
    old_id = seq2.block_table[-1]
    assert bm.blocks[old_id].ref_count == 2          # 共享部分块
    pair = bm.cow_block(seq2, seq2.num_tokens - 1)   # seq2要写末块（位置299）
    assert pair is not None
    old, new = pair
    assert old == old_id
    assert seq2.block_table[-1] == new and new != old   # 换表
    assert bm.blocks[old].ref_count == 1                # 旧块归seq1
    assert bm.blocks[new].ref_count == 1                # 新块归seq2
    assert seq1.block_table[-1] == old                  # seq1仍引用旧块
    assert bm.blocks[new].hash == -1                    # 新块不进缓存，待重新发布
    assert bm.cow_block(seq2, seq2.num_tokens - 1) is None  # 私有块不再COW
    # 释放后空闲池完整回收
    bm.deallocate(seq2)
    bm.deallocate(seq1)
    assert len(bm.free_block_ids) == 64


def _rehash(bm: BlockManager, seq: Sequence):
    """模拟decode一步后的postprocess：hash_blocks + 游标前进 + 清零。"""
    seq.num_scheduled_tokens = 1
    bm.hash_blocks(seq)
    seq.num_cached_tokens += 1
    seq.num_scheduled_tokens = 0


def test_twin_hash_no_keyerror():
    """两个内容相同的块（COW副本与原块）共享同一哈希时，dict条目会被后者覆盖；
    其中一个内容变化后，无条件 del 会误删/重复删导致KeyError——守卫版本必须不崩。"""
    bm = BlockManager(64, 256)
    seq1 = Sequence([7] * 300)
    _simulate_prefill(bm, seq1, 300)
    seq2 = Sequence([7] * 300)
    _simulate_prefill(bm, seq2, 0)
    pair = bm.cow_block(seq2, 299)           # seq2写前COW：复制块1
    assert pair is not None
    # 两个序列追加相同token → 两块内容相同 → 相同哈希，dict条目被后写者覆盖
    seq1.append_token(8)
    seq2.append_token(8)
    _rehash(bm, seq1)
    _rehash(bm, seq2)
    # 再各自追加相同token9 → 两块的旧哈希条目必须被守卫式删除，不能KeyError
    seq1.append_token(9)
    _rehash(bm, seq1)
    seq2.append_token(9)
    _rehash(bm, seq2)
    # 状态一致性：dict中每个条目指向的块，其当前哈希必须与key一致（无指向已变化内容的脏条目）。
    # 注意两个内容相同的"双胞胎"块可能共享同一哈希、dict只指向其中一块——这是合法的。
    for h, bid in bm.hash_to_block_id.items():
        assert bm.blocks[bid].hash == h