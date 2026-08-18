"""投机解码（n-gram/prompt-lookup）纯Python测试：草稿搜索、验收、调度、块记账。

GPU路径（引擎级等价性）由 benchmarks/_spec_equiv_check.py 在WSL中运行。
"""
import os

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.ngram import find_ngram_draft, verify_drafts
from conftest import _simulate_prefill


# ---------------------------------------------------------------------------
# find_ngram_draft：草稿搜索
# ---------------------------------------------------------------------------

def test_draft_basic_and_last_occurrence():
    # [1,2,3,4] 出现在位置0和位置5；取最近一次（位置5），草稿 = 其后token
    tokens = [1, 2, 3, 4, 9, 1, 2, 3, 4, 7, 8, 1, 2, 3, 4]
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=4, eos_id=0) == [7, 8, 1, 2]


def test_draft_no_prior_occurrence():
    # 严格递增序列：末尾窗口在历史上没有出现过 → 无草稿
    tokens = list(range(1, 17))
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=4, eos_id=0) == []


def test_draft_overlap_rule():
    # 相邻重复 [1,2,3,4][1,2,3,4]：前一次出现结束于 L-w（紧邻当前窗口）→ 合法，
    # 草稿 = 当前窗口内容本身（模式自我重复）
    tokens = [1, 2, 3, 4, 1, 2, 3, 4]
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=4, eos_id=0) == [1, 2, 3, 4]
    # 但窗口自身不能匹配自己：只有一次出现时无草稿（test_draft_no_prior_occurrence覆盖）


def test_draft_window_fallback():
    # 4-gram无匹配，2-gram有匹配 → 用2-gram的后续token
    tokens = [7, 8, 9, 10, 11, 12, 21, 22, 20, 21, 22]
    assert find_ngram_draft(tokens, window=4, min_window=2, max_len=4, eos_id=0) == [20, 21, 22]


def test_draft_eos_truncation():
    tokens = [1, 2, 3, 4, 5, 0, 6, 1, 2, 3, 4]  # 草稿区 [5,0,6]，EOS=0 截断
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=4, eos_id=0) == [5]


def test_draft_max_len_and_empty():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=2, eos_id=0) == [5, 6]
    assert find_ngram_draft(tokens, window=4, min_window=1, max_len=0, eos_id=0) == []
    assert find_ngram_draft([1, 2, 3], window=4, min_window=1, max_len=4, eos_id=0) == []


# ---------------------------------------------------------------------------
# verify_drafts：验收
# ---------------------------------------------------------------------------

def test_verify_all_accepted():
    # 全部匹配：drafts + bonus
    drafts, samples = [1, 2, 3], [1, 2, 3, 9]
    out, n_acc = verify_drafts(drafts, samples)
    assert n_acc == 4 and out == [1, 2, 3, 9]


def test_verify_partial():
    drafts, samples = [1, 2, 3], [1, 8, 3, 9]
    out, n_acc = verify_drafts(drafts, samples)
    assert n_acc == 2 and out == [1, 8]  # 第2个草稿被拒，输出目标采样8


def test_verify_first_rejected():
    drafts, samples = [1, 2, 3], [7, 2, 3, 9]
    out, n_acc = verify_drafts(drafts, samples)
    assert n_acc == 1 and out == [7]


def test_verify_no_draft():
    out, n_acc = verify_drafts([], [42])
    assert n_acc == 1 and out == [42]


# ---------------------------------------------------------------------------
# 调度器：spec模式
# ---------------------------------------------------------------------------

def _make_spec_scheduler(**kw):
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    kwargs = dict(model=path, num_kvcache_blocks=64, kvcache_block_size=256,
                  max_num_batched_tokens=2048, speculative="ngram")
    kwargs.update(kw)
    config = Config(**kwargs)
    return Scheduler(config)


def test_spec_schedule_pure():
    """running有草稿 → kind "spec"，行query长度=γ+1。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    # 内容重复（[1..8]*3）：末尾4-gram能匹配到前一次出现 → 草稿=重复内容
    tokens = list(range(1, 9)) * 3
    r = Sequence(tokens)
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    seqs, kind = sched.schedule()
    assert kind == "spec", kind
    assert seqs == [r]
    assert r.draft_tokens == [1, 2, 3, 4]  # 紧邻的前一次出现之后
    assert r.num_scheduled_tokens == 5    # γ+1
    assert not r.is_prefill


def test_spec_schedule_fallback_decode():
    """全部无草稿（内容不重复）→ 回落纯decode，draft_tokens清为None。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    r = Sequence(list(range(0, 1024)))  # 无重复n-gram → 无草稿
    _simulate_prefill(bm, r, 1024)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    seqs, kind = sched.schedule()
    assert kind == "decode", kind
    assert seqs == [r]
    assert r.draft_tokens is None
    assert r.num_scheduled_tokens == 1


def test_spec_schedule_mixed():
    """waiting非空 + running有草稿 → kind "mixed"，行序 [prefill..., verify...]。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    r = Sequence(list(range(1, 9)) * 3)
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    a = Sequence([11] * 1024)
    sched.add(a)
    seqs, kind = sched.schedule()
    assert kind == "mixed", kind
    assert seqs == [a, r]
    assert a.is_prefill and a.num_scheduled_tokens == 1024
    assert not r.is_prefill and r.num_scheduled_tokens == 5
    assert r.draft_tokens == [1, 2, 3, 4]


def test_spec_draft_budget_cap():
    """max_num_batched_tokens不足时截断草稿（保底γ=0的1-token行）。"""
    sched = _make_spec_scheduler(max_num_batched_tokens=6)
    bm = sched.block_manager
    r = Sequence(list(range(1, 9)) * 3)
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    r2 = Sequence([5] * 40)
    _simulate_prefill(bm, r2, r2.num_tokens)
    r2.status = SequenceStatus.RUNNING
    sched.running.extend([r, r2])
    seqs, kind = sched.schedule()
    assert kind == "spec"
    # 预算6：第1行γ+1=5（预算用5），第2行只剩1 → γ=0
    assert seqs[0].num_scheduled_tokens == 5
    assert seqs[1].num_scheduled_tokens == 1
    assert seqs[1].draft_tokens == []


def test_spec_draft_max_tokens_cap():
    """剩余输出预算不足时草稿为空（保留bonus位）→ 回落纯decode。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    tokens = list(range(1, 9)) * 3
    r = Sequence(tokens, sampling_params=_SP(max_tokens=1))
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    seqs, kind = sched.schedule()
    assert kind == "decode"
    assert seqs[0].draft_tokens is None


# 简化sampling params（避免依赖真实类）
class _SP:
    def __init__(self, temperature=0.6, max_tokens=100, ignore_eos=False):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos


def test_postprocess_spec_hash_range():
    """postprocess_spec：num_cached只计接受长度；哈希范围 = [len-1, len-1+n_acc)。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    tokens = list(range(1, 9)) * 3
    r = Sequence(tokens)
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    seqs, kind = sched.schedule()
    assert kind == "spec"
    drafts = r.draft_tokens
    # 模拟engine: 全接受（samples = drafts + bonus）
    accepted, n_acc = verify_drafts(drafts, drafts + [999])
    before = r.num_tokens
    sched.postprocess_spec(seqs, [accepted])
    assert r.num_tokens == before + n_acc
    assert r.num_cached_tokens == r.num_tokens  # 只含接受token
    assert r.num_scheduled_tokens == 0
    assert r.draft_tokens is None


def test_postprocess_spec_rejected():
    """部分接受：追加的token只到被拒处（含被拒处的目标采样）。"""
    sched = _make_spec_scheduler()
    bm = sched.block_manager
    tokens = list(range(1, 9)) * 3
    r = Sequence(tokens)
    _simulate_prefill(bm, r, r.num_tokens)
    r.status = SequenceStatus.RUNNING
    sched.running.append(r)
    seqs, kind = sched.schedule()
    drafts = r.draft_tokens
    samples = [drafts[0], 555, 666, 777, 888]  # 只接受第1个草稿
    accepted, n_acc = verify_drafts(drafts, samples)
    assert n_acc == 2
    sched.postprocess_spec(seqs, [accepted])
    assert r.num_tokens == len(tokens) + 2
    assert r.token_ids[-1] == 555
    assert r.num_cached_tokens == r.num_tokens


def test_hash_blocks_spec_range_small_blocks():
    """哈希范围跨块时正确发布（块尺寸8，直接测BlockManager）。"""
    Sequence.block_size = 8
    bm = BlockManager(64, 8)
    try:
        seq = Sequence(list(range(0, 6)))       # 位置0..5
        num_cached = bm.can_allocate(seq)
        bm.allocate(seq, num_cached)
        seq.num_cached_tokens = 6
        # 模拟调度器：写span [5, 10) 跨块0/块1
        assert bm.can_append_spec(seq, 5)
        bm.may_append_spec(seq, 5)
        # 模拟postprocess_spec：先追加已接受token，再哈希 [len-1-n_acc, len-1)
        seq.append_tokens([10, 11, 12, 13, 14])
        bm.hash_blocks(seq, False, start=seq.num_tokens - 5 - 1, end=seq.num_tokens - 1)
        # 块0（位置0..7）与块1（位置8..10）内容都被发布；被拒槽位（如有）不在内
        assert bm.blocks[seq.block_table[0]].hash != -1
        assert bm.blocks[seq.block_table[0]].token_ids == [0, 1, 2, 3, 4, 5, 10, 11]
        assert bm.blocks[seq.block_table[1]].token_ids == [12, 13, 14]
        # 哈希与内容一致：用同一内容重建序列应命中
        seq2 = Sequence([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14])
        assert bm.can_allocate(seq2) >= 1
    finally:
        Sequence.block_size = 256


# ---------------------------------------------------------------------------
# BlockManager：spec写span记账（小块尺寸直接测）
# ---------------------------------------------------------------------------

def test_can_append_spec_cross_block():
    Sequence.block_size = 8
    bm = BlockManager(64, 8)
    try:
        seq = Sequence(list(range(0, 6)))       # 位置0..5，块0
        num_cached = bm.can_allocate(seq)
        bm.allocate(seq, num_cached)
        seq.num_cached_tokens = 6
        # 写span [5, 5+5)：位置5..9 → 块0（0..7）和块1（8..15）→ 需要1个新块
        assert bm.can_append_spec(seq, 5)
        bm.may_append_spec(seq, 5)
        assert len(seq.block_table) == 2
        # 1-token写入与decode语义一致：位置5在块0内，不需要新块
        seq2 = Sequence(list(range(0, 6)))
        num_cached = bm.can_allocate(seq2)
        bm.allocate(seq2, num_cached)
        seq2.num_cached_tokens = 6
        assert bm.can_append_spec(seq2, 1)
        bm.may_append_spec(seq2, 1)
        assert len(seq2.block_table) == 1
    finally:
        Sequence.block_size = 256


def test_can_append_spec_shared_block_cow():
    """写span内的共享块需要COW副本预留。"""
    Sequence.block_size = 8
    bm = BlockManager(64, 8)
    try:
        a = Sequence(list(range(0, 12)))        # 位置0..11，块0和块1
        _simulate_prefill(bm, a, 12)
        # 共享块：b的前缀命中a的块（内容相同，先发布a的哈希）
        b = Sequence(list(range(0, 12)))
        num_cached = bm.can_allocate(b)
        assert num_cached >= 1
        bm.allocate(b, num_cached)
        assert bm.blocks[a.block_table[0]].ref_count > 1
        # a的写span [11, 11+2)：位置11在块1（与b共享）→ 需要1个COW副本
        assert bm.can_append_spec(a, 2)
        pair = bm.cow_block(a, 11)
        assert pair is not None
        old, new = pair
        assert old in b.block_table and new in a.block_table
        assert bm.blocks[old].ref_count == 1  # 复制后旧块只剩另一共享者
        # a的写span [11, 11+6)：跨到块2（位置16）→ 需要1个新块
        assert bm.can_append_spec(a, 6)
        bm.may_append_spec(a, 6)
        assert len(a.block_table) == 3
        bm.deallocate(b)
    finally:
        Sequence.block_size = 256
