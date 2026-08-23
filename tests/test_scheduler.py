from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from conftest import _simulate_prefill
from nanovllm.config import Config
import os


def _make_scheduler(max_num_batched_tokens=2048, kv_swap=True):
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    config = Config(model=path, num_kvcache_blocks=64, kvcache_block_size=256,
                    max_num_batched_tokens=max_num_batched_tokens, kv_swap=kv_swap)
    return Scheduler(config)


def test_scheduler():
    seqs = [Sequence([0] * 1024) for _ in range(4)]
    scheduler = _make_scheduler()
    bm = scheduler.block_manager
    _simulate_prefill(bm, seqs[0], seqs[0].num_tokens)
    seqs2 = [Sequence([0] * 1024) for _ in range(4)]
    for seq in seqs2:
        scheduler.add(seq)
    scheduled, kind = scheduler.schedule()
    assert kind == "decode"
    assert len(scheduled) == 4
    for se in scheduled:
        assert se.num_scheduled_tokens == 1
    assert len(scheduler.waiting) == 0
    assert set(scheduled) == set(scheduler.running)


def test_mixed_batch():
    """waiting与running都非空时返回混合批次：早完成prefill的序列立即decode（vLLM V1同款）。"""
    scheduler = _make_scheduler()
    bm = scheduler.block_manager
    # 1个已过prefill的running序列（内容与其他序列不同，避免前缀缓存命中干扰）
    r = Sequence([7] * 1024)
    _simulate_prefill(bm, r, 1024)
    r.status = SequenceStatus.RUNNING
    scheduler.running.append(r)
    # 2个waiting序列（内容互不相同）
    a = Sequence([1] * 1024)
    b = Sequence([2] * 1024)
    scheduler.add(a)
    scheduler.add(b)

    seqs1, kind1 = scheduler.schedule()
    assert kind1 == "mixed", kind1
    # 批次行序：prefill行在前、decode行在后
    assert seqs1 == [a, r], [s.seq_id for s in seqs1]
    assert a.num_scheduled_tokens == 1024 and a.is_prefill
    assert r.num_scheduled_tokens == 1 and not r.is_prefill
    assert r in scheduler.running and a in scheduler.running  # a完成prefill → RUNNING

    seqs2, kind2 = scheduler.schedule()
    assert kind2 == "mixed"
    assert seqs2 == [b, r, a]
    assert b.num_scheduled_tokens == 1024
    assert all(s.num_scheduled_tokens == 1 for s in (r, a))

    seqs3, kind3 = scheduler.schedule()
    assert kind3 == "decode"  # waiting空 → 纯decode
    assert set(seqs3) == {r, a, b}


def test_pure_phases():
    """waiting空 → 纯decode；running空 → 纯prefill（与旧行为一致）。"""
    scheduler = _make_scheduler()
    bm = scheduler.block_manager
    r = Sequence([7] * 1024)
    _simulate_prefill(bm, r, 1024)
    r.status = SequenceStatus.RUNNING
    scheduler.running.append(r)

    kinds = []
    for _ in range(3):
        _, kind = scheduler.schedule()
        kinds.append(kind)
    assert kinds == ["decode", "decode", "decode"], kinds

    scheduler2 = _make_scheduler()
    for i in range(2):
        scheduler2.add(Sequence([i + 1] * 1024))
    kinds = []
    for _ in range(2):
        _, kind = scheduler2.schedule()
        kinds.append(kind)
    # 2048预算下一个prefill步装下两个seq；第二步waiting空 → 纯decode
    assert kinds == ["prefill", "decode"], kinds


def test_kv_swap_state():
    """KV swap 抢占状态机：KV完整的decode序列抢占时换出（保留num_cached_tokens、
    进swapped队列），free块恢复后换入（直接decode，免重新prefill）。"""
    scheduler = _make_scheduler(kv_swap=True)
    bm = scheduler.block_manager
    s = Sequence([7] * 1024)
    _simulate_prefill(bm, s, 1024)
    s.status = SequenceStatus.RUNNING
    s.is_prefill = False       # decode序列（KV覆盖到len-1，最后token的KV本步写）
    s.num_cached_tokens = s.num_tokens - 1
    scheduler.running.append(s)

    scheduler.preempt(s)
    assert s.swapped and s in scheduler.swapped
    assert s.num_cached_tokens == s.num_tokens - 1  # KV 保留（免重新prefill）
    assert not s.is_prefill                          # 恢复走 decode 而非 prefill
    assert len(scheduler.swap_pairs) == 1 and scheduler.swap_pairs[0][3] == "out"
    assert not scheduler.is_finished()          # swapped 队列非空不算完成

    # engine 完成 GPU→CPU 拷贝后释放块
    block_ids = scheduler.swap_pairs[0][1]
    scheduler.finish_swap_out(s, block_ids)
    assert not s.block_table
    assert len(bm.free_block_ids) == 64         # 块全部回收

    # free 恢复 → 换入（直接 decode）
    scheduler._try_swap_in()
    assert s in scheduler.running and not s.swapped
    assert s.num_cached_tokens == s.num_tokens - 1
    assert len(scheduler.swap_pairs) == 2 and scheduler.swap_pairs[1][3] == "in"
    assert scheduler.swap_pairs[1][0] is s


def test_kv_swap_disabled_recompute():
    """kv_swap=False 时抢占走 recompute（原行为：释放块、回 waiting、下次 prefill）。"""
    scheduler = _make_scheduler(kv_swap=False)
    bm = scheduler.block_manager
    s = Sequence([7] * 1024)
    _simulate_prefill(bm, s, 1024)
    s.status = SequenceStatus.RUNNING
    s.num_cached_tokens = s.num_tokens
    scheduler.running.append(s)

    scheduler.preempt(s)
    assert not s.swapped and s in scheduler.waiting
    assert s.is_prefill                        # 恢复走 prefill（重算）
    assert s.num_cached_tokens == 0            # 缓存被清
    assert scheduler.num_swaps == 0
