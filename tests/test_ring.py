"""SWA 滚动缓冲 CPU 单测（阶段 2b，无需 GPU）。

覆盖 BlockManager 环语义：
- 驱逐后驻留内容必须覆盖 [max(0, N−W−slack), N)（含 spec 回读余量）；
- 稳态表长 ≤ ring_cap = (W+slack−1)//B + 2；
- refcount 守卫：滚动块私有、可释放（表头恒 ref_count==1）；
- 小块池下长解码不新增内存（can_append 靠驱逐腾块，不依赖 free 池）。
"""
from collections import deque

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def _bm(num_blocks, B, W, slack=2):
    return BlockManager(num_blocks, B, rolling_window=W, ring_slack=slack)


def test_ring_cap_formula():
    """表长上界 = (W+slack−1)//B + 2（含未对齐窗口边界的头块余量）。"""
    B, W = 256, 512
    m = _bm(64, B, W, slack=2)
    assert m.ring_cap == (512 + 2 - 1) // 256 + 2 == 4
    m2 = _bm(64, B, W, slack=6)
    assert m2.ring_cap == (512 + 6 - 1) // 256 + 2 == 4


def test_ring_content_and_cap_sweep():
    """长解码（多轮跨窗）模拟：驻留内容覆盖窗口+余量，表长有界。

    逐 token 模拟调度器时序：len%256==1 时 can_append→may_append（驱逐先于
    分配），token 追加后逻辑长度 +1。全程不检查 free 池（驱逐自产自销）。
    """
    B, W = 256, 512
    slack = 2
    m = _bm(8, B, W, slack=slack)          # 极小池：只能靠驱逐循环
    seq = Sequence(list(range(400)))       # prompt 400 token（2 块）
    m.allocate(seq, 0)                     # 滚动 prefill：全私有
    assert seq.kv_j0 == 0
    N = len(seq)
    assert len(seq.block_table) == 2
    for _ in range(3000):                  # 解码 3000 token（> 5 个窗口）
        # 边界时 must_append（预演调度器）：can_append 的 free 依赖先于驱逐
        if N % B == 1:
            ok = m.can_append(seq)
            assert ok, "环解码在驱逐可腾块时不应依赖 free 池"
            m.may_append(seq)
        N += 1
        seq.token_ids.append(0)
        seq.num_tokens = N
        # 驻留覆盖：任意需保留 token ≥ N−W−slack 必须在驻留块内
        if N > W + slack:
            resident_first = seq.kv_j0 * B
            need_first = N - W - slack
            assert resident_first <= need_first, \
                f"N={N} resident 起点 {resident_first} 晚于需求 {need_first}"
        # 块计数有界（稳态）
        if N > W + slack + B:
            assert len(seq.block_table) <= m.ring_cap, \
                f"N={N} 表长 {len(seq.block_table)} > cap {m.ring_cap}"
        # refcount 守卫：表头块独占（环可释放的前提）
        if seq.block_table:
            front = m.blocks[seq.block_table[0]]
            assert front.ref_count == 1
            assert front.hash == -1
    # 驱逐真实发生（kv_j0 > 0）且内容窗口对齐
    assert seq.kv_j0 > 0
    assert len(seq.block_table) <= m.ring_cap
    # 释放干净：全部块回池
    m.deallocate(seq)
    assert len(m.used_block_ids) == 0
    assert len(m.free_block_ids) == 8
    assert seq.kv_j0 == 0


def test_ring_evict_then_alloc_reuses_pool():
    """驱逐 = 释放回池再分配：长期解码后 used 块数有界（内存账本）。"""
    B, W = 256, 512
    m = _bm(16, B, W, slack=2)
    seq = Sequence(list(range(600)))       # 3 块 prompt
    m.allocate(seq, 0)
    used_peak = len(m.used_block_ids)
    N = len(seq)
    for _ in range(2000):
        if N % B == 1:
            m.can_append(seq)
            m.may_append(seq)
        N += 1
        seq.token_ids.append(0)
        seq.num_tokens = N
        used_peak = max(used_peak, len(m.used_block_ids))
    # prompt 600=3块；解码稳态窗口内容 3-4 块 → used 峰值 ≤ 4
    assert used_peak <= m.ring_cap, f"used 峰值 {used_peak} 超窗口 {m.ring_cap}"


def test_ring_two_seqs_share_pool():
    """两序列并发解码共享小池：驱逐互不干扰（块私有，refcount 守卫恒真）。"""
    B, W = 256, 512
    m = _bm(12, B, W, slack=2)
    s1 = Sequence(list(range(300)))
    s2 = Sequence(list(range(300)))
    m.allocate(s1, 0)
    m.allocate(s2, 0)
    n1 = n2 = len(s1)
    assert len(m.used_block_ids) <= 12
    for i in range(1600):
        for seq, n in ((s1, n1), (s2, n2)):
            if n % B == 1:
                assert m.can_append(seq)
                m.may_append(seq)
            seq.token_ids.append(0)
            seq.num_tokens = n + 1
        n1 += 1
        n2 += 1
        # 两序列稳态 ≤ 2×cap + prompt 前 2 块余量
        assert len(m.used_block_ids) <= 2 * m.ring_cap + 2, \
            f"used {len(m.used_block_ids)}"
    m.deallocate(s1)
    m.deallocate(s2)
