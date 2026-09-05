from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int,
                 rolling_window: int | None = None, ring_slack: int = 0):
        """滚动缓冲（SWA 环，rolling_window 非 None 时启用）：

        每序列块表 = **窗口内容的物理清单**（按逻辑块序号升序，第 i 项是第
        kv_j0+i 个块；旧块在解码越过窗口后立即释放，表长 ≈ 窗口/块大小 + 2）。
        key 位置 = (kv_j0+i)·B + 槽内偏移——flash-attn 从表下标推位置会错位，
        必须走自研内核（Context.chunk_starts 传每行 kv_j0，见 attention.py）。

        环的代价（诚实标注）：①滚动序列的块**永不共享**（哈希发布/命中全关），
        前缀缓存对这个模型失效——refcount 守卫因此恒为 1，环回收无需拉伸；
        ②环语义需要知道"未来回读深度"：spec verify 行会回读窗口前 γ 个 key →
        驱逐阈值留 ring_slack（= max_draft_len + 2）的余量；
        ③纯 decode 稳态：追加块时先释放已死的前块（净零 free 消耗），
        解码内存有界；prompt 本身的 KV 在首次越过窗口前线性增长（同 vLLM
        掩码式 SWA），解码后逐步收敛到窗口。
        """
        self.block_size = block_size # 块大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 所有可用块的编号
        self.hash_to_block_id: dict[int, int] = dict() # 哈希值到块id的映射，用于前缀缓存
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 空闲块队列
        self.used_block_ids: set[int] = set() # 已使用块集合
        self.rolling = rolling_window is not None
        self._win = rolling_window or 0
        self._slack = ring_slack if self.rolling else 0

    # 环容量上界（= 最多在表的块数；窗口 W 内含未对齐边界 → +2 块）
    @property
    def ring_cap(self):
        return (self._win + self._slack - 1) // self.block_size + 2

    def _front_dead(self, seq: Sequence, n_tokens: int) -> bool:
        """前块整块落在驱逐阈值外：块内最大 token 位置 (front+1)·B − 1
        < n_tokens − win − slack ⇔ (front+1)·B ≤ n_tokens − win − slack。"""
        if not seq.block_table or not self.rolling:
            return False
        return (seq.kv_j0 + 1) * self.block_size <= n_tokens - self._win - self._slack

    def _evict_front(self, seq: Sequence):
        """释放表头块（窗口外内容）。滚动块私有（无共享/无哈希）→ refcount 守卫：
        理论恒 1，这里显式断言——若未来放开共享，这里就是"守卫"的落点。"""
        assert seq.block_table
        block = self.blocks[seq.block_table.pop(0)]
        assert block.ref_count == 1 and block.hash == -1, \
            "滚动块被共享/已发布哈希——环回收不安全（refcount 守卫）"
        block.ref_count -= 1
        self._deallocate_block(block.block_id)
        seq.kv_j0 += 1

    # 修饰为类方法，不需要实例化BlockManager即可调用，第一个参数必须是cls
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # 如果有前一块的哈希值，则把前面块的哈希值转换成字节串喂进h中
        if prefix != -1:
            # 将整型转换为8字节的字节串，并以小端字节序存储
            h.update(prefix.to_bytes(8, "little"))
        
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 滚动缓冲：前缀缓存停用（窗口内容会过期、块不共享）→ 一律全新分配
        if self.rolling:
            if len(self.free_block_ids) < seq.num_blocks:
                return -1
            return 0
        h = -1
        # 可复用块的个数
        num_cached_blocks = 0
        # 需要新分配块的个数
        num_new_blocks = seq.num_blocks
        last_cached_id = -1  # 最后一个命中的缓存块id
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 计算当前块的链式哈希
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 找不到或者遇到了哈希碰撞，直接退出
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            last_cached_id = block_id
            # 如果块在已使用队列中，只需要将对应块id的引用计数+1即可，不需要重新分配
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 部分块共享后，若写起点（缓存末尾）落在被共享的缓存块内，
        # 该块在本次prefill中会被写入，触发COW需要额外预留一个空闲块
        if num_cached_blocks > 0 and last_cached_id in self.used_block_ids:
            cached_tokens = (num_cached_blocks - 1) * self.block_size + len(self.blocks[last_cached_id].token_ids)
            if cached_tokens % self.block_size != 0:
                num_new_blocks += 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        seq.kv_j0 = 0
        if self.rolling:
            # 全新私有块（滚动模型 can_allocate 恒返回 0）
            for _ in range(seq.num_blocks):
                seq.block_table.append(self._allocate_block())
            seq.num_cached_tokens = 0
            return
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        if num_cached_blocks == 0:
            seq.num_cached_tokens = 0
        else:
            seq.num_cached_tokens = (num_cached_blocks - 1) * self.block_size + len(seq.block(num_cached_blocks - 1))

    def allocate_private(self, seq: Sequence):
        """KV swap 换入专用：分配全新私有块（不查前缀缓存、不参与共享、不发布哈希）。

        KV 内容由 ModelRunner 从 CPU 缓冲拷回（bit-exact），逻辑上已缓存到
        num_cached_tokens（= len-1，decode 序列最后生成的 token 的 KV 本步才写）。
        **只分配 cached tokens 的块（ceil(num_cached_tokens/256)）**——待写 token
        的块由 may_append 在 decode 时正常分配，避免双重分配（num_blocks 含待写块，
        与正常 decode 路径的块表语义不一致 → swap_in 后 may_append 又加一块）。
        恢复后的 decode 步由 postprocess 的 hash_blocks 重新发布哈希。
        滚动缓冲：decode 序列逻辑长度含已驱逐 token，块数 = **表内现存窗口块**
        （ceil(num_cached_tokens/B) − kv_j0），kv_j0 保留（表头仍是原 chunk 序号）。
        """
        assert not seq.block_table
        if self.rolling:
            n = max(1, (seq.num_cached_tokens + self.block_size - 1)
                    // self.block_size - seq.kv_j0)
        else:
            n = (seq.num_cached_tokens + self.block_size - 1) // self.block_size
        for _ in range(n):
            seq.block_table.append(self._allocate_block())

    def release_blocks(self, block_ids: list[int]):
        """按显式块 id 列表释放（KV swap 换出完成后）：refcount--，为 0 回 free 池。"""
        for block_id in block_ids:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        seq.kv_j0 = 0

    def can_append(self, seq: Sequence) -> bool:
        # 需要多少个空闲块：
        # 1. 末块已满，追加的token需要新块（len % block_size == 1）
        # 2. 末块是共享的部分块（ref_count > 1），写入前需要COW复制一块
        need = 0
        if len(seq) % self.block_size == 1:
            need += 1
            # 滚动缓冲：新块可以"驱逐已死前块"腾出（净零 free 消耗）
            if self.rolling and self._front_dead(seq, len(seq) + 1):
                need -= 1
        if (len(seq) % self.block_size != 1 and seq.block_table
                and self.blocks[seq.block_table[-1]].ref_count > 1):
            need += 1
        return len(self.free_block_ids) >= need

    def can_append_spec(self, seq: Sequence, n: int) -> bool:
        """投机verify行：写span = [len-1, len-1+n)（n = γ+1 个 token）。

        需要：span 覆盖到的新块数 + span 内被共享块（ref_count>1）的COW副本数。
        n 可以跨块（γ+1 个 token 越过块边界），与 can_append 的 1-token 特例不同。
        """
        start = len(seq) - 1
        end = start + n
        first_blk = start // self.block_size
        last_blk = (end - 1) // self.block_size
        have_last = len(seq.block_table) - 1
        need = max(0, last_blk - have_last)
        for i in range(first_blk, min(last_blk, have_last) + 1):
            if self.blocks[seq.block_table[i]].ref_count > 1:
                need += 1
        if self.rolling and need > 0:
            # 环驱逐可抵新块：表头在写完后窗口外的块数（纯计算，不改状态）
            n_evict = 0
            front = seq.kv_j0
            while (n_evict < len(seq.block_table)
                   and (front + 1) * self.block_size
                   <= len(seq) + n - self._win - self._slack):
                n_evict += 1
                front += 1
            need = max(0, need - n_evict)
        return len(self.free_block_ids) >= need

    def may_append_spec(self, seq: Sequence, n: int):
        """为写span [len-1, len-1+n) 分配可能需要的额外块。"""
        start = len(seq) - 1
        end = start + n
        last_blk = (end - 1) // self.block_size
        if self.rolling:
            n_end = len(seq) + n
            while len(seq.block_table) <= last_blk:
                # 追加前驱逐窗口外前块（释放块回池，再取新块）
                while self._front_dead(seq, n_end):
                    self._evict_front(seq)
                seq.block_table.append(self._allocate_block())
            return
        while len(seq.block_table) <= last_blk:
            seq.block_table.append(self._allocate_block())

    def cow_block(self, seq: Sequence, write_start: int) -> tuple[int, int] | None:
        """COW：写起点 write_start 所在块若被共享（ref_count>1），复制一块并换表。

        返回 (old_block_id, new_block_id)；块未被共享或无需复制时返回 None。
        新块的 KV 内容由 ModelRunner 在 GPU 上从旧块复制（本方法只做CPU记账）。
        """
        block_idx = write_start // self.block_size
        if block_idx >= len(seq.block_table):
            return None
        block_id = seq.block_table[block_idx]
        block = self.blocks[block_id]
        if block.ref_count <= 1:
            return None
        new_id = self._allocate_block()
        # 新块继承旧块内容元数据（KV由GPU复制），但hash保持-1：
        # 不作为缓存条目发布，等hash_blocks随内容增长重新发布
        self.blocks[new_id].token_ids = list(block.token_ids)
        block.ref_count -= 1
        seq.block_table[block_idx] = new_id
        return (block_id, new_id)

    def may_append(self, seq: Sequence):
        # 如果需要追加一个新块，则调用_allocate_block()分配一个新块，并将其加入seq.block_table中
        if len(seq) % self.block_size == 1:
            if self.rolling:
                # 滚动缓冲：先释放已死前块（窗口外），再取新块
                n_end = len(seq) + 1
                while self._front_dead(seq, n_end):
                    self._evict_front(seq)
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, is_prefill: bool = True, start: int | None = None, end: int | None = None):
        # 滚动缓冲：整模型不发布/不消费前缀缓存（窗口内容过期、表头会整块释放；
        # decode 追加的哈希链起点在已驱逐块上也无法维护）→ 直接跳过。
        if self.rolling:
            return
        # 本次写入覆盖的范围：
        #   prefill: [num_cached_tokens, num_cached_tokens + num_scheduled_tokens)
        #   decode:  [num_tokens - 1, num_tokens)
        #   spec:    [num_tokens - n_acc - 1, num_tokens - 1)（追加后调用；只含已接受token，
        #             被拒草稿的槽位不哈希——见Scheduler.postprocess_spec）
        # 注意decode不能用num_cached_tokens当起点：全命中序列（缓存了全部prompt token，
        # num_cached_tokens == num_tokens）第一个decode步写入的是最后一个prompt token
        # （位置num_tokens-1），按num_cached_tokens算会越界（块对齐长度时尤其如此）。
        if start is None or end is None:
            if is_prefill:
                start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            else:
                start, end = seq.num_tokens - 1, seq.num_tokens
        start_blk = start // self.block_size
        end_blk = (end + self.block_size - 1) // self.block_size
        if start_blk == end_blk: return
        h = self.blocks[seq.block_table[start_blk - 1]].hash if start_blk > 0 else -1
        for i in range(start_blk, end_blk):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            # 块内容变化（如部分块追加token）时，删除旧哈希条目，避免脏数据。
            # 必须加守卫：两个内容相同的块（如COW副本）会共享同一哈希，dict条目可能
            # 已被后写入者覆盖指向别的块——只有条目仍指向本块时才删除，否则误删他人条目
            if block.hash != -1 and block.hash != h:
                if self.hash_to_block_id.get(block.hash) == block.block_id:
                    del self.hash_to_block_id[block.hash]
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
