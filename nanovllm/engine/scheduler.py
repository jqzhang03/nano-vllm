from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.ngram import find_ngram_draft


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0  # 抢占次数统计（基准测试使用）
        self.cow_pairs: list[tuple[int, int]] = []  # 本轮调度产生的COW复制对 (old_block_id, new_block_id)
        # ---- 投机解码（n-gram / prompt-lookup） ----
        self.spec_decode = config.speculative == "ngram"
        self.ngram_window = config.ngram_window
        self.ngram_min_window = config.ngram_min_window
        self.max_draft_len = config.max_draft_len

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], str]:
        """返回 (被调度序列, kind)；kind ∈ {"prefill", "decode", "mixed", "spec"}。

        waiting与running都非空时返回混合批次（decode行在后，prefill行在前）：
        早完成prefill的请求立即开始decode，不用等全部prefill跑完
        （vLLM V1同款策略；此前"先prefill后decode"会让早完成者空等数秒，
        见BENCHMARKS.md §5.3）。decode与prefill共享max_num_batched_tokens预算。

        投机解码（spec_decode）时：先给所有running序列算n-gram草稿；只要有任一
        草稿非空，running行全部变为verify行（γ=0的行退化为1-token varlen行），
        kind = "spec"（无waiting）或 mixed（有waiting，prefill行在前）。全部无
        草稿时回落纯decode（CUDA graph路径，不损失）。
        """
        self.cow_pairs = []
        if self.spec_decode:
            for seq in self.running:
                self._compute_draft(seq)
            if self.waiting and self.running:
                return self._schedule_mixed()
            if self.waiting:
                return self._schedule_prefill()
            if any(seq.draft_tokens for seq in self.running):
                return self._schedule_spec()
            for seq in self.running:
                seq.draft_tokens = None
            return self._schedule_decode()
        if self.waiting and self.running:
            return self._schedule_mixed()
        if self.waiting:
            return self._schedule_prefill()
        return self._schedule_decode()

    def _compute_draft(self, seq: Sequence):
        """给一个running序列计算本步n-gram草稿（CPU，无GPU成本）。

        上限 = min(最大草稿数, 剩余输出预算-1)：每步至少产出1个token（bonus/
        拒绝样本），所以草稿数最多 = remaining-1，保证追加后不超max_tokens。
        """
        remaining = seq.max_tokens - seq.num_completion_tokens - 1
        max_len = min(self.max_draft_len, remaining)
        if max_len <= 0:
            seq.draft_tokens = []
            return
        seq.draft_tokens = find_ngram_draft(seq.token_ids, self.ngram_window,
                                            self.ngram_min_window, max_len, self.eos)

    def _schedule_mixed(self) -> tuple[list[Sequence], str]:
        """混合批次：先安排decode（复用can_append/抢占逻辑），再用剩余预算做prefill。"""
        if self.spec_decode:
            return self._schedule_mixed_spec()
        # 1) decode部分（batch行序在后）
        decode_seqs = []
        while self.running and len(decode_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                pair = self.block_manager.cow_block(seq, seq.num_tokens - 1)
                if pair is not None:
                    self.cow_pairs.append(pair)
                decode_seqs.append(seq)
        self.running.extendleft(reversed(decode_seqs))

        # 2) prefill部分（batch行序在前），共享token预算
        prefill_seqs = []
        num_batched_tokens = len(decode_seqs)  # decode每序列1 token计入预算
        while self.waiting and len(prefill_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if num_tokens != 0:
                prefill_seqs.append(seq)

        if prefill_seqs and decode_seqs:
            return prefill_seqs + decode_seqs, "mixed"
        if prefill_seqs:
            return prefill_seqs, "prefill"
        assert decode_seqs
        return decode_seqs, "decode"

    def _spec_rows(self, max_rows: int, budget: int) -> list[Sequence]:
        """把running序列编排为verify行（草稿已由_compute_draft算好）。

        - 每行query长度 n = γ+1（含末token + 草稿），共享max_num_batched_tokens预算：
          预算不足时截断后续行的草稿（截断总是安全的，验收只验证剩余部分）；
        - can_append_spec 检查写span（可能跨块）所需的新块+COW副本，不足则抢占；
        - 写span内每个被共享的块都COW（含跨块时第二个块）。
        """
        rows = []
        used = 0
        while self.running and len(rows) < max_rows:
            seq = self.running.popleft()
            n = len(seq.draft_tokens) + 1
            avail = max(1, budget - used)
            if n > avail:
                seq.draft_tokens = seq.draft_tokens[:avail - 1]
                n = avail
            while not self.block_manager.can_append_spec(seq, n):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = n
                seq.is_prefill = False
                self.block_manager.may_append_spec(seq, n)
                start = len(seq) - 1
                first_blk = start // self.block_size
                last_blk = (start + n - 1) // self.block_size
                for b in range(first_blk, last_blk + 1):
                    pair = self.block_manager.cow_block(seq, b * self.block_size)
                    if pair is not None:
                        self.cow_pairs.append(pair)
                rows.append(seq)
                used += n
        self.running.extendleft(reversed(rows))
        return rows

    def _schedule_spec(self) -> tuple[list[Sequence], str]:
        """纯verify步（无waiting）：所有running行做一次并行验证。"""
        rows = self._spec_rows(self.max_num_seqs, self.max_num_batched_tokens)
        assert rows
        return rows, "spec"

    def _schedule_mixed_spec(self) -> tuple[list[Sequence], str]:
        """投机混合步：verify行（后）+ prefill行（前），共享token预算。"""
        spec_rows = self._spec_rows(self.max_num_seqs, self.max_num_batched_tokens)
        prefill_seqs = []
        num_batched_tokens = sum(seq.num_scheduled_tokens for seq in spec_rows)
        while self.waiting and len(prefill_seqs) + len(spec_rows) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and prefill_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            if num_tokens != 0:
                prefill_seqs.append(seq)

        if prefill_seqs and spec_rows:
            return prefill_seqs + spec_rows, "mixed"
        if prefill_seqs:
            return prefill_seqs, "prefill"
        assert spec_rows
        return spec_rows, "spec"

    def _schedule_prefill(self) -> tuple[list[Sequence], str]:
        # 需要被调度的序列列表
        scheduled_seqs = []
        # 在prefill阶段需要处理的token数量
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 判断当前序列是否占用KV Cache block块
            if not seq.block_table:
                # 计算当前序列匹配的共享前缀的KV Cache块个数，并不真正分配KV Cache块
                num_cached_blocks = self.block_manager.can_allocate(seq)
                # 返回-1则无法分配相应数量的KV Cache块
                if num_cached_blocks == -1:
                    break
                # 分配KV Cache块；allocate会按实际缓存长度设置seq.num_cached_tokens
                # （部分块按真实token数记账，而非num_cached_blocks*block_size）
                self.block_manager.allocate(seq, num_cached_blocks)
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 当可处理的token数小于需要处理的token数时，只有当前序列是第一个被调度的序列时才允许将其进行分块处理
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 部分块共享后，若本次prefill的写起点落在被共享的缓存块内（如共享了44-token的尾块
            # 且要继续写入），先把该块复制一份，避免污染其他共享者
            if seq.num_scheduled_tokens > 0:
                pair = self.block_manager.cow_block(seq, seq.num_cached_tokens)
                if pair is not None:
                    self.cow_pairs.append(pair)
            # 如果缓存的前缀token数量+当前调度的token数量等于总token数量，说明当前序列已经完成了prefill阶段，
            # 将其状态修改为RUNNING，加入decode序列当中
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 当前序列prefill如果未完成，再加入调度序列中
            if num_tokens != 0:
                scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, "prefill"
        # 本步无prefill可调度（如全部full-hit）→ 回落decode（与旧行为一致）
        return self._schedule_decode()

    def _schedule_decode(self) -> tuple[list[Sequence], str]:
        # decode
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 检查当前分配的KV Cache块能否追加上一轮生成的、但还没写入KV Cache块的token
            # KV Cache写入逻辑是：如果当前序列的token数量%block_size==1，说明需要申请一个新的KV Cache块来存储上一轮生成的token
            # 否则直接在最后一个KV Cache块中追加即可
            while not self.block_manager.can_append(seq):
                # 如果运行队列中还有其他序列，则中断当前序列，将其放回至等待队列中，释放其占用的资源
                if self.running:
                    self.preempt(self.running.pop())
                else: # 否则，运行队列中没有其他队列，只能将自己释放，自己回退到等待队列
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                # decode写入末块前，若末块是共享的部分块，先复制一块（COW）
                pair = self.block_manager.cow_block(seq, seq.num_tokens - 1)
                if pair is not None:
                    self.cow_pairs.append(pair)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, "decode"

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.draft_tokens = None  # 回waiting的序列下次以prefill行重新调度，草稿作废
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def _maybe_finish(self, seq: Sequence, token_id: int):
        # 判断当前序列是否满足结束条件
        if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
            seq.status = SequenceStatus.FINISHED
            seq.t_completed = perf_counter()
            self.block_manager.deallocate(seq)
            self.running.remove(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        # 混合批次里prefill与decode序列并存：按各序列的is_prefill（调度器设置）分支
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq, seq.is_prefill)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 记录首个生成token的时间（用于TTFT统计），仅第一次追加时触发
            if seq.t_first_token is None and seq.num_completion_tokens == 0:
                seq.t_first_token = perf_counter()
            seq.append_token(token_id)
            self._maybe_finish(seq, token_id)

    def postprocess_spec(self, seqs: list[Sequence], token_lists: list[list[int]]):
        """投机步后处理：verify行按已接受token数更新缓存与哈希；prefill行同原逻辑。

        KV提交语义：verify写span [len-1, len-1+num_scheduled) 含被拒草稿的槽位，
        不回滚（下一步覆盖即可），只截断逻辑长度；前缀缓存哈希只发布到接受长度
        （[num_tokens-n_acc-1, num_tokens-1)，追加后调用）——被拒token永不进哈希。
        """
        for seq, tokens in zip(seqs, token_lists):
            if seq.draft_tokens is not None:
                n_acc = len(tokens)
                if seq.t_first_token is None and seq.num_completion_tokens == 0:
                    seq.t_first_token = perf_counter()
                seq.append_tokens(tokens)
                self.block_manager.hash_blocks(seq, False,
                                               start=seq.num_tokens - n_acc - 1,
                                               end=seq.num_tokens - 1)
                seq.num_cached_tokens = seq.num_tokens
                seq.num_scheduled_tokens = 0
                seq.draft_tokens = None
                self._maybe_finish(seq, tokens[-1])
            else:
                self.block_manager.hash_blocks(seq, seq.is_prefill)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.is_prefill and seq.num_cached_tokens < seq.num_tokens:
                    continue
                if seq.t_first_token is None and seq.num_completion_tokens == 0:
                    seq.t_first_token = perf_counter()
                seq.append_token(tokens[0])
                self._maybe_finish(seq, tokens[0])
