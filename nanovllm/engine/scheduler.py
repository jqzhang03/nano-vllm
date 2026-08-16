from collections import deque
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


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

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]: # 返回被调度的序列列表和是否是prefill阶段
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
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            # 当可处理的token数小于需要处理的token数时，只有当前序列是第一个被调度的序列时才允许将其进行分块处理
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                # 分配KV Cache块
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            # 如果缓存的前缀token数量+当前调度的token数量等于总token数量，说明当前序列已经完成了prefill阶段，
            # 将其状态修改为RUNNING，加入decode序列当中
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 无论当前序列prefill是否完成，都加入调度序列中
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
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
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # 记录首个生成token的时间（用于TTFT统计），仅第一次追加时触发
            if seq.t_first_token is None and seq.num_completion_tokens == 0:
                seq.t_first_token = perf_counter()
            seq.append_token(token_id)
            # 判断当前序列是否满足结束条件
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                seq.t_completed = perf_counter()
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
