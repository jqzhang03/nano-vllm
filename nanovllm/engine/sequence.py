from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    # auto()自动为枚举对象赋值，从1开始
    WAITING = auto() # 1
    RUNNING = auto() # 2
    FINISHED = auto() # 3


class Sequence:
    block_size = 256 # KV Cache块大小
    counter = count() # 确保每个Sequence都有唯一的ID

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING

        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids) # 当前token总数，prompt + completion
        self.num_prompt_tokens = len(token_ids) # prompt token总数

        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.draft_tokens: list[int] | None = None  # 投机解码：本步n-gram草稿（None=非verify行；[]=verify行无草稿）
        self.swapped = False  # KV swap 抢占：KV 已换出到 CPU（在 swapped 队列，恢复时换入直接decode）

        # ---- 基准计时（仅driver侧使用，不随__getstate__跨进程传输） ----
        self.t_submitted: float | None = None      # 请求加入调度队列的时间（秒）
        self.t_first_token: float | None = None    # 生成第一个completion token的时间（用于TTFT）
        self.t_completed: float | None = None      # 请求完成（FINISHED）的时间

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def append_tokens(self, token_ids: list[int]):
        """一次追加多个 token（投机验收：n_acc 个已接受 token）。"""
        assert token_ids
        self.token_ids.extend(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens += len(token_ids)

    # 跨进程同步通信
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state, self.draft_tokens)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens,
         self.block_table, last_state, self.draft_tokens) = state
        if isinstance(last_state, list): # 如果是prefill阶段传来的
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else: # 如果是decode阶段传来的
            self.token_ids = []
            self.last_token = last_state
