import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 基准计时数据：已结束请求的per-request时间戳快照 + 逐step聚合统计
        self._req_metrics: list[dict] = []
        self._step_stats: dict[str, float | int] = {}
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # isinstance(a, b)：检查a是不是b类型的对象
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.t_submitted = perf_counter()  # 记录请求入队时间（基准计时）
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 如果是prefill阶段就返回正数，意味着当前step要处理多少token
        # 如果是decode阶段就返回负数，意味着当前step要处理多少个序列，或当前step新生成的token数
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 运行模型，返回采样出的token
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 调度器完成后处理，如追加token、更新缓存、判断是否结束等
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 快照已结束请求的计时信息（基准测试使用；driver侧数据完整）
        for seq in seqs:
            if seq.is_finished:
                self._req_metrics.append({
                    "seq_id": seq.seq_id,
                    "prompt_tokens": seq.num_prompt_tokens,
                    "completion_tokens": len(seq.completion_token_ids),
                    "t_submitted": seq.t_submitted,
                    "t_first_token": seq.t_first_token,
                    "t_completed": seq.t_completed,
                })
        # 收集已经完成的请求
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True, # use_tqdm:是否显示进度条
    ) -> list[str]:
        # 创建一个进度条，总共有len(prompts)个任务，进度条前缀显示"Generating",
        # dynamic_ncols=True：让进度条自适应终端宽度，disabel：是否禁用进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        # 重置基准计时统计
        self._req_metrics = []
        self._step_stats = dict(prefill_steps=0, decode_steps=0, prefill_tokens=0, decode_tokens=0,
                                prefill_time=0.0, decode_time=0.0)
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            dt = perf_counter() - t
            # 累计逐step统计（基准测试使用）
            if num_tokens > 0:
                self._step_stats["prefill_steps"] += 1
                self._step_stats["prefill_tokens"] += num_tokens
                self._step_stats["prefill_time"] += dt
                prefill_throughput = num_tokens / dt
            else:
                self._step_stats["decode_steps"] += 1
                self._step_stats["decode_tokens"] += -num_tokens
                self._step_stats["decode_time"] += dt
                decode_throughput = -num_tokens / dt
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs

    def collect_metrics(self) -> dict:
        """返回最近一次generate()的基准计时原始数据（供benchmarks/使用）。

        - per_request: 每个已结束请求的 {seq_id, prompt_tokens, completion_tokens,
          t_submitted, t_first_token, t_completed}，时间为秒（perf_counter基准）；
          t_first_token/t_completed 为None表示请求未生成token/未完成。
        - step_stats: 逐step聚合 {prefill_steps, decode_steps, prefill_tokens,
          decode_tokens, prefill_time, decode_time}。
        - num_preemptions: 本次generate中的KV cache抢占次数。
        """
        return {"per_request": list(self._req_metrics), "step_stats": dict(self._step_stats),
                "num_preemptions": self.scheduler.num_preemptions}
