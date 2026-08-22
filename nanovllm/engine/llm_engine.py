import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
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
        self.config = config
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
        # 幂等：显式调用与atexit可能都触发，且同一进程可能先后创建多个引擎（如精度对比）
        if getattr(self, "_exited", False):
            return
        self._exited = True
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

    def _verify(self, seqs: list[Sequence], token_ids: list[int]):
        """投机验收：把逐行样本与草稿比对，返回每序列已接受token列表。

        logits行序 = 批次行序：普通行每seq 1行（LM head已归约）；verify行每seq
        γ+1行（第i行预测位置len-1+i → 样本samples[i]验证草稿drafts[i]；
        最后一行是全接受时的bonus）。接受语义见 nanovllm/engine/ngram.verify_drafts。

        返回 (token_lists, n_decode, n_draft, n_draft_acc, n_verify, n_acc_list)：
        n_decode = 本步产出token总数（= Σ接受数）；n_draft = 草稿总数；
        n_draft_acc = 被接受草稿数（α = n_draft_acc / n_draft）；
        n_verify = verify forward处理的token数（Σ γ_i+1，含末token重算）；
        n_acc_list = 每seq的接受数（与seqs对齐，Medusa draft选行用）。
        """
        from nanovllm.engine.ngram import verify_drafts
        token_lists = []
        idx = 0
        n_decode = n_draft = n_draft_acc = n_verify = 0
        n_acc_list = []
        for seq in seqs:
            if seq.draft_tokens is None:
                token_lists.append([token_ids[idx]])
                idx += 1
                n_decode += 1
                n_acc_list.append(0)  # 非verify行（prefill行）；_medusa_drafts用它识别
                continue
            drafts = seq.draft_tokens
            n = len(drafts) + 1
            samples = token_ids[idx:idx + n]
            idx += n
            accepted, n_acc = verify_drafts(drafts, samples)
            token_lists.append(accepted)
            n_decode += n_acc
            n_draft += len(drafts)
            n_draft_acc += n_acc - 1
            n_verify += n
            n_acc_list.append(n_acc)
        return token_lists, n_decode, n_draft, n_draft_acc, n_verify, n_acc_list

    def _medusa_drafts(self, seqs: list[Sequence], hidden: torch.Tensor,
                       n_acc_list: list[int], n_rows_list: list[int]):
        """从verify步的hidden批量计算下一轮Medusa draft（语义见 layers/medusa.py）。

        draft输入行 = 验收后新t_last的hidden（当前verify输入的第min(n_acc,γ_i)行：
        非全接受时t_last是第n_acc个输入token；全接受时bonus无hidden，用第γ_i行）+
        head偏移（全接受 shift=1）。EOS/剩余预算截断与n-gram一致。
        n_rows_list 必须在postprocess前捕获（postprocess会清零num_scheduled_tokens）。
        """
        heads = self.model_runner.medusa_heads
        gamma = self.config.max_draft_len
        eos = self.config.eos
        # mixed步里prefill行在hidden前面，spec组从n_prefill_tokens起
        h_offset = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
        groups = {0: [], 1: []}  # shift → [(seq, hidden行号)]
        row = h_offset
        for seq, n_acc, n_rows in zip(seqs, n_acc_list, n_rows_list):
            if n_acc == 0:
                continue  # 非verify行（prefill行）
            gamma_i = n_rows - 1  # 本行实际γ（可能<全局γ）
            if n_acc <= gamma_i:
                # 非全接受：t_last（位置len+n_acc-1）是当前verify输入的第n_acc行
                idx = row + n_acc
                shift = 0
            else:
                # 全接受：t_last是bonus采样产物、无hidden → 用第γ_i行 + head偏移1
                idx = row + gamma_i
                shift = 1
            groups[shift].append((seq, idx))
            row += n_rows
        for shift, items in groups.items():
            if not items:
                continue
            h = torch.stack([hidden[i] for _, i in items])  # [n, hidden]（fp16）
            toks = [hd(h).argmax(dim=-1).tolist() for hd in heads.heads[shift:shift + gamma]]
            for i, (seq, _) in enumerate(items):
                cap = min(gamma, seq.max_tokens - seq.num_completion_tokens - 1)
                drafts = []
                for k in range(cap):
                    t = toks[k][i]
                    if t == eos:
                        break
                    drafts.append(t)
                seq.draft_tokens = drafts

    def step(self):
        seqs, kind = self.scheduler.schedule()  # kind ∈ {"prefill", "decode", "mixed", "spec"}
        # 本步是否含verify行（投机）：任一行带draft_tokens（[]也算，表示γ=0的verify行）
        has_spec = any(seq.draft_tokens is not None for seq in seqs)
        # Medusa模式：spec步需要最后一层hidden（下轮草稿输入）
        return_hidden = self.config.speculative == "medusa" and kind == "spec"
        # 在运行模型之前执行COW复制：任何序列写共享部分块之前，先把旧块复制给写者。
        # 必须发生在 run() 之前，prepare_decode/prepare_prefill 才能基于换表后的新块计算slot
        for old_id, new_id in self.scheduler.cow_pairs:
            self.model_runner.call("cow_block", old_id, new_id)
        # 运行模型，返回采样出的token（精度检查模式下同时返回本步logits）
        result = self.model_runner.call("run", seqs, kind, self._collect_logits, return_hidden)
        hidden = None
        if self._collect_logits:
            if return_hidden:
                token_ids, logits, hidden = result
            else:
                token_ids, logits = result
            self.collected_logits.append((kind, logits))
        else:
            if return_hidden:
                token_ids, hidden = result
            else:
                token_ids = result
        if has_spec:
            # 统计必须在postprocess_spec之前（之后draft_tokens/num_scheduled_tokens被清空）
            n_spec_rows = sum(1 for seq in seqs if seq.draft_tokens is not None)
            # Medusa：postprocess前捕获每行行数（_medusa_drafts选hidden行用）
            n_rows_list = [seq.num_scheduled_tokens if seq.draft_tokens is not None else 0
                           for seq in seqs]
            (token_lists, n_decode, n_draft, n_draft_acc, n_verify,
             n_acc_list) = self._verify(seqs, token_ids)
            self.scheduler.postprocess_spec(seqs, token_lists)
            # 投机统计（与kind无关，混合步同样累计）
            self._step_stats["spec_steps"] += 1
            self._step_stats["spec_rows"] += n_spec_rows
            self._step_stats["spec_verify_tokens"] += n_verify
            self._step_stats["spec_draft_tokens"] += n_draft
            self._step_stats["spec_accepted_drafts"] += n_draft_acc
        else:
            self.scheduler.postprocess(seqs, token_ids)
        # Medusa：用本步verify的hidden生成下一轮草稿（写回seq.draft_tokens）
        if return_hidden:
            self._medusa_drafts(seqs, hidden, n_acc_list, n_rows_list)
        # 统计用token数：prefill步为正（prefill token数），decode步为负（序列数），
        # mixed步拆分返回prefill/decode各自的数量
        if kind == "prefill":
            n_prefill, n_decode = sum(seq.num_scheduled_tokens for seq in seqs), 0
        elif kind == "decode":
            n_prefill, n_decode = 0, len(seqs)
        elif kind == "spec":
            n_prefill = 0  # n_decode 已在验收中按实际接受数统计
        else:  # mixed
            n_prefill = sum(seq.num_scheduled_tokens for seq in seqs if seq.is_prefill)
            if not has_spec:
                n_decode = sum(1 for seq in seqs if not seq.is_prefill)
        # 调度器完成后处理，如追加token、更新缓存、判断是否结束等
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
        return outputs, kind, n_prefill, n_decode

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True, # use_tqdm:是否显示进度条
        collect_logits: bool = False, # 精度检查：逐step收集logits到self.collected_logits
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
                                prefill_time=0.0, decode_time=0.0,
                                spec_steps=0, spec_rows=0, spec_verify_tokens=0, spec_draft_tokens=0,
                                spec_accepted_drafts=0)
        self._collect_logits = collect_logits
        self.collected_logits = []
        while not self.is_finished():
            t = perf_counter()
            output, kind, n_prefill, n_decode = self.step()
            dt = perf_counter() - t
            # 累计逐step统计（基准测试使用）；mixed步按token比例拆分时间归属
            if kind == "prefill":
                self._step_stats["prefill_steps"] += 1
                self._step_stats["prefill_tokens"] += n_prefill
                self._step_stats["prefill_time"] += dt
                prefill_throughput = n_prefill / dt
            elif kind in ("decode", "spec"):
                self._step_stats["decode_steps"] += 1
                self._step_stats["decode_tokens"] += n_decode
                self._step_stats["decode_time"] += dt
                decode_throughput = n_decode / dt
            else:  # mixed
                self._step_stats["prefill_steps"] += 1
                self._step_stats["decode_steps"] += 1
                self._step_stats["prefill_tokens"] += n_prefill
                self._step_stats["decode_tokens"] += n_decode
                total = n_prefill + n_decode
                self._step_stats["prefill_time"] += dt * n_prefill / total
                self._step_stats["decode_time"] += dt * n_decode / total
                if n_prefill:
                    prefill_throughput = n_prefill / dt
                if n_decode:
                    decode_throughput = n_decode / dt
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
          decode_tokens, prefill_time, decode_time}；投机解码时另有
          {spec_steps, spec_verify_tokens, spec_draft_tokens, spec_accepted_drafts}
          （α = spec_accepted_drafts / spec_draft_tokens）。
        - num_preemptions: 本次generate中的KV cache抢占次数。
        """
        return {"per_request": list(self._req_metrics), "step_stats": dict(self._step_stats),
                "num_preemptions": self.scheduler.num_preemptions}
