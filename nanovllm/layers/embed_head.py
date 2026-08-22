import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.linear import WeightQuantMixin
from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(WeightQuantMixin, VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)
        self.int4 = False  # w_int4/w_int4_scale/awq_scale缓冲由quantize_int4创建
        self.sparse24 = False  # w_sparse由quantize_sparse24创建

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_mixed:
            if context.is_spec:
                # 投机混合步：prefill组（每seq取最后一行，索引相对组内）+ verify组
                # （每seq保留全部γ+1行，从prefill组token边界起）
                pre_last = context.cu_seqlens_q[1:context.n_prefill_rows + 1] - 1
                x = torch.cat([x[pre_last], x[context.cu_seqlens_q[context.n_prefill_rows]:]], dim=0).contiguous()
            else:
                # 混合批次：prefill组（每seq取最后一行，索引相对组内） + decode组（每seq一行，原样）
                pre_last = context.cu_seqlens_q[1:] - 1
                x = torch.cat([x[pre_last], x[context.n_prefill_tokens:]], dim=0).contiguous()
        elif context.is_spec:
            pass  # verify步：保留全部行（每seq γ+1行，供逐行采样+验收）
        elif context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        if self.sparse24:
            logits = self._sparse24_forward(x)
        elif self.int4:
            logits = self._int4_forward(x)
        else:
            logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
