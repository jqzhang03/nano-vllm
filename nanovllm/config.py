import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str # 模型所在目录
    max_num_batched_tokens: int = 16384 # 一次批处理的最大token数
    max_num_seqs: int = 512 # 最多同时处理序列数
    max_model_len: int = 4096 # 最大上下文长度
    gpu_memory_utilization: float = 0.9 # GPU内存利用率
    tensor_parallel_size: int = 1 # 张量并行使用的GPU数
    enforce_eager: bool = False # 允许使用框架自己的推理策略。
    # True：不使用图优化，优点(兼容性更好、调试方便、某些环境下稳定)，缺点(降低推理速度)
    hf_config: AutoConfig | None = None # hugging face模型配置对象
    eos: int = -1 # EOS的token id
    kvcache_block_size: int = 256 # 在PagedAttention中，一个KV缓存块(页)的大小，在vllm生产环境下一般是16，必须保持为16的倍数
    num_kvcache_blocks: int = -1 # KV Cache块的数量，-1表示GPU根据显存大小、模型大小、block size等自动计算

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
