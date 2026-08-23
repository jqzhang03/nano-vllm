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
    kv_cache_dtype: str = "auto" # KV缓存数据类型："auto"（模型dtype，默认）或 "fp8_e4m3"（FP8 E4M3量化，容量翻倍，decode用自研Triton内核）
    kv_swap: bool = True # KV swap 抢占：KV块不足时把序列的KV拷到CPU内存并释放GPU块，恢复时直接换回（bit-exact，免重新prefill）。仅TP=1且非fp8 KV时生效（fp8的float8_e4m3是CUDA-only，无法分配CPU缓冲）
    kv_swap_space_gb: float = 2.0 # KV swap 的 CPU 缓冲空间上限（GB，vLLM swap_space 同款）。换出缓冲累计超限时回落 recompute 抢占，防止 CPU RAM 耗尽（本机 WSL 仅 7GB，0.6B 单 KV 块 28MB）
    quantization: str = "none" # 权重量化："none" | "w8a8"（per-channel int8权重+per-token int8激活，Triton int8 GEMM）| "int4"（per-group int4权重，Triton反量化GEMM）| "awq"（int4 + AWQ激活感知缩放）| "sparse24"（2:4结构化剪枝+cuSPARSELt半结构化matmul）| "fp8"（e4m3全量化：per-column权重+per-token激活；decode走Triton内核、prefill走硬件FP8 MMA _scaled_mm）
    awq_scales_path: str = "" # AWQ激活感知缩放文件（.pt，benchmarks/awq_calibrate.py真实文本校准产出）；为空时用随机token内联校准
    quantize_lm_head: bool = False # 是否量化LM head（默认不量化——与w8a8一致：logits由lm_head点积直接决定，量化它精度损失最大，见BENCHMARKS.md §10）
    int4_dense_path: bool = True # int4双路径模式（默认开）：大M prefill/decode 与小N层走 w_deq 稠密反量化（cuBLAS，收掉大M亏损与TTFT回归），小M大N层走int4内核；代价是显存 1.73GB（比fp16的1.50还大）。False=纯int4（0.85GB，大batch慢），见BENCHMARKS.md §10。streaming 模式下自动强制 False（w_deq 全尺寸副本与按层加载目的冲突）
    streaming_load: bool = False # 按层流式加载+即时量化（16GB 卡跑 7B+ 的前提，见 LEARNING.md 阶段7）：模型在 meta 设备构造（0显存）→ loader 逐 decoder layer 物化→加载→立即量化→释放 fp16。显式开启；或当估算 fp16 权重超过空闲显存 45% 且启用了权重量化时自动开启（7B+ 必触发）。限制：int4 强制纯 int4（无 w_deq）；w8a8 无 SmoothQuant 校准（需全模型前向）；awq 仅支持预生成 awq_scales_path（内联校准需全 fp16 模型）
    speculative: str = "none" # 投机解码："none" | "ngram"（n-gram/prompt-lookup草稿，无模型零显存，见BENCHMARKS.md §9）| "medusa"（Medusa多头，需medusa_path）| "eagle"（EAGLE-1草稿层：无RoPE transformer层 + 共享LM head 自回归草稿，需eagle_path）
    ngram_window: int = 4 # n-gram窗口上限（vLLM --ngram-prompt-lookup-max 默认同款）
    ngram_min_window: int = 1 # n-gram窗口下限（先长后短回退，vLLM --ngram-prompt-lookup-min 默认同款）
    max_draft_len: int = 4 # 每步最大草稿数γ（vLLM --num-speculative-tokens 常用值）
    medusa_path: str = "" # Medusa头权重文件（.pt，benchmarks/medusa_train.py训练产出）；speculative="medusa"时必须
    eagle_path: str = "" # EAGLE草稿层权重文件（.pt，benchmarks/eagle_train.py训练产出）；speculative="eagle"时必须
    medusa_hidden: int = 256 # Medusa头隐藏维（输出层256×vocab是主要参数，控制总规模）

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.speculative in ("none", "ngram", "medusa", "eagle"), \
            f"unknown speculative: {self.speculative}"
        assert self.quantization in ("none", "w8a8", "int4", "awq", "sparse24", "fp8"), \
            f"unknown quantization: {self.quantization}"
        if self.speculative == "medusa":
            assert self.medusa_path, "speculative=medusa requires medusa_path"
        if self.speculative == "eagle":
            assert self.eagle_path, "speculative=eagle requires eagle_path"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
