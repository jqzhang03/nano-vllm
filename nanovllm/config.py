import os
from math import isfinite
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
    execution_mode: str = "auto" # 执行模式：auto（单卡mixed，多卡可自动PD）| mixed | pd
    pd_separation: bool | None = None # 兼容旧接口：True=PD、False=mixed；建议使用 execution_mode
    prefill_device: int = 0 # PD分离时运行Prefill的CUDA设备序号
    decode_device: int = 1 # PD分离时运行Decode的CUDA设备序号
    enforce_eager: bool = False # 允许使用框架自己的推理策略。
    # True：不使用图优化，优点(兼容性更好、调试方便、某些环境下稳定)，缺点(降低推理速度)
    hf_config: AutoConfig | None = None # hugging face模型配置对象
    eos: int = -1 # EOS的token id
    kvcache_block_size: int = 256 # 在PagedAttention中，一个KV缓存块(页)的大小，在vllm生产环境下一般是16，必须保持为16的倍数
    num_kvcache_blocks: int = -1 # KV Cache块的数量，-1表示GPU根据显存大小、模型大小、block size等自动计算
    kv_cache_dtype: str = "auto" # KV缓存数据类型："auto"（模型dtype，默认）或 "fp8_e4m3"（FP8 E4M3量化，容量翻倍，decode用自研Triton内核）
    kv_calibration_path: str = "" # FP8 KV校准token IDs JSON；为空时使用内置随机token校准
    kv_fp8_scale_margin: float = 1.1 # 校准最大值对应E4M3量化上限的安全因子；<1会主动裁剪
    kv_swap: bool = True # KV swap 抢占：KV块不足时把序列的KV拷到CPU内存并释放GPU块，恢复时直接换回（bit-exact，免重新prefill）。支持TP=1的auto/fp8 KV；TP>1仍回退recompute
    kv_swap_space_gb: float = 2.0 # KV swap 的 CPU 缓冲空间上限（GB，vLLM swap_space 同款）。换出缓冲累计超限时回落 recompute 抢占，防止 CPU RAM 耗尽（本机 WSL 内存有限，0.6B 单 KV 块 28MB）
    latency_aware_scheduling: bool = True # 混合批次为等待prefill预留预算，并在候选窗口内优先预计prefill较短的请求
    cache_affinity_admission: bool = True # Top-W准入：优先复用更多前缀KV的请求
    prefix_feature_cache: bool = True # 缓存generation标记的Top-W前缀特征；False用于反复解析消融
    admission_window: int = 16 # latency/cache admission候选窗口 W
    aging_fairness: bool = True # 等待超过阈值后优先最老请求，防止长请求饿死
    aging_timeout_ms: float = 2000.0 # aging提升阈值
    prefill_reserve_tokens: int = 256 # 混合调度至少为一个等待prefill保留的token预算
    slo_aware_scheduling: bool = True # 按请求TTFT目标、队列压力和实测prefill速度调整配额
    default_ttft_slo_ms: float | None = 500.0 # 请求未单独设置时采用的TTFT目标；None表示不设默认目标
    tpot_aware_scheduling: bool = True # 按活动请求TPOT slack动态降低prefill配额、优先decode
    default_tpot_slo_ms: float | None = None # 请求未单独设置时使用的TPOT目标；None表示不设默认目标
    tpot_decode_ms_fallback: float = 20.0 # 尚无该请求decode样本时用于TPOT slack估算的回退值
    max_prefill_chunk_tokens: int = 4096 # 自适应调度单步prefill预算上限
    queue_depth_for_full_prefill: int = 16 # 等待队列达到该长度时进入最高队列压力
    multi_step_decode: bool = True # 纯decode时连续复用调度批次；prefill/spec/mixed路径仍单步
    max_decode_steps: int = 4 # 每个decode调度窗口最多执行的模型前向轮数（含首轮）
    recompute_aware_preemption: bool = True # 按估算swap往返成本与cache-aware recompute成本选择抢占方式
    preempt_prefill_tokens_per_second: float = 10000.0 # 尚无在线样本时的recompute估算回退值
    preempt_kv_transfer_gbps: float = 12.0 # 尚无在线样本时的KV swap带宽估算回退值
    rolling_cache: bool = False # SWA 滚动缓冲（阶段 2b）：解码期每序列 KV 只保留窗口内容（块数 ≤ 窗口/块大小 + 2），旧块到期自动释放——长生成序列的 KV 内存有界。支持：mistral（全层统一窗口，bf16/fp8 KV，可加 ngram 投机）；gemma2（**交替 local/global**，阶段 2b 扩展 split 模式：local 层走环池、global 层走独立 full 池普通分页永不驱逐，仅 bf16 KV、无投机、eager decode）。滚动模型不参与前缀缓存发布/消费（窗口内容过期，重复 prompt 有代价，见 block_manager.py 头注）；环语义需自研 paged 内核（flash-attn 无法表达环表位置偏移，vLLM 传统实现也只掩码不滚动）
    num_ring_kvcache_blocks: int = 0  # split 模式：环池块数（runner 分配后写回，scheduler 建 BM 用）
    num_full_kvcache_blocks: int = 0  # split 模式：full 池块数（同上）
    quantization: str = "none" # 权重量化："none" | "w8a8"（per-channel int8权重+per-token int8激活，Triton int8 GEMM）| "int4"（per-group int4权重，Triton反量化GEMM）| "awq"（int4 + AWQ激活感知缩放）| "sparse24"（2:4结构化剪枝+cuSPARSELt半结构化matmul）| "fp8"（e4m3全量化：per-column权重+per-token激活；decode走Triton内核、prefill走硬件FP8 MMA _scaled_mm）
    awq_scales_path: str = "" # AWQ激活感知缩放文件（.pt，benchmarks/awq_calibrate.py真实文本校准产出）；为空时用随机token内联校准
    quantize_lm_head: bool = False # 是否量化LM head（默认不量化——与w8a8一致：logits由lm_head点积直接决定，量化它精度损失最大，见INTERVIEW.md §10.3.6）
    int4_dense_path: bool = True # int4双路径模式（默认开）：大M prefill/decode 与小N层走 w_deq 稠密反量化（cuBLAS，收掉大M亏损与TTFT回归），小M大N层走int4内核；代价是显存 1.73GB（比fp16的1.50还大）。False=纯int4（0.85GB，大batch慢），见INTERVIEW.md §10.3.6。streaming 模式下自动强制 False（w_deq 全尺寸副本与按层加载目的冲突；MLA kv_b 例外保留反量化副本）
    int4_group_size: int = 128 # int4 量化组大小（K 维，须整除各线性层 K）。DeepSeek-V2-Lite 的 dense 中间维 10944 = 64×171 不能被 128 整除 → 该模型需 64（精度代价小，官方社区 int4 同样受此约束）
    streaming_load: bool = False # 按层流式加载+即时量化（16GB 卡跑 7B+ 的前提，见 INTERVIEW.md §1.7）：模型在 meta 设备构造（0显存）→ loader 逐 decoder layer 物化→加载→立即量化→释放 fp16。显式开启；或当估算 fp16 权重超过空闲显存 45% 且启用了权重量化时自动开启（7B+ 必触发）。限制：int4 强制纯 int4（无 w_deq，MLA kv_b 例外保留反量化副本 ~1%）；w8a8 无 SmoothQuant 校准（需全模型前向）；awq 仅支持预生成 awq_scales_path（内联校准需全 fp16 模型）
    speculative: str = "none" # 投机解码："none" | "ngram"（n-gram/prompt-lookup草稿，无模型零显存，见INTERVIEW.md §10.3.5）| "medusa"（Medusa多头，需medusa_path）| "eagle"（EAGLE-1草稿层：无RoPE transformer层 + 共享LM head 自回归草稿，需eagle_path）
    ngram_window: int = 4 # n-gram窗口上限（vLLM --ngram-prompt-lookup-max 默认同款）
    ngram_min_window: int = 1 # n-gram窗口下限（先长后短回退，vLLM --ngram-prompt-lookup-min 默认同款）
    max_draft_len: int = 4 # 每步最大草稿数γ（vLLM --num-speculative-tokens 常用值）
    medusa_path: str = "" # Medusa头权重文件（.pt，benchmarks/medusa_train.py训练产出）；speculative="medusa"时必须
    eagle_path: str = "" # EAGLE草稿层权重文件（.pt，benchmarks/eagle_train.py训练产出）；speculative="eagle"时必须
    medusa_hidden: int = 256 # Medusa头隐藏维（输出层256×vocab是主要参数，控制总规模）
    prefix_cache_max_free_blocks: int = 0 # 空闲前缀缓存块上限；0表示不设上限，分配压力下按LRU回收
    mixed_cudagraph: bool = True # 普通混合Prefill/Decode批次使用按形状惰性捕获的CUDA Graph
    mixed_cudagraph_max_graphs: int = 4 # 混合图形状缓存上限，超限淘汰最久未使用图
    mixed_cudagraph_max_tokens: int = 4096 # 超过该批次token数时走eager，限制图捕获峰值显存

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_swap_space_gb >= 0, "kv_swap_space_gb 必须非负"
        assert self.prefix_cache_max_free_blocks >= 0, \
            "prefix_cache_max_free_blocks 必须非负"
        assert self.mixed_cudagraph_max_graphs >= 1, \
            "mixed_cudagraph_max_graphs 必须至少为1"
        assert self.mixed_cudagraph_max_tokens >= 1, \
            "mixed_cudagraph_max_tokens 必须至少为1"
        assert isfinite(self.kv_fp8_scale_margin) and self.kv_fp8_scale_margin > 0, \
            "kv_fp8_scale_margin 必须是有限正数"
        if self.kv_calibration_path:
            assert self.kv_cache_dtype == "fp8_e4m3", \
                "kv_calibration_path 仅适用于 kv_cache_dtype='fp8_e4m3'"
            assert os.path.isfile(self.kv_calibration_path), \
                f"FP8 KV校准文件不存在: {self.kv_calibration_path}"
        assert self.admission_window >= 1, "admission_window 必须至少为1"
        assert self.aging_timeout_ms >= 0, "aging_timeout_ms 必须非负"
        assert self.prefill_reserve_tokens >= 1, "prefill_reserve_tokens 必须至少为1"
        assert self.default_ttft_slo_ms is None or (isfinite(self.default_ttft_slo_ms)
                                                    and self.default_ttft_slo_ms > 0), \
            "default_ttft_slo_ms 必须为正数或 None"
        assert self.default_tpot_slo_ms is None or (isfinite(self.default_tpot_slo_ms)
                                                    and self.default_tpot_slo_ms > 0), \
            "default_tpot_slo_ms 必须为正数或 None"
        assert isfinite(self.tpot_decode_ms_fallback) and self.tpot_decode_ms_fallback > 0, \
            "tpot_decode_ms_fallback 必须为有限正数"
        assert 1 <= self.max_decode_steps <= 16, \
            "max_decode_steps 必须在1到16之间"
        assert self.max_prefill_chunk_tokens >= 1, \
            "max_prefill_chunk_tokens 必须至少为1"
        assert self.queue_depth_for_full_prefill >= 1, \
            "queue_depth_for_full_prefill 必须至少为1"
        assert self.preempt_prefill_tokens_per_second > 0, \
            "preempt_prefill_tokens_per_second 必须为正数"
        assert self.preempt_kv_transfer_gbps > 0, "preempt_kv_transfer_gbps 必须为正数"
        assert self.execution_mode in ("auto", "mixed", "pd"), \
            f"unknown execution_mode: {self.execution_mode!r}"
        if self.pd_separation is not None:
            legacy_mode = "pd" if self.pd_separation else "mixed"
            if self.execution_mode == "auto":
                self.execution_mode = legacy_mode
            else:
                assert self.execution_mode == legacy_mode, \
                    "pd_separation 与 execution_mode 指定了不同的执行模式"
        if self.execution_mode == "pd":
            assert self.tensor_parallel_size == 1, \
                "当前 PD 分离版本每个角色使用一张 GPU，tensor_parallel_size 必须为 1"
            assert self.prefill_device >= 0 and self.decode_device >= 0, \
                "prefill_device/decode_device 必须是非负 CUDA 设备序号"
            assert self.prefill_device != self.decode_device, \
                "PD 分离需要为 Prefill 和 Decode 指定不同 GPU"
            assert self.speculative == "none", \
                "当前 PD 分离版本暂不支持投机解码"
            assert not self.rolling_cache, \
                "当前 PD 分离版本暂不支持 rolling_cache"
            assert self.kv_cache_dtype == "auto", \
                "当前 PD KV 传输暂要求 kv_cache_dtype='auto'"
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

    def pd_mode_incompatibility(self) -> str | None:
        """Return why PD mode cannot run with this configuration, if applicable."""
        if self.tensor_parallel_size != 1:
            return "PD separation requires tensor_parallel_size=1"
        if self.speculative != "none":
            return "PD separation does not support speculative decoding"
        if self.rolling_cache:
            return "PD separation does not support rolling_cache"
        if self.kv_cache_dtype != "auto":
            return "PD KV transfer currently requires kv_cache_dtype='auto'"
        return None
