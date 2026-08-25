# nano-vllm 学习路线（按依赖排序）

按"先懂主线、再懂优化、最后懂投机与量化"的顺序组织。每个功能列出：**读什么文件（精确到类/函数）**、配套脚本、文档章节。建议配合 `git log` 看每个功能的提交历史（提交信息是短摘要，能还原当时的问题与解法）。

---

## 阶段 0：全局图景（30 分钟，先读文档）

| 读什么 | 要点 |
|---|---|
| `CLAUDE.md` | 架构总览：请求生命周期、Context 单例契约、KV cache、前缀缓存/COW、调度、投机、CUDA graph、TP |
| `BENCHMARKS.md` 开头"指标定义" | TTFT/TPOT/E2E/p50/p99/SLO 的口径——后面所有数字都基于它 |
| `AGENTS.md` | 模块组织、开发约定 |
| `nanovllm/config.py` | 全部开关：quantization/speculative/kv_cache_dtype/int4_dense_path/awq_scales_path……每个字段对应一个功能 |

**目标**：能说出"一次 `LLM.generate` 从进队列到出 token 经过了哪几个大环节"。

---

## 阶段 1：核心推理主链路（最重要，2-4 小时）

**一次生成的全旅程**：`LLM.generate` → `Scheduler.schedule` → `ModelRunner.run` → 模型 forward → `Sampler` → `Scheduler.postprocess` → 循环。

| 文件（类/函数） | 学习要点 |
|---|---|
| `nanovllm/llm.py`、`nanovllm/sampling_params.py` | 入口与采样参数（禁止 greedy、温度下限） |
| `nanovllm/engine/sequence.py` | `Sequence`：token 存储、`block_table`、`num_cached_tokens`、`append_tokens`、`__getstate__/__setstate__`（TP 跨进程） |
| `nanovllm/engine/llm_engine.py` | `generate`（主循环）、`step`（一次调度+前向+postprocess）、`add_request`、`collect_metrics` |
| `nanovllm/engine/scheduler.py` | `schedule`（kind 分发）、`_schedule_prefill/_schedule_decode`、`postprocess`（append/EOS/哈希）、`preempt`（KV 不足抢占） |
| `nanovllm/engine/model_runner.py` | `run`（入口）、`prepare_prefill`（打包 varlen 张量）、`prepare_decode`（paged）、`run_model`（模型前向 + CUDA graph 选择）、`prepare_sample` |
| `nanovllm/models/qwen3.py` | 模型结构：`Qwen3Attention`（QKV 投影+QK-Norm+RoPE+Attention）、`Qwen3MLP`（gate_up+SiLU+down）、`Qwen3DecoderLayer`（残差+RMSNorm）、`Qwen3ForCausalLM`（`packed_modules_mapping` + `compute_logits`） |
| `nanovllm/layers/attention.py` | `Attention.forward`：prefill 走 `flash_attn_varlen_func`、decode 走 `flash_attn_with_kvcache`；`store_kvcache` 写缓存 |
| `nanovllm/layers/layernorm.py` | `RMSNorm`（含残差融合的 `add_rms_forward`）、`@torch.compile` |
| `nanovllm/layers/rotary_embedding.py` | `RotaryEmbedding` + `apply_rotary_emb`（fused 旋转） |
| `nanovllm/layers/activation.py` | `SiluAndMul`（门控融合） |
| `nanovllm/layers/sampler.py` | `Sampler`：Gumbel 指数噪声采样（`torch.manual_seed` 不够、要 `torch.cuda.manual_seed` 的原因） |
| `nanovllm/layers/embed_head.py` | `VocabParallelEmbedding`、`ParallelLMHead`（**按 context 分支取行**：prefill 取末行/mixed 拼接/spec 全保留） |
| `nanovllm/utils/context.py` | `Context` 单例 + `set_context/get_context/reset_context`——**理解每步张量如何从 runner 传给内核的关键契约** |
| `nanovllm/utils/loader.py` | `load_model` + `weight_loader` 约定（packed 映射：q/k/v→qkv_proj） |

**建议读法**：先看 `qwen3.py`+`layers/`（模型长什么样）→ 再看 `model_runner.py`（张量怎么打包）→ `scheduler.py`（批次怎么选）→ `llm_engine.py`（循环怎么转）→ 最后 `context.py` 把所有数据流串起来。

---

## 阶段 2：调度与内存管理（2-3 小时）

| 功能 | 读什么 | 配套脚本/文档 | 面试要点 |
|---|---|---|---|
| **混合调度**（vLLM V1 同款） | `scheduler.py` `_schedule_mixed`；`model_runner.py` `prepare_mixed`；`attention.py` 混合路由；`embed_head.py` `ParallelLMHead` 的 `is_mixed` 分支 | `benchmarks/bench.py`；BENCHMARKS §6 | 为什么比"先全 prefill 后 decode"好（死等消除、抢占下降） |
| **前缀缓存 + COW** | `block_manager.py`：`compute_hash`（链式哈希）、`can_allocate`、`allocate`、`hash_blocks`（部分块也发布哈希）、`cow_block`（写共享块前复制）、`can_append/may_append`；`scheduler.py` 的哈希删除守卫 | `tests/test_block_manager.py`；BENCHMARKS §3/§3.5/§4 | 哈希链为什么带前块哈希；部分块缓存为什么安全；COW 在 GPU 上怎么执行 |
| **分块 prefill** | `scheduler.py` `_schedule_prefill`（只允许第一个序列切块）；`model_runner.py` `prepare_prefill`（key 超 query → 缓存形状 K/V + block_tables） | BENCHMARKS §1/§2 | 前缀命中时 K 长于 Q 的 varlen 怎么表示 |
| **抢占与恢复** | `scheduler.py` `preempt`（**KV swap 分流**：decode 序列换出到 CPU `swap_out`/`swap_in` + 独立 `swapped` 队列 + `kv_swap_space_gb` 预算；prefill 序列走 recompute）；`block_manager.py` `allocate_private`/`release_blocks`；`model_runner.py` `swap_out`/`swap_in`（**index_copy_ 原位写**）；`_swap_bitexact.py`（KV 0 误差验证） | `bench.py --no-swap-kv`、`_swap_smoke.py` | swap 比 recompute 保持采样流确定（bit-exact 免重算）；本机 0.6B+WSL2 上重算更便宜（swap 慢 2.4×）；7B+ 与真实 Linux 才有价值 |

---

## 阶段 3：CUDA graph 与启动税（1-2 小时）

| 功能 | 读什么 | 配套脚本 | 面试要点 |
|---|---|---|---|
| **decode CUDA graph** | `model_runner.py` `capture_cudagraph`（批量族 [1,2,4,8]+16 步进、共享内存池）、`run_model` 的图选择与静态输入拷贝 | `bench.py`（默认非 eager） | graph 捕获要求固定形状/地址；`enforce_eager` 只关图不关 torch.compile |
| **启动税诊断**（方法学） | `benchmarks/_verify_probe.py`、`_step_timing.py` | — | 探针分层：分页 vs 连续、形状、CPU launch 计数——**先证伪假设再修** |

---

## 阶段 4：量化（建议 3-5 小时，按依赖顺序）

### 4.1 FP8 KV cache（先看，因为注意力内核最独立）
| 读什么 | 要点 |
|---|---|
| `model_runner.py` `calibrate_fp8_kv` | 随机 token 校准每层固定 scale（max/448×1.1） |
| `attention.py` `store_kvcache_kernel` | 写路径：fp32→fp8 cast **不饱和产生 NaN 位模式，必须 clamp(-448,448)**（BENCHMARKS §7 的潜伏 bug） |
| `attention.py` `paged_decode_attention_fp8_kernel`（v6） | decode 内核：直接 fp8 load + 硬件 cvt 反量化、QPAD=16 MMA、GQA 融合 |
| `attention.py` `paged_varlen_attention_fp8_kernel`（v7） | 投机 verify 的多查询扩展（见阶段 5） |
| 配套 | `_fp8_kernel_check.py`、`_kernel_bench.py`、`_kernel_v6_exp.py`、`_mask_probe.py`、`accuracy_check.py`；BENCHMARKS §7 |

### 4.2 W8A8（int8 GEMM + SmoothQuant）
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_int8_kernel`/`w8a8_gemm` | per-group(128) 权重 scale，BLOCK_K=128=组大小，int32 累加后乘组 scale 以 fp32 跨组累加 |
| `linear.py` `LinearBase.quantize_w8a8`/`_w8a8_forward` | SmoothQuant 折叠：`s = x_max^0.5 / w_col^0.5`，`W'=W·s, X'=X/s` 恒等变换 |
| `model_runner.py` `calibrate_and_quantize_w8a8` | 校准 hook 收集逐通道 amax |
| 配套 | `_w8a8_check.py`、`_w8a8_layers.py`；BENCHMARKS §8 |

### 4.3 INT4 + AWQ（当前主力）
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_int4_kernel`/`int4_gemm` | 2-dot 拆分：按 K 奇偶拆 a 与半字节、两个 dot；打包沿 K（字节低/高半字节 = k=2j/2j+1）；tile 按 M 自适应 |
| `linear.py` `WeightQuantMixin.quantize_int4`/`_int4_forward` | per-group 128 对称 int4（组 scale=amax/7、码偏移+8）；**双路径路由**：`w_deq`（bf16 反量化副本）供大 M/小 N 走 cuBLAS，`M≤128 且 N≥2048` 走 int4 内核 |
| `linear.py` 类常量 `int4_max_m/int4_min_n` | 路由阈值（来自 `_int4_tune.py` 的形态结论） |
| `model_runner.py` `quantize_int4_weights`/`quantize_awq_weights`/`_calibrate_awq_scales` | 加载后量化；AWQ 缩放文件加载或内联随机校准 |
| `benchmarks/awq_calibrate.py` | **按层 α 搜索**：`s=(mean\|X\|/w_col)^α`，目标=校准批量化输出误差；方向 `W'=W·s, X'=X/s`（论文方向，反了会塌缩） |
| `config.py` `int4_dense_path`/`quantize_lm_head`/`awq_scales_path` | 双路径开关 / lm_head 量化（默认关，Qwen3-0.6B tie 绑定自动跳过） |
| 配套 | `_int4_check.py`（含双路径一致性）、`_int4_tune.py`、`_int4_layers.py`、`_quant_ppl.py`（**端到端 ppl 是决定性指标**）、`_quant_mem.py`、`_awq_diagnose.py`、`_int4_mem_debug.py`；BENCHMARKS §10.1/§10.2 |

### 4.4 2:4 结构化稀疏
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_sparse24_kernel`/`sparse24_gemm` | 4 路拆分：a 按 K 步长 4 加载、`idx==p` 掩码重建权重块、4 个 dot；打包 `v [N,K//2] bf16` + `idx [N,K//4] uint8` |
| `linear.py` `WeightQuantMixin.quantize_sparse24` | 幅值剪枝（组内保留最大 2）+ 打包 |
| 配套 | `_sparse24_probe.py`（**cuSPARSELt/CUTLASS 在 sm_120 的结论**）、`_sparse24_check.py`、`_sparse24_tune.py`、`_sparse24_layers.py`、`_sparse24_engine_debug.py`；BENCHMARKS §10.3 |

---

## 阶段 5：投机解码（建议 3-4 小时，依赖阶段 3 的 graph 概念）

| 功能 | 读什么 | 配套脚本 | 面试要点 |
|---|---|---|---|
| **n-gram 草稿** | `engine/ngram.py` `find_ngram_draft`（窗口 4→1 回退、EOS 截断）、`verify_drafts`（点质量验收） | `tests/test_spec_decode.py` | 验收为什么严格保持分布（输出恒等于目标采样） |
| **verify 步 = varlen prefill** | `model_runner.py` `prepare_spec`/`_prepare_mixed_spec`；`scheduler.py` `_compute_draft`/`_spec_rows`/`_schedule_spec`/`_schedule_mixed_spec`/`postprocess_spec`；`llm_engine.py` `_verify` | `_spec_equiv_check.py`（三层验证） | query=[末token+草稿]、num_cached=len-1；**KV 提交语义：被拒草稿不回滚、哈希只发布到接受长度** |
| **verify CUDA graph** | `model_runner.py` `capture_spec_graph`（容量族×双 stride、零长度填充行）、`_spec_graph_hidden`、`run_model` 的 spec 重放 | `_graph_pad_probe.py`（bit-exact）、`_verify_probe.py`、`_spec_step_timing.py` | 启动税 ~10ms/步 的定位与消除；varlen 怎么用固定容量图 + 空行填充 |
| **Medusa 多头** | `layers/medusa.py` `MedusaHead/MedusaHeads`；`llm_engine.py` `_medusa_drafts`（行选择 + 全接受 shift）；`model_runner.py` medusa 加载 | `benchmarks/medusa_train.py`（自蒸馏训练）、`_medusa_debug.py`、`_medusa_integration.py`、`_medusa_eval.py` | head_k 语义（预测 t+k+1）；训练必须 exit 引擎（allocator 60× 慢）；三个集成 bug |
| **EAGLE-1 草稿层** | `layers/eagle.py`（无 RoPE transformer 层，F(h_t,e(w))→h̃；**对角注意力退化为 o=v**；SDPA 需 [1,heads,n,hd] 4-D）；`llm_engine.py` `_eagle_drafts`（按步跨 seq 批量自回归）；`benchmarks/eagle_train.py`（自蒸馏 CE+特征MSE） | `_eagle_quality.py`（teacher-forced 命中率）、`spec_bench.py --speculative eagle --max-draft-len` | **γ 是成本关键**：0.6B 上 γ=2 repeat +3.26×（α 0.525）、γ=4 +0.63×（每草稿一次 LM head ~0.8ms + 特征误差累积）；自由文本被 35% 可预测性封顶 |
| **fp8 varlen 内核** | `attention.py` `paged_varlen_attention_fp8_kernel` | `_fp8_varlen_check.py`（bit-exact） | 逐列因果掩码 `key_pos <= seqlen-qlen+r`（`<` 差 1 的 bug 故事） |

---

## 阶段 6：框架设施（可选，1-2 小时）

| 功能 | 读什么 |
|---|---|
| **张量并行** | `model_runner.py` `loop/read_shm/write_shm/call`（SharedMemory + Event 命令分发）；`linear.py` 各并行层的 `weight_loader`（Column/Row/Merged/QKV 分片）；`embed_head.py` 的 all_reduce/gather |
| **torch.compile 层** | `layernorm.py`/`activation.py`/`rotary_embedding.py`/`sampler.py` 上的 `@torch.compile`（与 enforce_eager 无关） |
| **计时与指标** | `sequence.py` 的 `t_submitted/t_first_token/t_completed`；`llm_engine.py` `collect_metrics`；`benchmarks/bench.py` 的报告生成 |

---

## 阶段 7：多模型适配（已完成：注册表 + 按层加载 + Qwen2.5 + Llama-3.1 + Mistral（SWA）+ Gemma-2（soft-cap）+ 脚本 argv 化）

| 功能 | 读什么 | 状态 |
|---|---|---|
| 模型注册表 | `models/registry.py`（`get_model_class(model_type)`）；`model_runner.py` 第 34 行 | **已实现**：qwen3 / qwen2 / llama / mistral / gemma2；mixtral 占位（构造时报具体卡点） |
| Qwen2.5 端口 | `models/qwen2.py`（模板 `models/qwen3.py` 删 QK-Norm） | **已实现**：0.5B/7B 验证（0.5B 与 HF 参考 top-1 100% 一致） |
| Llama-3.1 端口 | `models/llama3.py`（qwen2.py 模板，`attention_bias` 默认 False） | **已实现**：8B int4/w8a8 流式 16GB 卡跑通（int4 峰值 11.62GB）；`rope_scaling` llama3 变体已实现并单元对照 HF（0 误差） |
| Mistral-7B 端口（SWA） | `models/mistral.py`（llama3.py 模板 + `sliding_window`）；`layers/attention.py` 的 `window_size`/`_flash_window`；fp8 内核 `WINDOW` | **已实现**：与 HF 参考 top-1 100%（mean diff 0.014）；int4 流式 16GB 卡峰值 11.37GB；4876-token 跨窗口压测通过；bs=32 311 tok/s |
| Gemma-2 端口 | `models/gemma2.py`（`gemma2_layer_types` / embed ×√d / 层内双残差四 norm / soft-cap / gelu_tanh）；`layers/layernorm.py` 的 `weight_offset` | **已实现**：2B 与 HF 参考 top-1 100%（mean diff 0.022）；bs=64 1095 tok/s；三个"读源码才能发现"的架构细节见 note.md 故事 15 |
| 按层流式加载 + 即时量化 | `loader.py` `load_model(streaming=True)` + `_load_streaming`；`model_runner.py` `_decide_streaming`/`_streaming_quant_hook`/`_finalize_streaming` | **已实现**：Qwen2.5-7B（峰值 10.66GB）/ Llama-3.1-8B（峰值 11.62GB）/ Mistral-7B（峰值 11.37GB） |
| 脚本 argv 化 | benchmarks/ 全部脚本（模型路径 = 第一位置参数 / `--model`，缺省 Qwen3-0.6B） | **已实现**：22+ 个脚本，新模型可直接复用全部基准/诊断 |

### 卡点清单（剩余模型适配的阻塞点，对应 registry 里的 `_PLANNED_BLOCKERS`）

| 模型 | 需要的改动 | 卡点 / 依赖 | 工作量 |
|---|---|---|---|
| **Mixtral / MoE** | MoE 层：router + top-k + expert FFN + load-balancing aux loss + 专家并行分片 | registry 已占位（构造报错）；本机无小 MoE 真模型可验证（Qwen3-30B-A3B int4≈15.3GB 放不下 16GB）；只能合成小 MoE 验证机制 | ~1-1.5 天（概念必会，本机验证受限） |
| **通用** | `get_rope` 其余 `rope_scaling`（yarn/linear/dynamic） | llama3 变体已实现（波长分段缩放，`_scaled_inv_freq_llama3`，单元对照 HF 0 误差）；yarn/linear/dynamic 仍未实现 | 半天（频率插值内核） |
| **通用** | SWA 滚动块复用（真省 KV 显存） | 当前只掩码不滚动（与 vLLM 一致）；flash-attn 从块表索引推导 key 位置 → 滚动表需 flash fork 或自研内核 | ~1 天（明确设计，未实现） |
| **通用** | 流式加载限制 | int4 强制纯 int4（无 w_deq 双路径）、w8a8 无 SmoothQuant 校准、awq 仅预生成 scales（内联校准需全 fp16 模型） | 已知限制，非阻塞 |

**已解决的卡点**：Llama 的 `attention_bias` 默认 False；`rope_scaling` 的 llama3 变体；**Mistral 滑动窗口**（`window_size=(W-1,0)` 约定 + fp8 内核 WINDOW 掩码 + m 从 0 起步的 NaN 修复）；**Gemma-2**（layer_types 复刻安装版默认、embed ×√d、RMSNorm (1+weight) 偏移、层内双残差四 norm、flash 原生 softcap、fp8 KV 断言拦截）。

**流式加载的坑（阶段 7 特有）**：①meta 物化必须用 `to_empty`（torch 2.8 禁止 `.to()` 与 set_data 跨 meta）；②`to_empty` 替换 Parameter 对象 → 丢掉 `weight_loader`，须按模块重挂；③**计算型 buffer（RoPE `cos_sin_cache`）在 meta 上无数据，物化后是全零 → q/k 被零旋转 → 逐层发散**——`_finalize_streaming` 必须 `build_cache()` 重建；④tie 词表时文件不含 `lm_head.weight` → 加载后重绑（先物化再 `weight.data =` 共享存储）。

**裸模型诊断脚本的坑（阶段 7 补）**：CPU 构造再 `.to(cuda)` 会重设每个 Parameter 的 `.data`，
打破 `__init__` 的 tie 共享 → tie 模型 checkpoint 又常不含 `lm_head.weight` → lm_head 残留
空张量 → **logits 全零（ppl=词表大小的 uniform）**。裸模型加载后必须按 tie 配置重绑
（`_quant_ppl.py`/`_int4_layers.py`/`_sparse24_layers.py`/`_w8a8_layers.py` 已含）。
信号：ppl 恰好等于 `math.exp(ln(V))` = 词表大小。

---

## 验证/回归工具箱（每改完一个功能必跑）

| 脚本 | 验证什么 | 何时跑 |
|---|---|---|
| `python -m pytest tests/ -q` | 调度/块管理/投机/注册表/layer_types 纯 Python 逻辑（41 个用例） | 任何引擎改动后 |
| `benchmarks/_swa_probe.py` | SWA window/softcap 约定 + fp8 内核窗口掩码 vs torch 参考 | 动 attention.py / 内核后 |
| `benchmarks/_parity.py <model>` | 新架构端口 vs HF 参考 logits（top-1 100% = 端口正确） | 新增/修改模型文件后 |
| `benchmarks/_port_smoke.py <model> int4 --long` | 新模型 int4 冒烟 + 长上下文（跨 SWA 窗口） | 新增模型后 |
| `benchmarks/_softcap_probe.py <model>` | attn soft-cap 的 tanh 近似误差（真实 logits 上） | 动 gemma2 后 |
| `benchmarks/_fp8_kernel_check.py` | fp8 注意力内核 vs 参考 | 动 attention.py 后 |
| `benchmarks/_int4_check.py` | int4 内核 + 双路径一致性 | 动 linear.py 后 |
| `benchmarks/_sparse24_check.py` | 2:4 内核 vs 剪枝参考 | 动 sparse24 后 |
| `benchmarks/accuracy_check.py <model> <quant> <kv>` | 引擎级 logits 对齐/KL/top-1 | 任何量化/图改动后 |
| `benchmarks/_spec_equiv_check.py --fp8` | 投机 verify 与 plain 路径对齐 | 动 spec/graph/attention 后 |
| `benchmarks/_quant_ppl.py <model>` | 端到端困惑度（决定性精度指标） | 量化校准改动后 |
| `benchmarks/_qwen2_smoke.py <model> <quant> <streaming> <kv> [--check-hf]` | 多模型端口冒烟：生成 + 结构字段 + 显存 + 可选 HF 参考 logits 对比 | 新增/修改模型文件或加载器后 |
| `benchmarks/_stream_weights_check.py <model> [--runner]` | 流式加载 vs eager 逐参数/buffer 对比 | 动 loader.py / streaming 路径后 |

> **模型路径传参约定（1.3）**：benchmarks 下所有脚本已 argv 化——模型目录 = 第一个位置参数
> （如 `accuracy_check.py <model> <quant> <kv>`、`_quant_mem.py <model> <mode>`），argparse 脚本用
> `--model`；缺省均为 `~/huggingface/Qwen3-0.6B/`，不传参行为与之前完全一致。

## 面试对照

每个功能在 `INTERVIEW.md` 里都有对应工作项（§1.x）、数字速查（§2）、问答（§3）与踩坑故事（§4）——学完一个阶段后去读对应段落，用"能讲清数字怎么来的"作为掌握标准。
