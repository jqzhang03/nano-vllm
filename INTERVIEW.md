# INTERVIEW.md — nano-vllm 总档（学习路线 · 面试串讲 · 基准档案）

> **本文件是原 `BENCHMARKS.md` + `INTERVIEW.md` + `LEARNING.md` 三合一合并版**（2026-09-07）：
> 重复内容只写一遍。实现状态按 2026-10-05 源码校对；代码位置以核心符号定位，历史性能数据按原测试日期保留。
> 原三文件已删除；git 历史保留旧版。
>
> **姊妹文件**：`note.md` = 个人时间线工作梳理（截止 2026-08-24 阶段 1.12，其踩坑故事编号
> 与本档 §6 一一对应，见 §11 映射）；`benchmarks/_stage2b_ext_report.md` = 阶段 2b-ext
> 的完整证据表与复现命令。
>
> **全部数字带条件**：单卡 RTX 5060 Ti 16GB（Blackwell sm_120，36 SM）/ WSL2 Ubuntu /
> conda `nano-vllm`（torch 2.8.0+cu128、triton 3.4.0、flash-attn 2.8.3.post1）/ bf16 /
> Qwen3-0.6B 为主，除非另注模型。WSL 内存 11GB + 4GB swap（`C:\Users\admin\.wslconfig`）。
>
> **实现状态快照（2026-10-06）**：代码已包含 OpenAI 风格在线服务与 SSE、进程内会话文本/截断/可选摘要、每会话 prompt token 预算和累计前缀命中统计、持续请求接收、动态准入与背压、Top-W cache-affinity、aging、recompute-aware 抢占、TTFT SLO 感知调度与自适应 prefill 配额、TP=1 的 auto/FP8 KV CPU swap，以及实验性本机 PD 分离。Prefix feature context 只为 Top-W 排序/配额候选懒解析，并按 BlockManager 的 `kv_generation` 失效；`prefix_feature_cache=False` 可保留亲和策略但改为每次解析，作为独立消融。CPU 生命周期测试覆盖特征失效、LRU/容量淘汰、swap 释放、共享 deferred-free 提交与计数重置；benchmark 记录 parse/reuse、stale reparse、generation mutation、eviction 和 deferred-free 排队/提交引用数及峰值唯一块数/引用数。缓存回收优先消耗未缓存空闲块，再按真实缓存复用 LRU 淘汰空闲前缀块，完成批次的 deferred-free 在 postprocess 结束后统一提交。普通 MHA mixed Prefill/Decode batch 增加了按形状惰性捕获、有限 LRU 图缓存的 CUDA Graph，MLA、rolling/split 和 spec-mixed 路径继续 eager。新增 `benchmarks/context_concurrency.py` 提供并发 × 上下文长度压测和实际 KV 池容量账本；`benchmarks/kv_fp8_calibrate.py` 生成真实文本校准 token，并按 scale margin 对比 FP8 与 auto KV 的 held-out decode logits。动态准入通过引擎空闲时的不可变 prefix-cache 快照估算命中 token、prefill 工作量与增量 KV 块，并结合输出长度全局 EWMA、实测 Prefill/Decode 速率、队列压力；过载时有限时 FIFO 延迟，超时/硬上限时返回 429。KV 容量预测仍按最大输出上限保守预留。单卡 `auto` 选择 mixed；多卡只有在配置兼容时才选 PD。PD 仍是单进程、双模型/KV 池、主机内存交接和串行阶段执行。
>
> **本轮实测（2026-10-06，RTX 5060 Ti / WSL2，Qwen3-0.6B，mixed 单卡）**：**⑤ 复跑修订（当日第二次）**：上一轮把 `without_decode_burst_yield` 判为"结构上不可达"，方向对但结论不完整——那次运行里新功能只写了一半：`_run_decode_burst` 收集了 `finished` 却没返回（调用点解包 2 元组 → 任何进入 burst 的 step 直接 `TypeError`）、`Config` 缺 `decode_burst_yield_on_arrival`、服务端调用的 `_sync_decode_burst_pressure()` 未定义、准入用的 `_estimate_prefix_cache_match()`/`_projected_kv_blocks()` 未定义、`_estimate_admission` 里返回键引用了不存在的变量。补齐后重跑：**burst 中途到达确实能打断后续轮次**（服务端实测 8/8 轮让出，迟到请求尾延迟 68.3 → 53.6 ms，最差轮 676 → 329 ms，§10.3.11），原命令重跑结论不变（TPOT 配额仍是 collapse 根因，§10.3.10）；CPU 用例 `tests/test_decode_burst_yield.py` 钉住返回 arity 与到达压力谓词。**⑥ 指标正确性与 TPOT 饥饿修复（当日第三轮）**：①per-request 指标对 burst 内完成的序列重复落盘，128 请求只统计到 95 个不同 seq_id（另 38 条重复），驱动侧据此**提前结束跑批**——已改 `_record_once()` 幂等 + `reset_benchmark_metrics()` + harness 断言，所有变体现为 128/128 唯一；②TPOT 预填配额收缩加**死区**与**队列深度门限**（`tpot_prefill_throttle_margin=0.5` / `tpot_throttle_max_waiting=16`），把 self-locking 的 floor 钉死解开：128 请求 / 16 req/s / 20 ms 目标下 all_on 350 → **1140 tok/s**、TTFT p50 44.5 s → **1.6 s**、TPOT p50 173 → **82 ms**；③准入校准换到会打穿 SLO 的负载（24 req/s、128 请求）：关闭准入时只有 **26.6%** 的请求在 2 s 内拿到首 token，开准入（感知缓存）后接受 15.6% 请求、被服务者 **95%** 达标、吞吐 777 tok/s（不感知缓存仅 617）。①**前缀命中估算已与实测对齐**——准入现在按 `BlockManager.prefix_snapshot()`（engine 空闲时物化的不可变哈希表副本 + `prefix_map_version()` 失效）用候选请求的链式块哈希估算可复用 token，`cache_affinity_admission=False` 时恒为 0（"不感知缓存"消融）。进程内探针 `benchmarks/prefix_cache_probe.py` 10/10 请求估算==实际（0/256/512 三档，绝对误差 0，全部块对齐）；HTTP `benchmarks/prefix_cache_verify.py` 8/8 请求 `admission.estimated_cache_hit_tokens == context.prefix_cache_hit_tokens`（冷启动服务：首发 0/0、复用 512/512，Δ=0）。②**准入消融**（`admission_ablation.py` 三档：关闭 / 开但不感知缓存 / 开且感知缓存，同一到达 trace）：1 req/s、32 请求、共享前缀 512 时三档吞吐 264.9/264.5/264.6 tok/s、**0 拒绝 0 延迟**——低费率下准入根本不触发，缓存感知只把命中估算误差从 -496 token 修正到 0。4/8/16 req/s poisson、64 请求下：**缓存感知在 4-8 req/s 稳定有用**（比不感知多接受 4~13 个请求、吞吐 +4.9%~+51%，估算误差 -490 → -10~0 token），但**默认准入阈值整体是净损失**——相对关闭准入吞吐 -12.4%（4 req/s）、-12.2%~-54.0%（8 req/s，3 seed）、-45.0%（16 req/s），越忙越亏；16 req/s 饱和点上感知与不感知打平（接受数相同）。费率带宽、多种子重复与逐请求命中对照见 `results/admission_ablation_*.json`。③**decode burst 让出**：`decode_burst_yield`（默认开）新增为独立消融项 `without_decode_burst_yield`（关掉后 burst 即使有新 prefill 到达也跑满 `max_decode_steps`），并新增 `decode_bursts/rounds/yields/skipped_slots` 计数。④**修掉两处"跑不通"**：`Qwen3/Qwen2/Llama3Attention` 直接用 `dist.get_world_size()`，TP=1 时进程组未初始化 → 模型构造即崩（服务与离线路径都受影响，改用 `tp_size()` 回退）；`admission_ablation.py` 未在变体之间 `gc.collect()+empty_cache()` → 第二个变体 KV 分配断言失败。新实现尚未在 RTX 5060 Ti/WSL2 上验证收益，不作为性能结论。
>
> **当前限制**：混合 CUDA Graph、multi-step decode 与 TPOT 调度尚未在 RTX 5060 Ti/WSL2 上运行 benchmark；CUDA Graph 首次遇到每种形状会额外执行一次 eager warmup 和 capture，最多保留 `mixed_cudagraph_max_graphs` 种形状，超过 `mixed_cudagraph_max_tokens`（默认 4096）则 eager 回退。Multi-step 只覆盖纯非投机 decode，默认最多 4 轮；在线服务在 burst 结束后发送这段时间产生的 token，因此可能合并为一个 SSE 文本块。TPOT 调度是基于请求级 token 间隔 EWMA 的启发式目标，不保证硬 deadline。服务准入的输出长度比例是跨请求的全局 EWMA，未按请求类型区分；**前缀命中预测已实现且与实测一致，但准入阈值本身未校准**——2026-10-06 实测 8 req/s 下默认参数（work budget 10s / soft pressure 0.85）会拒绝约一半请求并损失 31~36% 吞吐（详见 §10.3.9），说明"估算准"不等于"准入策略对"；Mixtral 未实现。PD 缺少异步 P/D 重叠、远端 worker/RDMA，且本机单卡环境不能验证多卡收益。历史性能表不作为新增策略的效果证明。
>
> **串讲三原则**：①先讲成本模型与上界，再讲实现；②主动交代"哪里亏、为什么"（比吹嘘可信）；
> ③所有结论要么有探针证据、要么明确标注"未验证"。方法学信条：**跑通 ≠ 写对**。

---

## 目录

- §1 学习路线（怎么读代码） · §2 电梯陈述 · §3 主线叙事 · §4 深水区问答
- §5 数字速查 · §6 踩坑故事 · §7 方法论 · §8 精进路线图 · §9 代码地图 · §10 基准档案 · §11 附录

---

## 0. 全局图景（30 分钟，先读文档）

| 读什么 | 要点 |
|---|---|
| `CLAUDE.md` | 架构总览：请求生命周期、在线服务、Context 单例契约、KV cache/offload、调度、投机、CUDA graph、TP/PD |
| 本档 §10.1 | TTFT/TPOT/E2E/p50/p99/SLO 的口径——后面所有数字都基于它 |
| `AGENTS.md` | 模块组织、开发约定 |
| `nanovllm/config.py` | 全部开关：quantization/speculative/kv_cache_dtype/int4_dense_path/awq_scales_path/rolling_cache……每个字段对应一个功能 |

**目标**：能说出"一次 `LLM.generate` 从进队列到出 token 经过了哪几个大环节"（§9.2 链 B）。

---

## 1. 学习路线（按依赖排序）

按"先懂主线、再懂优化、最后懂投机与量化"的顺序组织。每步：**读什么（文件/类/函数）**、
配套脚本、验证出口。建议配合 `git log` 看每个功能的提交历史（提交信息是短摘要，能还原当时的问题与解法）。

### 1.1 主链路速读（最重要，2-4 小时）：十步走通一次生成

**一次生成的全旅程**：`LLM.generate` → `Scheduler.schedule` → `ModelRunner.run` → 模型 forward →
`Sampler` → `Scheduler.postprocess` → 循环。

| 步 | 读什么 | 验证出口 |
|---|---|---|
| 1 | `example.py` + `llm.py` + `config.py` | 跑通 `python example.py` |
| 2 | `llm_engine.py` 的 `generate`/`step`（§9.2 链 B） | 打断点看每步的 kind 变化 |
| 3 | `scheduler.py` 的 `schedule` + 四个 `_schedule_*` | 打印每步 (seqs, kind) |
| 4 | `model_runner.py` 的 `prepare_*` + `context.py` | 打印 `set_context` 的各张量 shape |
| 5 | `models/qwen3.py`（一个模型吃透，其余是变体） | 对照 HF 实现看逐层等价 |
| 6 | `layers/attention.py` 的 `forward` 路由（§9.2 链 D） | 三种批次形态各跑一次 |
| 7 | `layers/linear.py`（量化全链） | `--quantization int4` 对比输出 |
| 8 | `block_manager.py`（前缀缓存 + COW） | `--shared-prefix-len 512` 看命中 |
| 9 | `model_runner.py` 的 CUDA graph 两段 | `enforce_eager` 开/关对比 |
| 10 | 投机（`ngram.py` → `_verify` → spec graph）→ TP → 流式加载 | `benchmarks/spec_bench.py` |

第一轮的文件级精读（类/函数锚点见 §9.1）：

| 文件（类/函数） | 学习要点 |
|---|---|
| `nanovllm/llm.py`、`nanovllm/sampling_params.py` | 入口与采样参数（禁 greedy——Sampler 用 Gumbel，温度必须 >1e-10；`torch.manual_seed` 不控 GPU RNG，需 `torch.cuda.manual_seed`） |
| `nanovllm/engine/sequence.py` | `Sequence`：token 存储、`block_table`/`kv_table`、`num_cached_tokens`、`__getstate__/__setstate__`（TP 跨进程） |
| `nanovllm/engine/llm_engine.py` | `generate`（主循环）、`step`（一次调度+前向+postprocess）、`_verify`、`collect_metrics` |
| `nanovllm/engine/scheduler.py` | `schedule`（kind 分发）、`_schedule_*`、`postprocess`（append/EOS/哈希）、`preempt`（KV 不足抢占/swap 分流） |
| `nanovllm/engine/model_runner.py` | `run`（入口）、`prepare_prefill`/`prepare_decode`（打包）、`run_model`（前向 + CUDA graph 选择） |
| `nanovllm/models/qwen3.py` | 模型结构 + `packed_modules_mapping` + `compute_logits` |
| `nanovllm/layers/attention.py` | `Attention.forward`：prefill 走 varlen、decode 走 kvcache；`store_kvcache` 写缓存 |
| `nanovllm/layers/layernorm.py` / `rotary_embedding.py` / `activation.py` / `sampler.py` / `embed_head.py` | 各基础算子与 `@torch.compile`（与 `enforce_eager` 无关，首轮前向必有 JIT） |
| `nanovllm/utils/context.py` | **每步张量从 runner 传给内核的契约**——`set_context/get_context/reset_context`，新字段必须追加在 dataclass 末尾 |
| `nanovllm/utils/loader.py` | `load_model` + `weight_loader` 约定（packed 映射：q/k/v→qkv_proj） |

**建议读法**：先看 `qwen3.py` + `layers/`（模型长什么样）→ 再看 `model_runner.py`（张量怎么打包）→
`scheduler.py`（批次怎么选）→ `llm_engine.py`（循环怎么转）→ 最后 `context.py` 把所有数据流串起来。

### 1.2 调度与内存管理（2-3 小时）

| 功能 | 读什么 | 配套脚本/文档 | 面试要点 |
|---|---|---|---|
| **混合调度**（vLLM V1 同款） | `scheduler.py` `_schedule_mixed`；`model_runner.py` `prepare_mixed`；`attention.py` 混合路由；`embed_head.py` `ParallelLMHead` 的 `is_mixed` 分支 | `benchmarks/bench.py`；§10.3.2 | 为什么比"先全 prefill 后 decode"好（死等消除、抢占下降） |
| **前缀缓存 + COW** | `block_manager.py`：`compute_hash`（链式哈希）、`can_allocate`、`allocate`、`hash_blocks`（部分块也发布哈希）、`cow_block`（写共享块前复制）、`can_append/may_append` | `tests/test_block_manager.py`；§10.3.2 | 哈希链为什么带前块哈希；部分块缓存为什么安全；COW 在 GPU 上怎么执行 |
| **分块 prefill** | `scheduler.py` `_schedule_prefill`（只允许第一个序列切块）；`model_runner.py` `prepare_prefill`（key 超 query → 缓存形状 K/V + block_tables） | §10.3.2 | 前缀命中时 K 长于 Q 的 varlen 怎么表示 |
| **抢占与恢复** | `scheduler.py` `preempt`（**KV swap 分流**：decode 序列换出到 CPU `swap_out`/`swap_in` + 独立 `swapped` 队列 + `kv_swap_space_gb` 预算；prefill 序列 recompute）；`block_manager.py` `allocate_private`/`release_blocks`；`model_runner.py` `swap_out`/`swap_in`（**`index_copy_` 原位写**） | `bench.py --no-swap-kv`、`_swap_smoke.py`、`_swap_bitexact.py` | swap 比 recompute 保持采样流确定（bit-exact 免重算）；**本机 0.6B+WSL2 上重算更便宜（swap 27.8s vs 11.8s）**；价值在 7B+ 与真实 Linux |
| **滚动环**（阶段 2b，Mistral/gemma2-local） | `block_manager.py`（`ring_cap`/`_evict_front`/`_t`/`no_share`）；`scheduler.py` 滚动断言；`model_runner.py` `_finalize_rolling`；`attention.py` `_ring_varlen`/bf16 环 decode 内核 | `benchmarks/_stage2b_ext_report.md`；§8 阶段 2 | 窗口内容清单 = 块表、驱逐先释放再分配；**滚动模型停用前缀缓存**（内容过期，重复 prompt 有代价） |

### 1.3 CUDA graph 与启动税（1-2 小时）

| 功能 | 读什么 | 配套脚本 | 面试要点 |
|---|---|---|---|
| **decode CUDA graph** | `model_runner.py` `capture_cudagraph`（批量族 [1,2,4,8]+16 步进、共享内存池）、`run_model` 的图选择与静态输入拷贝 | `bench.py`（默认非 eager） | graph 捕获要求固定形状/地址；`enforce_eager` 只关图不关 torch.compile |
| **spec verify CUDA graph** | `model_runner.py` `capture_spec_graph`（容量族 × 双 stride、零长度填充行）、`_spec_graph_hidden`、`run_model` 的 spec 重放 | `_graph_pad_probe.py`（bit-exact）、`_verify_probe.py`、`_spec_step_timing.py` | varlen 用固定容量图 + 空行填充（cu_seqlens 尾部重复末值，flash 按空行跳过）；`max_seqlen_q/k` 烘焙为标量无开销 |
| **启动税诊断**（方法学） | `benchmarks/_verify_probe.py`、`_step_timing.py` | — | 探针分层：分页 vs 连续、形状、CPU launch 计数——**先证伪假设再修**（spec 步 38ms = GPU 25ms + CPU 启动税 ~10ms） |

### 1.4 量化（3-5 小时，按依赖顺序）

#### 1.4.1 FP8 KV cache（先看，注意力内核最独立）
| 读什么 | 要点 |
|---|---|
| `model_runner.py` `calibrate_fp8_kv` | 支持 JSON token-ID 校准集并按预算分批；未提供文件时用随机 token；每层固定 scale = max/448×margin，MLA fused 行 [c_kv\|k̃_pe] 两段独立 scale |
| `attention.py` `store_kvcache_kernel` | 写路径：fp32→fp8 cast **不饱和产生 NaN 位模式，必须 clamp(-448,448)**（§6 故事 1） |
| `attention.py` `paged_decode_attention_fp8_kernel`（v6） | decode 内核：直接 fp8 load + 硬件 cvt 反量化、QPAD=16 MMA、GQA 融合、BLOCK_T=32/warps=1 |
| `attention.py` `paged_varlen_attention_fp8_kernel`（v7） | 投机 verify 的多查询扩展（逐列因果掩码必须 `<=`，§6 故事 3） |
| 配套 | `benchmarks/kv_fp8_calibrate.py`（真实文本 scale-margin 扫描、held-out 首个 decode logits、范围余量）及 `_fp8_kernel_check.py`、`_kernel_bench.py`、`accuracy_check.py`；§10.3.3 |

#### 1.4.2 W8A8（int8 GEMM + SmoothQuant）
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_int8_kernel`/`w8a8_gemm` | per-group(128) 权重 scale，BLOCK_K=128=组大小，int32 累加后乘组 scale 以 fp32 跨组累加 |
| `linear.py` `LinearBase.quantize_w8a8`/`_w8a8_forward` | SmoothQuant 折叠：`s = x_max^0.5 / w_col^0.5`，`W'=W·s, X'=X/s` 恒等变换 |
| `model_runner.py` `calibrate_and_quantize_w8a8` | 校准 hook 收集逐通道 amax |
| 配套 | `_w8a8_check.py`；§10.3.4 |

#### 1.4.3 INT4 + AWQ（当前主力）
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_int4_kernel`/`int4_gemm` | 2-dot 拆分：按 K 奇偶拆 a 与半字节、两个 dot；打包沿 K；tile 按 M 自适应；**尾 K 掩码 + 组粒度静态展开**（`int4_group_size`，真实模型 K=10944 非 128 倍数，§8 阶段 2b-ext） |
| `linear.py` `WeightQuantMixin.quantize_int4`/`_int4_forward` | per-group 对称 int4；**双路径路由**：`w_deq`（bf16 反量化副本）供大 M/小 N 走 cuBLAS，`M≤128 且 N≥2048` 走 int4 内核 |
| `model_runner.py` `quantize_int4_weights`/`quantize_awq_weights`/`_calibrate_awq_scales` | 加载后量化；AWQ 缩放文件加载或内联随机校准 |
| `benchmarks/awq_calibrate.py` | **按层 α 搜索**：`s=(mean\|X\|/w_col)^α`，目标 = 校准批量化输出误差；方向 `W'=W·s, X'=X/s`（论文方向，反了会塌缩，§6 故事 7） |
| `config.py` `int4_dense_path`/`quantize_lm_head`/`awq_scales_path`/`int4_group_size` | 双路径开关 / lm_head 量化（默认关）/ 校准文件 / 组大小 |
| 配套 | `_int4_check.py`、`_quant_ppl.py`（**端到端 ppl 是决定性指标，注意 run 间波动**，§10.3.6）、`_awq_diagnose.py`；§10.3.6 |

#### 1.4.4 2:4 结构化稀疏
| 读什么 | 要点 |
|---|---|
| `linear.py` `gemm_sparse24_kernel`/`sparse24_gemm` | 4 路拆分：a 按 K 步长 4 加载、`idx==p` 掩码重建权重块、4 个 dot；打包 `v [N,K//2] bf16` + `idx [N,K//4] uint8` |
| `linear.py` `WeightQuantMixin.quantize_sparse24` | 幅值剪枝（组内保留最大 2）+ 打包 |
| 配套 | `_sparse24_check.py`、`_sparse24_probe.py`（**cuSPARSELt/CUTLASS 在 sm_120 的结论**）；§10.3.6 |

### 1.5 投机解码（3-4 小时，依赖 1.3 的 graph 概念）

| 功能 | 读什么 | 配套脚本 | 面试要点 |
|---|---|---|---|
| **n-gram 草稿** | `engine/ngram.py` `find_ngram_draft`（窗口 4→1 回退、EOS 截断、预算封顶）、`verify_drafts`（点质量验收） | `tests/test_spec_decode.py` | 验收为什么严格保持分布（输出恒等于目标采样） |
| **verify 步 = varlen prefill** | `model_runner.py` `prepare_spec`/`_prepare_mixed_spec`；`scheduler.py` `_compute_draft`/`_spec_rows`/`_schedule_spec`/`postprocess_spec`；`llm_engine.py` `_verify` | `_spec_equiv_check.py`（三层验证） | query=[末 token+草稿]、num_cached=len-1；**KV 提交语义：被拒草稿不回滚、哈希只发布到接受长度** |
| **Medusa 多头** | `layers/medusa.py`；`llm_engine.py` `_medusa_drafts`（行选择 + 全接受 shift）；`model_runner.py` medusa 加载 | `benchmarks/medusa_train.py`（自蒸馏）、`_medusa_debug.py`、`_medusa_integration.py` | head_k 语义（预测 t+k+1）；训练必须 exit 引擎（allocator 60× 慢）；三个集成 bug（§6 故事 2） |
| **EAGLE-1 草稿层** | `layers/eagle.py`（无 RoPE 层，F(h_t,e(w))→h̃；**对角注意力退化为 o=v**；SDPA 需 [1,heads,n,hd] 4-D，§6 故事 13）；`llm_engine.py` `_eagle_drafts`；`benchmarks/eagle_train.py` | `_eagle_quality.py`、`spec_bench.py --speculative eagle` | **γ 是成本关键**：0.6B 上 γ=2 repeat +3.26×（α 0.525）、γ=4 只有 +0.63×（每草稿一次 LM head ~0.8ms + 特征误差累积）；自由文本被 35% 可预测性封顶 |
| **fp8 varlen 内核** | `attention.py` `paged_varlen_attention_fp8_kernel` | `_fp8_varlen_check.py`（bit-exact） | fp8+spec 从 0.15× 变 +3.92×（bs=8 repeat）：消除"逐层全缓存反量化"（~18GB/步 搬运） |
| **投机 × 滚动环**（2b-ext） | `block_manager.py` 环 spec 账本（表项 = 逻辑块 j0+i）、verify 行 key 集 [j0·B, end) 稠密装配 | `_ring_spec_e2e.py`；§8 阶段 2b-ext | 环上 verify = 把环内容装配成段喂 flash，段内相对下标 ⇒ 窗口掩码精确 |

### 1.6 框架设施（可选，1-2 小时）

| 功能 | 读什么 |
|---|---|
| **张量并行** | `model_runner.py` `loop/read_shm/write_shm/call`（SharedMemory + Event 命令分发）；`linear.py` 各并行层 `weight_loader`（Column/Row/Merged/QKV 分片）；`embed_head.py` 的 all_reduce/gather |
| **torch.compile 层** | `layernorm.py`/`activation.py`/`rotary_embedding.py`/`sampler.py` 上的 `@torch.compile`（与 enforce_eager 无关） |
| **计时与指标** | `sequence.py` 的 `t_submitted/t_first_token/t_completed`（driver 侧）；`llm_engine.py` `collect_metrics`；`benchmarks/bench.py` |

### 1.7 多模型适配、流式加载与卡点清单

| 功能 | 读什么 | 状态 |
|---|---|---|
| 模型注册表 | `models/registry.py`（`get_model_class(model_type)`） | **已支持**：qwen3 / qwen3_moe / qwen2（Qwen2.5 同属）/ llama / mistral / gemma2 / deepseek_v2 |
| 单模型模板 | `models/qwen3.py` → 删 QK-Norm = qwen2；加 `attention_bias=False` = llama3；加 `sliding_window` = mistral；gemma2 是"读源码才能发现"的三细节家族 | 详见 §4.6/§8 各阶段 |
| MoE（1.5） | `layers/moe.py`（`MoE`/`ExpertFFN`：2D per-expert 对齐 HF 存盘格式，router 永不量化）+ `models/qwen3_moe.py`/`deepseek_v2.py`（`DeepseekV2Moe`） | **已实现**：数学同构位级对照 + CPU 单测 + 端到端 parity top-1 100%（mean diff 0.003）；grouped 批量后端（§4.7） |
| MLA（2a） | `layers/attention_mla.py` + `models/deepseek_v2.py`（fused [c_kv\|k̃_pe] 缓存、吸收式 decode 内核、共享 rope key） | **已实现**：decode 内核 vs 稠密参考位级 0 误差；引擎 parity top-1 100% |
| 滚动环 / split（2b/2b-ext） | `config.py` `rolling_cache`；`block_manager.py` 环驱逐；双池（gemma2 交替窗口） | **已实现**：真实 Mistral-7B/gemma-2-2b-it/DeepSeek-V2-Lite 验证（§8 阶段 2） |
| 在线服务 / 会话上下文 | `server.py`：OpenAI 风格 completions/chat、SSE、断连取消、会话 GET/DELETE、截断/摘要压缩 | **已实现**：文本会话默认最多 256 个、空闲 24h 过期；重启不持久化，摘要有损；KV 不跨轮保留 |
| 动态准入与背压 | `server.py` `GenerationManager`：prompt 工作量（已扣除前缀命中预测）、全局输出长度比例 EWMA、Prefill/Decode 吞吐 EWMA、队列/KV 压力；FIFO defer、超时 429 | **已实现，默认开启**：空闲时接收一个请求保证进展；硬队列上限 256、defer 上限 64、等待 2s；KV 预测按请求最大输出上限预留；前缀命中预测用 idle 期快照的 `prefix_snapshot()`+`estimate_cached_tokens`（估算与实测逐请求一致，见 §10.3.9）；`/health` 与响应暴露估算与实际命中；`--no-dynamic-admission`（开关消融）与 `--no-cache-affinity-admission`（不感知缓存消融）。**诚实结论**：默认阈值在 8 req/s 下过度拒绝、净损吞吐，阈值待校准 |
| 持续请求与调度策略 | `server.py` 单 engine worker；`scheduler.py` Top-W、aging、recompute-aware、TTFT/TPOT SLO、自适应分块/配额；`llm_engine.py` bounded multi-step decode（含 `decode_burst_yield`）；`block_manager.py` generation 失效、prefix 快照版本与 deferred-free | **已实现，有消融脚本**：`benchmarks/scheduling_ablation.py` 对齐同一到达 trace，记录服务指标、request TPOT 目标达成、decode forward/burst token 数、burst 轮数与提前让出次数/放弃槽位，以及 lazy-feature parse/reuse、stale reparse、generation mutation、LRU eviction、deferred-free queued/committed/peak；包含 `without_tpot_aware`、`without_multi_step_decode` 与 `without_decode_burst_yield`。当前 checkout 尚无新策略的目标硬件实测结论；TPOT 是软目标启发式，不保证 deadline |
| KV CPU swap / FP8 | `scheduler.py` 缓冲与抢占；`model_runner.py` 拷贝；`kv_transfer.py` P/D KV 布局交接 | **已实现**：TP=1 的 MHA/MLA、auto/FP8 KV；同步普通 CPU 内存，预算受限；TP>1 回退 recompute |
| 执行模式自动选择 / PD | `config.py` / `llm_engine.py`：`auto|mixed|pd` | **已实现，PD 为实验性**：单卡 auto=mixed；兼容的多卡 auto=PD；TP>1、spec、rolling cache、非 auto KV 时 auto 回退 mixed；手动 PD 对不兼容配置报错 |
| 按层流式加载 + 即时量化 | `loader.py` `load_model(streaming=True)`；`model_runner.py` `_decide_streaming`/`_streaming_quant_hook`/`_finalize_streaming` | **已实现**：Qwen2.5-7B 峰值 10.66GB / Llama-3.1-8B 11.62GB / Mistral-7B 11.37GB；自动触发 = fp16 估重 > 空闲显存 45% 且启用量化 |

**卡点清单（未完成项或尚未验证的组合）**：

| 模型/功能 | 需要的改动 | 卡点 / 依赖 |
|---|---|---|
| **Mixtral 端口** | MoE 层已通（1.5），但 mixtral 具体端口（router + top-k norm_topk + aux loss 的 HF 对齐）未做 | registry 占位报错；本机无小 MoE 真模型可对照（DeepSeek-V2-Lite 已覆盖 MoE 真模型验证） |
| **rope_scaling 其余变体** | YaRN / linear / dynamic | 已支持：无操作（default）+ llama3 变体（波长分段，单元对照 HF 0 误差）；其余构造时报错 |
| **滚动环 × KV swap / ring × medusa/eagle** | 组合验证与记账 | 断言关（§8 阶段 2 诚实边界）；KV swap × ring/split 未验证 |
| **split 双池 CUDA graph** | gemma2 split 模式 decode 入图 | 现为 eager（动态 python 路径烘焙即错）；softcap 层禁 fp8 KV |
| **准入估算校准** | 按模型/批大小拟合 service rate、减少误拒和漏控 | 前缀命中预测已完成并与实测逐请求一致（§10.3.9）；仍未按模型/批大小校准 `admission_work_budget_ms`/`admission_soft_pressure`，2026-10-06 扫描显示关闭准入反而更快（4 req/s -12%、8 req/s -12~-54%、16 req/s -45%） |
| **multi-step decode / TPOT 调度** | 扩大支持到 mixed/spec decode、改善目标硬件效果 | 纯非投机 decode burst 与 request-level TPOT 排序/配额已实现；burst 上限默认 4；`decode_burst_yield` 默认开（有 prefill 等待即提前收尾），`without_decode_burst_yield` 为独立消融；mixed/spec 仍单步。RTX 5060 Ti/WSL2 效果尚未实测，TPOT 是启发式软目标 |
| **PD 生产化** | 异步重叠、远端 worker、网络/RDMA KV 传输、多卡基准 | 当前仅本地两卡、同步主机内存交接和串行阶段；单卡环境无法验证 |

**流式加载的坑（阶段 7 特有，详见 §6 故事 9/10/18）**：①meta 物化必须 `to_empty`（torch 2.8 禁止 `.to()` 与 `set_data` 跨 meta）；②`to_empty` 替换 Parameter → 丢 `weight_loader`，须按模块重挂；③**计算型 buffer（RoPE `cos_sin_cache`）meta 上无数据 → `_finalize_streaming` 必须重建**；④tie 词表文件通常不含 `lm_head.weight` → 加载后重绑（先物化再 `weight.data =` 共享存储）；⑤**meta 计数构造会污染 `get_rope` 的共享实例 → 必须 `cache_clear()`**（2026-09 修复，§6 故事 18）。

**裸模型诊断脚本的坑**：CPU 构造再 `.to(cuda)` 会重设每个 Parameter 的 `.data`，打破 `__init__`
的 tie 共享 → tie 模型 checkpoint 又常不含 `lm_head.weight` → lm_head 残留空张量 → **logits 全零
（ppl = 词表大小的 uniform）**。裸模型加载后必须按 tie 配置重绑（`_quant_ppl.py` 等已含）。
信号：ppl 恰好等于 `math.exp(ln(V))` = 词表大小。

### 1.8 验证与回归工具箱（每改完一个功能必跑）

| 脚本 | 验证什么 | 何时跑 |
|---|---|---|
| `python -m pytest tests/ -q` | 调度/块管理/投机/注册表/layer_types/KV 交接等 CPU 可测逻辑；用例数随测试扩充变化 | 任何引擎改动后 |
| `benchmarks/_swa_probe.py` | SWA window/softcap 约定 + fp8 内核窗口掩码 vs torch 参考 | 动 attention.py / 内核后 |
| `benchmarks/_parity.py <model>` | 新架构端口 vs HF 参考 logits（top-1 100% = 端口正确） | 新增/修改模型文件后 |
| `benchmarks/_port_smoke.py <model> int4 --long` | 新模型 int4 冒烟 + 长上下文（跨 SWA 窗口） | 新增模型后 |
| `benchmarks/_softcap_probe.py <model>` | attn soft-cap 的 tanh 近似误差（真实 logits 上） | 动 gemma2 后 |
| `benchmarks/_fp8_kernel_check.py` | fp8 注意力内核 vs 参考 | 动 attention.py 后 |
| `benchmarks/_fp8_varlen_check.py` | fp8 varlen（verify）内核 bit-exact | 动 varlen 内核后 |
| `benchmarks/_int4_check.py` | int4 内核 + 双路径一致性 + 组粒度路径 | 动 linear.py 后 |
| `benchmarks/_sparse24_check.py` | 2:4 内核 vs 剪枝参考 | 动 sparse24 后 |
| `benchmarks/accuracy_check.py <model> <quant> <kv>` | 引擎级 logits 对齐/KL/top-1 | 任何量化/图改动后 |
| `benchmarks/_spec_equiv_check.py --fp8` | 投机 verify 与 plain 路径对齐 | 动 spec/graph/attention 后 |
| `benchmarks/_quant_ppl.py <model>` | 端到端困惑度（决定性精度指标；**注意 run 间波动**） | 量化校准改动后 |
| `benchmarks/_qwen2_smoke.py <model> <quant> <streaming> <kv> [--check-hf]` | 多模型端口冒烟：生成 + 结构字段 + 显存 + 可选 HF 参考 | 新增/修改模型文件或加载器后 |
| `benchmarks/_stream_weights_check.py <model> [--runner]` | 流式加载 vs eager 逐参数/buffer 对比 | 动 loader.py / streaming 路径后 |
| `benchmarks/_stage2b_ext_report.md` | 阶段 2b-ext 组合（fp8 KV×MLA/环、spec×环、split 双池、纯 int4 MLA）全套证据 | 动 block_manager/attention_mla/环相关后 |

> **模型路径传参约定**：benchmarks 下所有脚本已 argv 化——模型目录 = 第一个位置参数
> （argparse 脚本用 `--model`）；缺省均为 `~/huggingface/Qwen3-0.6B/`。

## 2. 电梯陈述（30 秒 / 2 分钟 / 5 分钟）

### 30 秒
> 我从零实现了一个 vLLM 风格的推理引擎（纯 PyTorch + Triton + flash-attn，不依赖任何推理框架）：
> 连续批处理调度（混合 prefill/decode）、paged KV cache + 前缀缓存 + COW、CUDA graph、
> 五条量化路径（w8a8/int4/AWQ/fp8/2:4 稀疏）+ fp8 KV cache、三种投机解码（n-gram/Medusa/EAGLE）、
> KV swap 抢占、按层流式加载、MoE（Qwen3-MoE/DeepSeek-V2）、MLA + SWA 滚动环（Mistral/Gemma-2），
> 移植 Qwen/Llama/Mistral/Gemma-2/DeepSeek-V2 七个模型族。
> 每个功能都与 HF 参考逐 token 对齐、与真实 vLLM 同 workload 对比过，并诚实记录了哪些场景是亏的。

### 2 分钟
> 主线：**调度 → 内存 → 内核 → 量化 → 投机 → 系统 → 多模型**，每段 2-3 句 + 一个数字。
> - **调度**：先做"先全 prefill 再 decode"，实测发现早完成者死等 → 改成 vLLM V1 同款混合批次
>   （prefill 行在前、decode 行在后共享 token 预算），吞吐全档 +7~21%、抢占减半。
> - **内存**：paged KV cache（块 256）+ 链式哈希前缀缓存 + COW 安全写共享块 + 分块 prefill；
>   KV swap 抢占用 CPU 缓冲换出 decode 序列，**bit-exact 0 误差**——但诚实结论是本机
>   （0.6B+WSL2）swap 比重算慢（27.8s vs 11.8s），价值在 7B+ 与真实 Linux。
> - **内核**：flash-attn 的 fp8 KV 路径在 sm_120（Blackwell 消费卡）不可用（FA3 是 Hopper-only），
>   自己写 Triton 内核（decode + varlen 两套），fp8 KV 容量 1.9×、KL 0.0073；这是自研内核里
>   最能打的点——"vLLM 在这张卡上跑不了的东西我能跑"。
> - **量化**：五条路径里 **fp8 权重是唯一大 M 不输 cuBLAS 的**（decode 权重-only Triton、
>   prefill 硬件 `_scaled_mm`），ppl 与 fp16 同批误差 ~0.6%、bs=256 5825 tok/s 是全模式峰值；
>   int4 靠**双路径路由**反超 fp16（小 batch +35%）；2:4 稀疏的结论是"内核 bit-exact 但一次性
>   剪枝丢 35% 权重质量"——技术可行性评估型交付。
> - **投机**：n-gram/Medusa/EAGLE 三种都做通。核心理解是成本模型 γ·T_draft+T_verify vs
>   收益 (1+αγ)，而上界 α ≤ 模型 top-1 可预测性（实测自由文本 ~35%）——EAGLE γ=2 在重复
>   内容 +3.26×，free 文本只有 +0.70×，数字与理论互相印证。
> - **系统**：16GB 卡跑 7B+ 靠按层流式加载（meta 构造 → 逐层物化 → 加载即量化，峰值 10.7~11.6GB）；
>   meta 物化踩了四个坑（to_empty 丢 weight_loader、RoPE 缓存全零、tie 重绑、meta 计数污染
>   get_rope 共享实例）。
> - **多模型**：注册表按 model_type 分发；Gemma-2 有"读源码才能发现"的三个细节（embed ×√d、
>   RMSNorm (1+w)、双残差四 norm），逐层二分定位；阶段 2 打通 DeepSeek-V2 的 MLA + MoE、
>   Mistral/gemma2-local 的滚动环与真实模型长解码。

### 5 分钟
> 2 分钟版本 + 三个"证伪"故事 + 两个诚实结论：
> - **证伪 1（大 tile 假设）**：int4 内核最初按"大 tile 更优"直觉调优，实测小 M 大 N 的
>   权重带宽主导形态才赢（4.36×），大 M 全输 → 做**双路径路由**而不是硬碰 cuBLAS。
> - **证伪 2（分页瓶颈假设）**：假设 paged attention 的 gather 是瓶颈，实测 decode 瓶颈是
>   权重带宽而非 KV 索引——这决定了 fp8 权重的方向（0.5× 字节）。
> - **证伪 3（启动税假设）**：投机 verify 步慢，先怀疑内核，逐层 hook 计时发现是 CPU 启动
>   税 ~10ms/步 → CUDA graph 固定容量重放，spec 从打平变赢（bs=8 repeat +3.87×）。
> - **诚实结论 1**：KV swap 本机比 recompute 慢（27.8s vs 11.8s）——机制正确但不划算。
> - **诚实结论 2**：2:4 稀疏、纯 int4 大 batch、EAGLE γ=4、medusa 大 batch 都是"内核/机制正确
>   但整体亏"——知道边界在哪，比什么都做更能体现对推理系统的理解。

---

## 3. 主线叙事（按阶段，每阶段：一句话贡献 + 关键数字 + 为什么值得讲）

| # | 阶段 | 一句话贡献 | 关键数字 | 为什么值得讲 |
|---|---|---|---|---|
| 1 | vLLM 对比基准工程 | 先把"测量"做对：逐请求时间戳、同 workload 同 seed、两侧同 flash-attn | 吞吐 1.35-1.61× 领先；**decode 单步与 vLLM 持平（kernel 级可比）** | 口径诚实：离线 API 不暴露逐请求指标时用聚合直方图并声明近似 |
| 2 | FP8 KV cache + 自研 decode 内核 | FA3 是 Hopper-only，sm_120 只能自研 | 容量 1.9×；KL 0.0073；长上下文 decode 单步 32.2ms = vLLM fp16 | **"vLLM 在这张卡跑不了，我能跑"** |
| 3 | W8A8（per-group + SmoothQuant） | int8 权重 + int8 激活 + 平滑折权重 | KL 0.0379（平滑后）；吞吐 -16% | 量化精度方法论：per-group 比 per-channel 细 8× |
| 4 | 混合调度（V1 同款） | 消除"先全 prefill 后 decode"的死等 | 吞吐 +7~21%；抢占 85→68 / 141→71 | 调度器设计的核心权衡 |
| 5 | 投机解码框架（ngram→Medusa→EAGLE） | verify 步 = 带前缀复用的 varlen prefill + CUDA graph | 启动税 ~10ms/步被消除；EAGLE γ=2 repeat +3.26× | 成本模型 + α 上界 = 投机解码的完整理解 |
| 6 | fp8 varlen 内核 | verify 步直接读 fp8 缓存（免逐层反量化） | fp8+spec 从 0.15× 变 +3.92×（bs=8 repeat） | 内核与调度配合消除显存搬运 |
| 7 | INT4/AWQ 双路径 | 按形态路由：小 M 大 N 走 int4 内核，其余走 w_deq 稠密 | bs=8 +35%、bs=256 +3~6%；awq 把 int4 ppl 差距砍半以上 | "带宽优化型内核在计算主导区间的天花板"的正面解法 |
| 8 | 2:4 稀疏（可行性评估） | 内核 bit-exact；一次性剪枝是精度灾难 | KL 8.5；cuSPARSELt sm_120 每调用 0.3-0.5ms | 技术评估类交付：知道"为什么不做" |
| 9 | 多模型：注册表 + 流式加载 + Qwen2.5/Llama | 16GB 卡跑 7B+ 的唯一路径 | Qwen2.5-7B 峰值 10.66GB；Llama-3.1-8B 11.62GB；parity top-1 100% | meta 物化四坑（to_empty/weight_loader/RoPE/tie） |
| 10 | FP8 权重 | 唯一大 M 不输 cuBLAS 的量化 | ppl 与 fp16 同批 ~0.6% 差；bs=256 5825 tok/s（1.22×）；8B prefill TTFT 210ms vs int4 813ms | 双路径（Triton 内核 + 硬件 _scaled_mm） |
| 11 | EAGLE-1 | 无 RoPE 草稿层 + 共享 LM head 自回归 | γ=2 repeat +3.26×（α 0.525）；γ=4 只有 +0.63× | γ 是成本模型的关键变量 |
| 12 | KV swap 抢占 | decode 序列 KV 换 CPU，恢复 bit-exact 免重算 | bit-exact 0 误差；699 次换出；**本机比重算慢（27.8 vs 11.8s）** | 机制正确 + 条件诚实的样板 |
| 13 | Mistral（SWA）+ Gemma-2（soft-cap） | 两个新机制家族：滑动窗口 + logit soft-cap | parity top-1 100%（0.014/0.022）；Mistral 311 tok/s（bs=32）、Gemma2 1095 tok/s（bs=64，均为 08-24 首测） | Gemma-2 三个隐藏架构细节的定位方法论 |
| 14 | 工程素养：脚本 argv 化 + 验证工具箱 | 全部脚本模型路径参数化；pytest **57** 例 + parity/内核对照/ppl 回归矩阵 | 新模型复用全部基准零改动 | "测量与回归"是可信度的基础设施 |
| 15 | MoE 支持（1.5） | router top-k + 循环专家 + 量化专家 + Qwen3-MoE/DeepSeek-V2 端口 | 引擎 parity top-1 100%（mean diff 0.003）；decode int4/fp8 ≈2×；grouped 后端组织税 3.8→0.7ms/层 | 反直觉实测（负载不均偏好集中 / toy 顶层贴边假警报 / 段式内核赢面很窄） |
| 16 | MLA + DeepSeek-V2（2a） | latent 压缩 + decoupled rope + 吸收式 decode 内核 | fused 576 元素/token/层（**7.11×** 压缩，修正口径）；decode 内核 vs 稠密参考位级 0 误差；引擎 parity top-1 100% | DeepSeek 面试主线；真实模型验证（V2-Lite 4 层切片 + 全量流式 int4） |
| 17 | SWA 滚动环（2b） | per-seq 物理环：驱逐到窗口 + B 余量，decode KV 有界 | 稳态 KV = 窗口+B/seq（vLLM 掩码式做不到）；toy 84 采样步 vs 掩码参考 top-1 全一致 | 文档 TODO 变完成；自研内核处理环表位置偏移 |
| 18 | 组合解锁 + 真实模型验证（2b-ext） | fp8 KV×MLA / fp8×环 / spec×环 / gemma2 非统一窗口双池 / 纯 int4 MLA 免兜底 | 真实 Mistral-7B 5050-token 环 vs 掩码 **逐位一致**；真实 gemma-2-2b-it split 30/30 稀疏步；V2-Lite 4L parity 0 失配 | 全链路真模型证据 + int4 组 64/尾 K 掩码等真实尺寸 bug（§8 阶段 2b-ext） |

## 4. 深水区问答（八大模块，每个：机制 + 必问必答 + 数字 + 诚实结论 + 追问应对）

### 4.1 调度与批处理

**机制**：`Scheduler.schedule()` 返回 `(seqs, kind)`，kind ∈ prefill/decode/**mixed**/spec。
三队列 WAITING/RUNNING/FINISHED；混合批次 = prefill 行在前、decode 行在后，共享
`max_num_batched_tokens` 预算（vLLM V1 同款）；分块 prefill 只允许第一个被调度序列拆分；
KV 块不足时抢占（decode/spec 序列优先 swap_out / 其余 recompute 回 waiting）；滚动/MLA
模型的调度断言见 §8 阶段 2。在线服务持续将到达请求送入 engine；调度器可选 Top-W 前缀亲和、aging
公平性、recompute-aware 抢占，以及按请求 TTFT slack、队列长度和实测 prefill 速度调整的配额。

**必问必答**：
- **Q：混合批次为什么赢？** A：早完成 prefill 的请求立即 decode，消除死等；decode 提前释放
  KV 块、降低抢占压力。实测吞吐 +7~21%、抢占减半（512 档 141→71）。但 TPOT 改善有限——
  它受总工作量下界约束，收益在流式延迟与资源利用率。
- **Q：分块 prefill 为什么只拆第一个序列？** A：多序列同时分块会让每步 token 预算碎片化且
  块表管理复杂化；只拆第一个等价于"按到达顺序把预算给第一个长序列"，实现简单且覆盖主要场景。
- **Q：抢占后怎么恢复？** A：recompute = 回 waiting + 释放块，恢复时按前缀缓存哈希命中部分免算；
  swap = KV 拷 CPU、释放块、进独立 swapped 队列，恢复时直接 decode（免 prefill）。草稿作废。
- **Q：MoE/MLA/滚动模型对调度有什么影响？** A：MoE 动态路由形状不能入 CUDA graph → eager；
  MLA kv_b 的 int4/fp8 路径保留小份 `w_deq` 视图，维持吸收式 decode；没有 float 视图的量化兜底
  （如 w8a8/sparse24）会触发 MLA 稠密路径/eager。滚动模型停用前缀缓存发布/消费（内容过期，refcount 守卫断言）。

**数字**：混合调度吞吐全档 +7.2%~+21.3%；抢占 384 档 31→20、512 档 141→71；256 档峰值
5840 tok/s（fp16，§10.3.2，早期基准）。后续加入的在线调度策略有 `benchmarks/scheduling_ablation.py`
做同 trace 消融。**诚实边界**：TTFT 按 deadline slack 排序；TPOT 按已观测 token 间隔 EWMA 排序，并在压力升高时压低 prefill 配额，但只是软目标启发式。Multi-step decode 默认最多连续 4 轮，只在纯非投机 decode 窗口运行，`decode_burst_yield`（默认开）在 burst 期间有新 prefill 到达时提前收尾；该窗口内 SSE token 会成组交付。服务准入用 token 数（**已扣除前缀命中预测**）、在线吞吐和 KV/队列压力估算后接收或 FIFO 延迟，达到等待预算或硬上限时返回 429；2026-10-06 实测：命中估算与引擎实际复用逐请求一致，但默认准入阈值在 8 req/s 下过度拒绝、净损 31~36% 吞吐（§10.3.9）。**追问应对**：用相同到达 trace 对比 baseline、all-on 和各单项移除结果，并检查 TPOT 目标达成率、TTFT、吞吐、burst forward 数以及 burst 提前让出次数。

### 4.2 KV cache 与内存管理（paged / 前缀缓存 / COW / 分块 prefill / KV swap / 滚动环）

**机制**：BlockManager 管理固定块池 + `hash_to_block_id`；块哈希链式（xxhash(token_ids) +
前块哈希 8B LE）；**部分块也缓存**（末块按实际 token 数记账）；写共享块前 COW（GPU 整块 K/V
拷贝，`cow_pairs` 在 run 前执行）；哈希条目删除带守卫（双胞胎块共享哈希）；KV swap 用独立
swapped 队列 + CPU 非 pinned 缓冲 + `kv_swap_space_gb` 预算；**滚动环**（`rolling_cache`）：
块表 = 窗口内容清单（`Sequence.kv_j0` 行首逻辑块序号），驱逐 `(front+1)·B ≤ N−W−slack`
先释放再分配（净零 free 消耗），环模型块恒私有（不发布/消费前缀缓存）→ refcount 守卫断言。
KV swap 支持 TP=1 的 MHA/MLA 缓存，KV dtype 为模型原生 `auto` 或 FP8 E4M3；FP8 在 CPU 侧用
`uint8` 保存原始字节。拷贝是同步普通主机内存传输，且受 `kv_swap_space_gb` 限制；TP>1 回退 recompute。

**必问必答**：
- **Q：paged attention 与 vLLM 的差异？** A：块大小 256 vs vLLM 16（内部碎片与哈希粒度不同）；
  我们的 COW 在调度器记账、引擎 step 执行；vLLM 的 block manager 更细（BlockAllocator 分
  CPU/GPU、块级记账）。
- **Q：COW 的安全性怎么保证？** A：写起点落在共享块（ref_count>1）时复制一块并换表；复制对
  在 GPU 上执行；哈希条目删除带守卫防误删他人条目。
- **Q：KV swap 为什么 bit-exact？** A：KV 内容原样拷 CPU、换入 `index_copy_` 原位写回新私有块
  （不查前缀缓存），恢复后采样流确定。**坑**：高级索引 `kv_cache[:, :, ids]` 返回副本，`.copy_`
  只改副本——必须 `index_copy_`（§6 故事 14）。
- **Q：swap vs recompute 怎么选？** A：成本模型：swap = D2H+H2D 拷贝带宽；recompute = 重新
  prefill 计算量。7B+ 重算贵 → swap 赢；0.6B 重算便宜 + WSL2 D2H 慢 → recompute 赢（实测
  27.8s vs 11.8s）。
- **Q：滚动环和掩码式 SWA 差在哪？** A：掩码式（vLLM 同款）KV 随生成长度线性增长，只限注意力；
  滚动环让**稳态 KV = 窗口 + B 余量**，但 flash-attn 无法表达环表的 key 位置偏移（它从表下标推
  位置）→ 需要自研 paged decode 内核 + varlen 环装配；滚动模型不参与前缀缓存（重复 prompt 有
  代价，见 config 注释与 §8 阶段 2）。

**数字**：前缀缓存跨批次 prefill 降为 0 token/0 步；FP8 KV 容量 1.9×（421→802 块）；旧版 KV swap
测量 bit-exact 0 误差、96×512 压力 699 次换出、27.8s vs recompute 11.8s；滚动环真实 Mistral-7B
5050-token（>W=4096）环 vs 掩码 fp8 KV **全程逐位一致**，真实 gemma-2-2b-it split 环池表长
到 cap(18) 封顶而 full/掩码继续线性增长（§8 阶段 2）。
**诚实结论**：Mistral 统一窗口 rolling decode 可用 CUDA Graph；rolling+ngram verify 支持但走 eager；
Gemma-2 split 双池仅 auto KV、无投机、无 KV swap，decode eager；KV swap × rolling/split 尚未验证。
**追问应对**：被问"前缀缓存怎么失效"→ 内容哈希链式、被拒草稿永不进
哈希（投机）、COW 副本重新发布哈希、滚动模型整体退出。

### 4.3 CUDA 内核与性能工程（Triton 内核 + CUDA graph + roofline 归因）

**机制**：自研内核按形态分两类——**计算型 GEMM**（int8/int4/fp8/sparse24，M-adaptive tile：
小 M 用 16×128 权重带宽主导、int4 大 M 用 BM16/BN128）与**访存型注意力**（fp8 KV decode/varlen
内核，GQA 融合 + MMA + 寄存器内反量化；MLA 吸收式 decode 内核与稠密装配）。CUDA graph：decode
按 batch 容量族捕获共享内存池；spec 步按"行容量 × 双 stride"捕获、零长度填充行重放（bit-exact）。
**必问必答**：
- **Q：int4 内核为什么小 M 赢、大 M 输？** A：小 M（decode）权重带宽主导——int4 权重字节 =
  bf16 的 1/4 → 实测 gate_up M=8 4.36×；大 M（prefill）计算主导，MMA 数与稠密相同 + 反量化
  开销 → 0.2-0.6×。**结论：软件低比特 GEMM 是带宽优化，不是计算优化**——双路径路由的动机。
- **Q：roofline 归因怎么做？** A：实测标定上界（TC 48.5 TFLOPS、D2D 370 GB/s），按 arithmetic
  intensity 分类：AI≥128 算力受限、<128 带宽受限，另有"启动/并行度受限"（小 M）。归因结果：
  fp8 大 M 到 81% TC、int4 大 M 只有 44%、decode 小 M 启动受限。
- **Q：手写 MatMul 到什么水平？** A：SMEM-tiled fp16（Triton）达 **101% cuBLAS**（4096³/16384³）；
  CUDA C 版本（`_cuda_gemm_report.md`，nvcc 12.8）到 cuBLAS 53%（8 tile/warp + BsT 布局、bank
  conflict 消融 +86%、SASS 实证）。消融：GROUP_M swizzle +4%、stages=2 最优；**最优配置 occupancy
  只有 17%**——大 GEMM 是 TC 吞吐型，occupancy 不是瓶颈（反直觉反例）。
- **Q：tile 搜索找到什么？** A：**int4 大 M 用 BM16/BN128（regs=128）反超 BM64/BN256（regs=255）
  19%**——大 tile 的 acc 累加器把寄存器打到 255 上限、占用 1 block/SM；落地后纯 int4 大 batch
  +28%（0.64×→0.82× fp16）。教训：快速搜索的 "+144%" 异常值被高迭代复测推翻。
- **Q：自研 fp8 注意力内核的关键决策？** A：①GQA 融合（一 program 处理 seq×kv_head 整组 q 头，
  KV 只读一次）；②直接 fp8 load + 硬件 cvt 反量化（无 LUT）；③MMA 计算（QPAD=16 满足 dot 的
  N≥16，8× 计算浪费换内存效率——decode 是 memory-bound）；④BLOCK_T=32/warps=1（跨 warp 归约
  顺序变化放大误差）。归因：有效 KV 读带宽 517-529 GB/s（超过 copy 370 的双向口径）。
- **Q：CUDA graph 的坑？** A：形状必须静态（按容量族）；共享内存池避免碎片；spec 图用尾部重复
  cu_seqlens 的空行填充（bit-exact 用 probe 验证过）；只有特定动态图路径需 eager：MLA 的稠密兜底、
  Gemma-2 split 双池；Mistral rolling decode 和覆盖容量族的纯 spec（含 FP8 KV）有 graph 路径，
  mixed/超容量 spec 会回退 eager。
- **Q：MLA decode 内核的形态？** A：吸收式：W_UK 折进 q（q_abs）、W_UV 折进输出——每 token
  只读 fused 576 元素而非重建稠密 K/V；kv_b 对 int4/fp8 特意保留小型 `w_deq` 视图（约占参数
  1%），因此纯 int4/fp8 可继续走吸收式路径；w8a8/sparse24 等无 float 视图的路径仍回退稠密兜底并强制 eager。

**数字**：硬件锚点 TC 48.5 TFLOPS / 带宽 370 GB/s / 36 SM / SMEM 100KB；手写 MatMul 101%
cuBLAS；int4 gate_up M=8 4.36×、lm_head 3.42×、down_proj 0.40×；fp8 权重 K=4096 M=8 4.22×、
M=256 scaled_mm 1.86×；fp8 decode 内核逐层 ~1.15ms；verify 启动税 ~10ms/步被 graph 消除。
**追问应对**：被问"还能怎么快"→ cp.async/TMA 双缓冲流水（主差距）、ldmatrix canonical、KV 块
排序提 L2 命中、w_deq 降精度存储（fp8 引入 dequant 流量不划算）；persistent/split-K 已实测
（persistent 亏、split-K 仅小 M 赢——给条件不给背书）。

### 4.4 量化（路径取舍 + 精度方法论 + 诚实数字）

**机制**：w8a8（per-group 128 int8 + per-token int8 + SmoothQuant）、int4（per-group 对称 +
2-dot 反量化内核 + 双路径 + `int4_group_size` 可配）、AWQ（α 搜索缩放折叠）、fp8 权重（e4m3
全量化：per-column 权重 + per-token 激活，decode 权重-only Triton / prefill 硬件 `_scaled_mm`）、
sparse24（2:4 幅值剪枝）、fp8 KV cache（另线）。**量化精度口径（2026-09-07 复测，真实文本
ppl，12 条模型自生成续写 3060 token）**：fp16 3.23 / fp8 3.25（+0.6%）/ int4(RTN) 4.17（+29%）/
awq 3.70（+15%，把 int4 差距砍半）——**ppl 语料每次运行重新采样，run 间有波动**（历史 run 曾
报 fp16 3.32/int4 4.38/awq 3.76 与 fp16 3.60/int4 4.81/awq 4.22），只比同次 run 内；引擎级
对齐指标（KL/top-1）不受此影响。

**必问必答**：
- **Q：AWQ 为什么有效？** A：大激活通道的权重误差贡献大——把 s 折进权重（W'=W·s）让大激活
  通道的量化相对误差变小，激活侧除 s（X'=X/s）压小误差贡献。**方向必须对**（权重乘、激活除）；
  方向错了 α 搜索会假装"不缩放最优"（§6 故事 7）。实测把 int4 相对 fp16 的 ppl 差距砍半。
- **Q：fp8 为什么近乎无损？** A：e4m3 的 3 位尾数 + per-column scale + per-token 动态激活 scale；
  同批 ppl +0.6%；引擎级 KL 0.017、top-1 100%（fp8 权重）。
- **Q：2:4 稀疏为什么失败？** A：内核 bit-exact，但**一次性幅值剪枝丢 ~35% 权重质量**（KL 8.5、
  top-1 0%）——SparseGPT 式误差补偿或剪枝感知训练是修复路线；且 sm_120 上 cuSPARSELt 每调用
  0.3-0.5ms、CUTLASS 仅 sm_8x。
- **Q：量化精度怎么测才可信？** A：8-prompt KL 会被尾部单点主导（曾与 ppl 结论相反）→ 换真实
  文本困惑度；**指标的样本量决定结论方向**。ppl 语料随机 → 跨 run 比较只比同批。
- **Q：fp8 KV 的写路径有什么坑？** A：torch 的 fp32→fp8 cast **溢出不饱和而是产生 NaN 位模式**
  （实测 500→0x7F）→ 写路径必须 clamp(-448,448)；位模式解码（LUT）与硬件 cvt 在 NaN 语义上
  不等价（§6 故事 1）。

**数字**：fp8 KV KL 0.0073 / top-1 100% / 容量 1.9×；w8a8 KL 0.0379 / 吞吐 -16%；int4 双路径
bs=8 +35%、bs=256 +3~6%、显存 1.73GB（比 fp16 还大 15%，w_deq 定价）；纯 int4 0.85GB、
bs=256 0.82× fp16；AWQ 112/112 层逐层误差全赢；Qwen2.5-0.5B fp16 5.12 / int4 7.45（0.5B 量化
鲁棒性弱）。
**诚实结论**：int4 双路径的 w_deq 让显存超 fp16——吞吐无损的定价；纯 int4 仍是显存优先。
**追问应对**：被问"为什么不用 GPTQ"→ GPTQ 用 Hessian 逆做逐列误差补偿，理论上优于 RTN；我们
没实现——已知差距（§8 阶段 4）。

### 4.5 投机解码（n-gram / Medusa / EAGLE / fp8 verify / 环上 verify）

**机制**：verify 步 = 带前缀复用的 varlen prefill（query = [末 token, 草稿...]，位置从 len-1 起，
num_cached = len-1）；接受规则（Leviathan et al.）：草稿是点质量分布，"接受 iff 目标采样==草稿"
严格保持分布；被拒草稿不回滚 KV、永不进前缀缓存哈希（hash 范围只到接受长度）；verify 步 CUDA
graph 化（行容量族 + 双 stride，stride = γ_max+1 与 3）。三个草稿源：n-gram（历史窗口搜索，零
成本）、Medusa（γ+1 个 MLP 头 1024→256→vocab）、EAGLE（无 RoPE 草稿层 + 共享 LM head 自回归）。

**必问必答**：
- **Q：成本模型？** A：期望加速 = (1+αγ)/(γ·T_draft+T_verify+1)，α = 草稿接受率；**上界 α ≤
  模型 top-1 可预测性**（实测自由文本 ~35%）——这解释了为什么所有方案在 free 文本 ~1.5-2×、
  重复内容 3-4×。
- **Q：为什么 EAGLE γ=2 赢、γ=4 输？** A：γ=4 每草稿一次 LM head 前向（0.6B 上 ~0.8ms）+ 特征
  误差累积（草稿质量随深度下降）；γ=2 的 (1+αγ) 收益 > 成本。实测 γ=2 repeat +3.26×（α 0.525）、
  γ=4 只有 +0.63×。
- **Q：verify 步为什么用 varlen prefill 而不是 decode？** A：多草稿 = "一行多个 query token"——
  本质是变长小 prefill；复用分块 prefill + 前缀复用路径（缓存形状 K/V + block tables）。
- **Q：fp8 KV + 投机怎么结合？** A：verify 步走自研 fp8 varlen 内核直接读 fp8 缓存（免逐层全缓存
  反量化，~18GB/步 搬运）——从 0.15× 变 +3.92×（bs=8 repeat）。逐列因果掩码必须 `<=`（差 1 会
  让真实数据 logits 差 9-25，§6 故事 3）。
- **Q：滚动环上怎么 verify？** A：verify 行的 key 集 = [kv_j0·B, end)（环驻留内容 + 本步刚写行）
  稠密装配成段喂 flash——**段内相对下标使窗口掩码精确**（origin 常数抵消）；BlockManager 的 spec
  账本按逻辑块 j0+i 记账；slack = γ+2（§8 阶段 2b-ext）。
- **Q：为什么 α 低时反而亏？** A：每步固定付 γ+1 行 verify 成本，产出只有 1+αγ；α≈0.05-0.25
  （Medusa 自由文本）时 1.2-2.0 token/行 盖不住大 batch 的 GPU 行成本。

**数字**：ngram bs=8 repeat +3.87×、bs=256 +1.43×；Medusa bs=8 repeat +1.57×（head_0 达模型
top-1 的 87%，vs 真实 next 30.9% = 上限 35.3% 的 87%）；EAGLE γ=2 repeat +3.26×；free 文本全部
~1.5-2×（α 0.19-0.23）；0.6B top-1 可预测性 ~35%（temp=0.6 采样只有 ~20-30% 概率等于 argmax）。
**诚实结论**：α 被模型可预测性封顶——投机在"模型太笨"时赚不到；工程侧（verify 路径、graph、
fp8、环）已闭环，剩余瓶颈是草稿质量。**追问应对**：被问"怎么提草稿质量"→ 更大模型（7B+ top-1
更高）、更久训练（原文百万级 vs 49K）、medusa_hidden 256→512、tree attention（Medusa-2）——
但可预测性天花板不随这些改变。

### 4.6 系统与工程（流式加载 / 多模型 / TP / 验证方法论）

**机制**：按层流式加载（meta 构造 → 逐层 `to_empty` 物化 → 加载即量化 → 释放 fp16；自动触发 =
fp16 估重 > 空闲显存 45% 且启用量化，估算用 **meta 实建数参数**——通用 qkv/o/inter 公式不含
MoE/MLA 专家权重，会把 DeepSeek 33GB 误判成小模型直接建而 OOM）；注册表按 model_type 分发；
TP 用 NCCL + 共享内存命令通道（weight_loader 分片 + all_reduce）；验证方法论 = HF 参考 logits
对照（top-1 100%）+ 内核独立对照 + 引擎级 smoke + 真实文本 ppl。

**必问必答**：
- **Q：meta 物化踩了什么坑？** A：①torch 2.8 禁止 meta→真实设备 `.to()`/`set_data`，必须
  `to_empty`；②to_empty 替换 Parameter 丢 weight_loader → 按模块重挂；③RoPE `cos_sin_cache`
  是计算型 buffer，物化后全零 → q/k 被零旋转逐层发散 → `_finalize_streaming` 重建；④tie 词表
  重绑；⑤meta 计数构造污染 `get_rope` 共享实例 → `cache_clear()`（§6 故事 18，2026-09 修复）。
- **Q：16GB 卡怎么跑 7B+？** A：按层加载 + 即时量化——任一时刻 ≈ 累计量化权重 + 单层 fp16 +
  embed；7B int4 峰值 10.7-11.6GB。**诚实边界**：streaming 限制——int4 强制纯 int4（MLA kv_b
  例外保留 w_deq，占参 ~1%）、w8a8 无 SmoothQuant、awq 仅预生成 scales。
- **Q：新架构端口（Gemma-2）怎么验证？** A：HF parity 逐层二分（embed → 层0 → 层1 → hidden）；
  发现三个隐藏细节（embed ×√d、RMSNorm (1+w)、双残差四 norm），都是"checkpoint 里看不出来、
  读源码才能发现"的初始化语义。**教训：debug 脚本自身的形状/口径也要先钉对**（§6 故事 15）。
- **Q：transformers 5.15 的坑？** A：DeepseekV2 无缓存前向不传因果掩码（对照只能取末行/显式掩码）；
  experts 内存 3D 存盘 2D（`use_experts_implementation`）；**MoE grouped 路径的 `torch._grouped_mm`
  仅 sm_90**（CC 9.0）→ sm_120 无法 HF-GPU 直连做真权重对照（记录在案，用 nano fp16 稠密参考）。
- **Q：TP 为什么没实测多卡？** A：当前开发环境只有单卡；TP 路径包含 weight_loader 分片、NCCL
  命令通道和跨进程序列传递，但本机没有 multi-GPU 性能/稳定性数据。独立的 PD 路径已经实现为实验性
  模式，仍只支持兼容配置下的本地双卡同步 KV 交接。

**数字**：Qwen2.5-7B int4 峰值 10.66GB / Llama-3.1-8B 11.62GB / Mistral-7B 11.37GB；parity：
qwen2.5-0.5B top-1 100%（mean 0.096）、mistral 100%（0.014）、gemma2 100%（0.022）；DeepSeek-V2
4 层真权重切片引擎 vs 稠密参考 0 失配（§8 阶段 2）。**诚实结论**：TP 有代码但当前单卡环境未做
多卡验证；PP/DP/EP 模型并行未实现；CacheBlend 未实现。PD 已有本地实验性路径，但限 TP=1、双卡、
KV dtype=auto、无 speculative/rolling cache，且主机内存 KV 交接与两阶段执行均为同步串行。

### 4.7 MoE（router + 循环专家 + grouped 后端 + 量化专家）——阶段 1.5

**机制**：MoE 只替换 FFN——`mlp.gate`（router，fp32 softmax → top-k，可选 norm_topk_prob）+
`mlp.experts.{i}.gate_proj/up_proj/down_proj`（2D per-expert，参数名与 HF checkpoint 直配 →
loader 零改动、量化路径继承）；前向 = 逐专家循环（串行、e 升序、x dtype 累加，与 transformers
eager 同语义）；router 永不量化（gate 精度决定路由）。**grouped 后端（1.5b）**：排序分段 +
padded 批量 bmm（gate_up 3D 融合 + silu·up 融合 + down）→ 组织税 3.8→0.7ms/层；条件 = 专家有
float 权重（未量化或 int4 dual 的 w_deq），纯 int4/fp8 回退循环。**段式内核（1.5c）**：Triton
真段式（无 padding，位级正确）——实测赢面窄：小 E + 极长段才赢（E=8/R=32K 快 43%），默认仍是
padded bmm。

**必问必答**：
- **Q：MoE 影响引擎哪些部分？** A：只 FFN；调度/KV/注意力/CUDA graph 结构全复用；但动态 gather
  形状不能进 CUDA graph → MoE 模型 eager（graph 化需路由 padding，未做）。
- **Q：怎么验证 MoE 正确性？** A：三层：①数学同构参考（全行掩码 vs gather/index_add，同序累加）
  → 位级一致；②CPU 单测进 pytest；③端到端：随机 toy（4 层混合 + tie）引擎 vs transformers 5.15
  同权重同 dtype → top-1 100%、mean diff 0.003。
- **Q：负载不均怎么办？** A：串行循环实现免疫且偏好集中（强制全 token 进一专家反而快 3.6-5×）；
  padded 批量在不均衡时放大到 E·max_n（本尺度实测仅 +30%）；真段式内核（无 padding）是 128+
  专家的下一步。
- **Q：MoE 量化值不值？** A：decode 专家 GEMM 权重带宽受限 → int4/fp8 ≈2×（方向稳健、绝对值受
  WSL 时钟噪声影响）；层误差 fp8 6.4% / int4 12.5%（随机 toy）。
- **Q：DeepSeek-V2 的 MoE 形态？** A：shared experts（always-on）+ routed scaling（α 缩放）；
  transformer 5.15 存盘 2D、内存 3D；V2-Lite 真实权重：27 层、n_routed_experts 64、topk 6、
  intermediate 10944（=64×171，int4 组 64，§8 阶段 2b-ext）。

**数字**：parity top-1 100%（mean 0.003）；decode int4/fp8 ≈2×；grouped 后端组织税 3.8→0.7ms/层
（-80%），小 T 快 4-5×、T=4096 1.9×；int4 dual 自动走 grouped（w_deq）→ toy decode 2588 tok/s
（~2.4-2.8× 观察）。**诚实边界**：量化纯 int4/fp8 无 float 视图仍回退循环；padded 后端在不均衡/
大 E 时放大到 E·max_n。**追问应对**：被问"怎么快"→ grouped GEMM、路由 padding 入图、shared
expert（Qwen3-235B 类，5.15 已删该结构）、EP + all-to-all 通信模型（§8 阶段 5）。

### 4.8 MLA 与滚动环（阶段 2 主线，DeepSeek 面试必考）

**机制（MLA，2a）**：DeepSeek-V2 latent 注意力——Q 也压缩（q_lora_rank）；K/V 共享一个 latent
`c_kv`（kv_lora_rank）+ 解耦 rope key `k̃_pe`；W_UK 与 W_DKVᵀ 权重绑定（KV 无独立投影矩阵）；
每 token 每层缓存 fused `[c_kv | k̃_pe]`（V2-Lite：512+64=576 元素，**7.11×** 压缩——修正口径：
论文共享 rope key 全头共用，非"每头 rope key"的 naive 1536/2.7×）。decode 用**吸收式内核**：
W_UK→q（q_abs）、W_UV→输出，每 token 只读 576 元素，重建不出稠密 K/V；varlen（prefill/verify）
走稠密装配喂 flash（v head_dim 零填充对齐）。**机制（环，2b）**：见 §4.2。

**必问必答**：
- **Q：MLA 为什么省？** A：KV 每 token 每层从 (2×n_kv_heads×head_dim) 稠密降到 latent+rope；
  V2-Lite 等效 GQA 口径 4096+ → 576（7.11×）；V3 官方口径 576 同理推导。
- **Q：吸收式内核怎么工作的？** A：q_abs = q_nope·W_UKᵀ（把 latent-key 投影吸进 query 侧），
  score = q_abs·c_kv + q_pe·k̃_pe；输出 o = Σ p·(c_kv·W_UV)（W_UV 是 down 投影，p·c_kv 先算再
  投回）。注意与 W_UK/rope 的缩放因子按论文对齐（absorb 前乘 head 缩放）。
- **Q：MLA 的 fp8 KV 怎么做？** A：fused 行 [c_kv|k̃_pe] 量级不同 → **两段独立 scale**（cal_c/
  cal_r，主机侧 clamp±448 → e4m3）；读路径（内核/稠密装配）按段反量化回 fp32×scale→fp16。
  验证：gather+dequant 内核逐位一致、toy 引擎 44 行 1 翻转（fp8 噪声带）。
- **Q：纯 int4/fp8 的 MLA decode 呢？** A：吸收式内核要求 kv_b 投影有 float 视图；纯 int4/fp8
  时 kv_b 保留反量化副本 w_deq（is_mla_kv_b 例外，占参 ~1%，V2-Lite 实测 113MB）→ 走稠密兜底
  的自动 eager 被消除、吸收式 decode + CUDA graph 恢复（2b-ext）。
- **Q：滚动环的正确性怎么保证？** A：驱逐保证 resident_start ≤ N−W−slack（verify 需要的
  γ+2 余量）；decode/varlen 内核用 chunk_starts/装配偏移表达 key_pos=(j0+b)·B+t；refcount
  守卫断言环块恒私有（无共享 → 无 COW）；与掩码式参考对齐验证（同内核位级、异内核 top-1）。

**数字**：MLA decode 内核 vs 稠密参考**位级 0 误差**（3 场景跨块）；CPU 全模型 vs transformers
5.15 同权重 top-1 100%（max 1e-6）；引擎 prefill/decode parity top-1 100%（mean 0.003-0.03）；
真实 DeepSeek-V2-Lite 27 层流式 int4：启动 55s、权重 ~5GB 常驻（int4 + kv_b w_deq 113MB）、
decode ~2.8 tok/s（3 并发、MoE eager、纯 int4）、中英连贯（§8 阶段 2b-ext 完整表）。
**诚实边界**：fp8 KV×MLA 的组合路径有 toy 覆盖，但真实 V2-Lite 尚未测 FP8 KV；ring+medusa/eagle
断言关；Gemma-2 split 仅 auto KV/无投机/无 swap/eager；KV swap×rolling 尚未验证；MLA spec
（投机 verify 的吸收式路径）未做。
**追问应对**：被问"V3/V3.1 的 MLA 变体"→ 官方 576 口径推导、多 token 预测（MTP）未实现、EP 专家
并行理论（§8 阶段 5）。

## 5. 数字速查（归档实测；条件：RTX 5060 Ti 16GB / WSL2 / bf16 / 单卡）

以下数值来自表中标明的旧 workload/日期，主要早于在线服务与新调度策略；它们仍可说明当时的内核和
引擎形态，但不能代表 2026-10 当前代码，也不能用于宣称新策略带来收益。新策略请使用 §10.2 的
`scheduling_ablation.py` 对相同请求 trace 做对照。

| 项 | 数字 | 条件 |
|---|---|---|
| 引擎吞吐峰值 | **5825 tok/s**（fp8 权重，1.22× fp16）；fp16 基线 4792 tok/s（同批） | Qwen3-0.6B，bs=256，干净 workload，2026-08 批次 |
| batch 缩放峰值 | **5840 tok/s**（fp16 @256） | 同 workload 另一批次（混合调度重测）；跨批只作量级参考 |
| vLLM 对比 | 吞吐 1.35-1.61× 领先；prefill ~2×；**decode 单步持平** | 同 workload 同 seed 同 flash-attn，2026-08-18 |
| 混合调度 | 吞吐 +7~21%；抢占减半（512 档 141→71） | 全 batch 档 |
| 前缀缓存 | 满块重复批次 prefill 0 tok/0 步；300-token 部分块场景 batch1=2816（COW 正确语义） | Qwen3-0.6B |
| FP8 KV | 容量 1.9×（421→802 块）；KL 0.0073；top-1 100% | fp8_e4m3 + 自研内核 |
| fp8 权重 | 同批 ppl 3.23→3.25（+0.6%）；KL 0.017 top-1 100%；bs=256 5825 tok/s；8B prefill TTFT 210 vs int4 813ms | e4m3 全量化；K=4096 微基准 M=8 4.22×、M=256 scaled_mm 1.86× |
| int4 双路径 | bs=8 +35%、bs=256 +3~6%；同批 ppl 4.17（RTN，+29%）；显存 1.73GB | w_deq 副本定价；qwen3-0.6B int4 dual 实测 1.730GB（2026-09-07） |
| int4 纯模式 | 显存 0.85GB；bs=256 3916.5 tok/s（0.82× fp16，tile 搜索后从 0.64× +28%） | 大 batch 仍慢于 fp16，显存优先 |
| AWQ | 同批 ppl 3.70（+15%，int4 差距砍半）；112 层 α 搜索逐层误差全赢 | 真实文本校准；历史 run：4.38→3.76 |
| ppl 口径 | 语料每次随机续写 → run 间波动（fp16 基线 3.2-3.6 均见过） | **只比同 run 内**（§10.3.6） |
| 7B ppl（Mistral-7B，2026-09-07） | fp16 4.05 / fp8 4.05（+0.2%）/ int4(RTN) 4.23（**+4.5%**）/ awq 4.15（+2.7%，砍 int4 差距 ~40%） | fp8 引擎语料 3060 token；**7B 量化鲁棒性 ≫ 0.6B**（0.6B int4 +29%） |
| W8A8 | KL 0.0379（per-group 后）；吞吐 -16% | per-group 128 + SmoothQuant |
| 2:4 稀疏 | 内核 bit-exact；一次性剪枝 KL 8.5；cuSPARSELt 0.02-0.17× | sm_120 无硬件稀疏 MMA |
| ngram spec | bs=8 repeat +3.87×；bs=256 +1.43×；fp8 版 +3.92× | verify CUDA graph 后 |
| Medusa | bs=8 repeat +1.57×；head_0 = 模型 top-1 的 87% | 自蒸馏 ~7min；大 batch 亏 |
| EAGLE-1 | γ=2 repeat +3.26×（α 0.525）；γ=4 只有 +0.63× | 0.6B，重复内容 |
| 模型可预测性 | 自由文本 top-1 ~35% → 投机 α 天花板 | 0.6B；7B+ 会右移 |
| KV swap | bit-exact 0 误差；699 次换出；**27.8s vs recompute 11.8s（本机亏）** | 0.6B+WSL2；价值在 7B+ 与真实 Linux |
| 流式加载峰值 | Qwen2.5-7B 10.66GB / Llama-3.1-8B 11.62GB / Mistral-7B 11.37GB | int4，16GB 卡 |
| 端口 parity | qwen2.5 100%（0.096）/ mistral 100%（0.014）/ gemma2 100%（0.022） | prefill logits vs HF |
| 模型吞吐 | Llama-3.1-8B 303.5 tok/s（bs=16）；Mistral-7B 311.4（bs=32，2026-08-24）；Gemma-2-2B 782.7-836.6（bs=64，2026-09-07 复测 ×2；08-24 首测 1094.7） | int4；WSL run 间波动 ±25% 级别 |
| MLA | fused [c_kv\|k̃_pe] **576 元素**/token/层（**7.11×**，修正口径）；decode 内核 vs 稠密**位级 0 误差**；引擎 parity top-1 100% | DeepSeek-V2；V2-Lite 27 层流式 int4 decode ~2.8 tok/s（3 并发、MoE eager） |
| 滚动环（2b） | 稳态 KV = 窗口+B/seq；真实 Mistral-7B 5050-token 环 vs 掩码 fp8 **逐位一致** | 环表 ≤ cap；滚动模型停用前缀缓存 |
| split 双池（2b-ext） | gemma-2-2b-it 真实 split：环池 cap 18 封顶 vs full/掩码线性增长；30/30 稀疏步 | 仅 bf16/无投机/eager；softcap 层禁 fp8 KV |
| 精度方法论 | 8-prompt KL 与 ppl 结论相反 → 用 3000+ token 困惑度；语料随机 → 只比同批 | 样本量决定结论方向 |

---

## 6. 踩坑故事精选（讲"方法论"而非"事故"；编号与 note.md §4 一一对应）

1. **torch fp8 cast 溢出静默变 NaN**：fp32→fp8 cast 溢出不饱和而是产生 NaN 位模式（实测
   500→0x7F）；v4 的 LUT 把 NaN 位模式读成 0.0 **掩盖**了它，v5/v6 换硬件 cvt 后 NaN 直接进
   logits——修复（clamp±448）后精度反而变好（KL 0.0077→0.0073）。教训：**位模式解码与硬件 cvt
   在 NaN 语义上不等价——换内核解码方式必须重跑引擎级精度检查**。
2. **Medusa 三个集成 bug**：训练标签错位 1（head_0 训成预测当前位置 → 永不接受）、postprocess
   清零 num_scheduled_tokens 后才读它（行索引全错）、capture_spec_graph 只对 ngram 调用——每个
   都让 α 归零，靠"单元验证头是好 + 引擎 draft 与模型 argmax 49% 重合"逐层剥离定位。
3. **fp8 varlen 差 1 掩码**：`tok_mask = key_pos < key_upper` 排除了 query 自己的 key——随机数据
   上影响 ~1/key_len 没被独立检查抓住，真实数据上 self-attention 分量大 → logits 差 9-25。
   **独立检查（随机数据）通过 ≠ 引擎正确——必须跑引擎级对齐检查**（掩码要 `<=`）。
4. **训练 60× 慢**：引擎占 ~14GB 时 caching allocator 每步 7.3s（vs 释放后 123ms）——先
   `llm.exit()` 再训练。
5. **Gumbel 噪声在 GPU**：`torch.manual_seed` 不控 GPU RNG（噪声在 GPU 生成）——同 seed 对比
   实验必须先种 `torch.cuda.manual_seed`；同进程多引擎必须 `empty_cache`。
6. **INT4 打包列缺块偏移**：`offs_j` 没加 `pid_n·(BLOCK_N//2)` → 所有 N 块≥1 的程序读通道 0-63。
   小形状检查全过、大 N 全错；按 N 块打印误差分布一眼定位（块 0 对、其余全错）。**小形状通过 ≠
   内核正确，要覆盖多块路径**。
7. **AWQ 方向三连错**：`W·s且X·s`（KL 8.6）→ 反方向 `W/s且X·s`（组塌缩 KL 12.4）→ 论文方向
   `W·s且X/s` 才对；方向错时 α 搜索会假装"不缩放最优"（α=0）——**必须独立验证恒等式再信搜索**。
8. **8-prompt KL 与 ppl 结论相反**：KL 对单个极端位置极敏感（seq3 单点 p=0.93 贡献 ~1.4），
   8 个 prompt 的对比被一两点主导 → 换 3000+ token 困惑度（决定性指标）。教训：**指标的样本量/
   敏感度决定结论方向**。
9. **meta 物化丢计算 buffer（RoPE 全零）**：按层流式加载时 `to_empty` 只给未初始化内存，meta
   设备上算出的 `cos_sin_cache` 值丢失 → q/k 被零旋转 → 从第 1 层起逐层发散。定位：逐层 hook
   对比——embed 完全一致、layer0 首 token（只 attend 自己）精确、q_pre/k_pre 一致但 q_post/
   k_post（RoPE 后）差 80-130 → 锁死 RoPE。此前"buffer 对比通过"是假象：`lru_cache(1)` 让同进程
   两个模型共享同一实例，对比平凡相等——**检查脚本必须清缓存再重建**。
10. **裸模型路径 .to() 打破 tie 共享（lm_head 全零）**：诊断脚本先 CPU 构造再 `.to(cuda)`——
    `_apply` 重设每个 Parameter 的 `.data`，`__init__` 的存储共享被拆开；tie checkpoint 常不含
    `lm_head.weight`（Qwen2.5-0.5B 没有，Qwen3-0.6B 有——所以 qwen3 一直侥幸没炸）→ logits 全零
    → **ppl = 词表大小（uniform）是强信号**。修复：加载后按 tie 重绑。共享存储的 Parameter 在
    `_apply`/`.to()` 下不保共享。
11. **Triton JIT 编译落进计时区间（TTFT 435ms 假象）**：融合激活量化 kernel 每个 BLOCK_K 变体
    首次调用编译 100-400ms；warmup 用 M=16384 而实测步 M=1042 → 编译成本吃进 TTFT（435 vs 真实
    22ms）。定位：逐层 hook 发现"只有 layer0 慢"→ 事件计时拆出 quant 289ms → 冷/热首调对比。
    **修复：bench 预热用真实 workload 形状**。否则会报告 20× 假数字。
12. **投机多 token 接受跳过 max_tokens（`_maybe_finish` 的 `==`）**：精确相等在多 token 步进下
    永不命中（62→65 跳过 64）→ 序列一路长到 max_model_len（EAGLE 实测 4093 → 块表溢出 spec
    graph 16 列 → 崩）。ngram 的草稿预算恰好避免，EAGLE 没加就暴露。**修复：改 `>=` + 草稿循环
    加 remaining-1 预算**。预存 bug 常由新路径触发。
13. **SDPA 3-D 输入把 head_dim 当序列维（EAGLE 注意力语义错）**：`[n, heads, hd]` 被按 [B,H,L]
    解释——因果/对角掩码全做在特征维上（数值稳定、训练照常收敛，但语义全错）。正确形式是
    [1,heads,n,hd] 4-D；且 1-token-per-step 推理下对角注意力退化为 o=v，训练/推理语义才一致。
    **API 的维度语义 ≠ 直觉——用之前先验证解释**。
14. **高级索引写回是临时副本（swap 静默写垃圾 KV）**：`kv_cache[:, :, block_ids]`（list 索引）
    返回 gather 副本，`.copy_()` 只改副本 → swap_in 写回后 KV 永远不变（输出全错但长度/无崩溃
    全过）。bitexact 测试（读-写-读回对比，Δ=300 vs 期望 0）才抓到。修复：`index_copy_` 原位写。
    **"跑通"不等于"写对"——写回缓存/参数必须验证内容**（torch 高级索引：读=copy、写=副本，
    单 int 索引才是 view）。
15. **Gemma-2 三个"读源码才能发现"的架构细节**：端口写完跑 HF parity，top-1 0%、logits 全不
    相关。逐层二分：①embed diff 恒为 ~48× → HF 源码 `Gemma2TextScaledWordEmbedding` = **embed ×
    √hidden_size**（√2304=48）；②层 0 仍炸（attn 输入 diff 31.7 非恒定比例）→ 单测 RMSNorm 是
    bf16 噪声 → 反查 HF：`Gemma2RMSNorm` 是 **norm(x)×(1+weight)**（权重 init 0、存偏移量）——
    用 ×weight 差 (1+w)/w 倍，小权重列 ~10×（实测比例 9.57 = 1.1167/0.1167 精确吻合）；③层内
    双残差四 norm（第二个残差基 = x1）。教训：**架构细节藏在"初始化语义"里**；新端口唯一可靠验证
    是逐层对照 HF，且 debug 脚本自身形状/口径先钉对。
16. **flash window_size 的含两端语义**：flash 文档 `[i-left, i+right]` **含两端**——causal 窗口
    W 个 key 必须传 (W-1, 0)，传 (W, 0) 多 attend 一个（probe 实测 (256,0) diff 2.1e-1 vs
    (255,0) 1.6e-2）。**"窗口大小"语义不唯一**（SDPA 的 sliding_window=W 是"含自己共 W 个"），
    先用最小对照实验钉死再写进生产代码。
17. **fp8 内核窗口掩码的 `m=-inf` 全掩块 NaN**：SWA 掩掉前导块后 m 从 -inf 起步遇到全掩块 →
    `exp(-inf-(-inf))=NaN`。修复：WINDOW>0 时 m 从 0 起步（softmax 平移不变）。flash 用"循环从
    首有效块开始"回避——**任何给内核加掩码的改动都要检查 m/l 累加器的空块行为**。
18. **meta 计数构造污染共享 RoPE 实例（2026-09 修）**：`_decide_streaming` 为估算权重用 meta
    实建模型（c2e04d4），而 `get_rope` 是 `@lru_cache(1)` 的共享实例 → 计数后 `cos_sin_cache`
    留在 meta；非流式（eager 量化）路径复用同一实例 → 首次前向 "Tensor on device meta is not
    on the expected device cuda"（gemma2 int4 实测复现；流式路径因 `_finalize_streaming` 重建
    而幸免、量化 none 因提前返回而幸免——所以此前探针全没踩到，**预存 bug 由新路径（meta 计数）
    触发且被老路径掩盖**）。修复：计数后 `get_rope.cache_clear()`。教训：共享缓存实例要警惕
    "在什么设备/上下文里被首次构造"。

---

## 7. 方法论总结（如何证明你懂推理系统）

- **先讲成本模型与上界**：投机 γ·T_draft+T_verify vs (1+αγ)、α≤模型 top-1；swap 的 D2H 带宽
  vs 重算；int4 的带宽 vs 计算主导；量化是"带宽优化不是计算优化"。
- **讲"我证伪过什么"**：大 tile 假设、分页瓶颈假设、启动税假设——证明有实证习惯而不是背书。
- **主动说"哪里是亏的"**：KV swap 本机亏、2:4 精度灾难、纯 int4 大 batch 慢、EAGLE γ=4 亏、
  Medusa 大 batch 亏、vLLM 对比的近似口径——比吹嘘可信。
- **数字带条件**：单卡、WSL2、模型量级、flash-attn 版本、dtype、run 日期；ppl 等随机语料指标
  只比同 run 内。
- **口径诚实**：vLLM 离线 API 不暴露逐请求指标 → 用聚合直方图并声明近似；decode 单步才是
  kernel 级可比口径。
- **验证分层**：单元级（内核 vs 参考）→ 引擎级（logits 对齐，同输入同位置）→ 端到端（token 流
  /困惑度）；"独立检查通过 ≠ 引擎正确"、"小形状通过 ≠ 内核正确"。
- **定位流程**：①探针分层（微基准隔离变量）；②步级计时拆阶段；③单元/引擎级分开验证；
  ④用实验证伪自己的假设。每次定位留下可复现探针脚本。
- **亮点句**："vLLM 在这张卡上跑不了 fp8 KV（FA3 是 Hopper-only），我的自研内核是唯一可跑的
  实现，且精度 KL 0.0073、top-1 100%。"

## 8. 精进路线图（已完成阶段 = 深水区素材；未完成 = 下一步计划）

> 原则：按 **面试追问频率 × 技能可迁移性 × 本机可验证性** 加权排序。穿插项：每阶段读对应
> vLLM/llama.cpp 源码做对照。

### 阶段 1：内核与性能工程（CUDA 记忆模型 + roofline 归因 + 手写 MatMul）——✅ 已完成
**交付（`benchmarks/_kernel_roofline.md` + `benchmarks/_cuda_gemm_report.md`）**：实测标定硬件
锚点（TC 48.5 TFLOPS / 带宽 370 GB/s / 36 SM）；GEMM 三种性能形态归因（fp8 大 M 81% TC、
int4 44% TC、decode 小 M 启动受限）；手写 SMEM-tiled fp16 MatMul **101% cuBLAS** + 消融
（最优配置 occupancy 仅 17%——TC 吞吐型反直觉反例）；tile 网格搜索 → **int4 大 M 小 tile 反超
19%（regs 255→128）→ 纯 int4 大 batch +28%**；教训：快速搜索的 +144% 异常值被高迭代复测推翻。
**CUDA C 补课（1b，全部 SASS 实证）**：工具链四坑（pip nvcc wheel 拆包只剩 ptxas → conda nvcc
12.8.93；gcc15 崩 pybind11 → conda gcc14；CUDAHOSTCXX 失效 → symlink；cuobjdump 12.4 不能解码
SM120 → 12.8）。手写 fp16 GEMM：FMA naive 1.6 → mma 单 tile 6.1 → **8 tile/warp + BsT 转置
布局 20.8 TFLOPS（cuBLAS 53%）**；**bank-conflict 消融：行距 32→34 消 16-way 写冲突，+86%**；
split-K 只在 block 数 < SM 数时赢（M=64 S=4 +59%）；persistent 本机全亏（0.89×）。

### 阶段 1.5：MoE（router + 循环专家 + grouped/段式后端 + 量化专家）——✅ 已完成
**为什么插入**：DeepSeek-V2-Lite 是 MLA + MoE 双机制——两个新东西一起排错会互相污染归因。
**交付**：①`layers/moe.py`（2D per-expert 对齐 HF 存盘格式——5.15 内存 3D/存盘 2D 的事实修正；
loader 零改动、量化路径继承；router 永不量化）；②loader packed 匹配改"点分段相等"；③qwen3_moe/
deepseek_v2 端口 + registry；④验证链：数学同构参考位级对照 → CPU 单测 → 端到端 parity top-1
100%（mean 0.003）；⑤量化专家 decode ≈2×；反直觉实测：串行循环**偏好集中路由**、toy 量化 top-1
全翻是**顶层贴边**假警报（gap 0.4-0.7σ）。
**1.5b（grouped）**：组织税 3.8→0.7ms/层（小 T 快 4-5×）；auto 后端（w_deq 可组；纯 int4/fp8
回退循环）；不均衡 padding 放大实测 +30%。
**1.5c（Triton 段式）**：无 padding、位级正确 + parity 100%；实测赢面 = **小 E + 极长段**
（E=8/R=32K 快 43%），E=128 持平、小段 3-10× 慢 → 默认仍 padded-bmm，段式作显式开关。教训：
内核的"理论优势"要用实测边界校验。
**下一步**：shared expert（Qwen3-235B 类）；EP 理论在阶段 5。

### 阶段 2：MLA（DeepSeek latent attention）+ SWA 滚动缓冲——✅ 已完成（2a/2b/2b-ext）
**为什么第二**：注意力是推理核心；MLA 有真模型可验证（DeepSeek-V2-Lite）；滚动缓冲把文档 TODO
变完成且用上自研内核能力。

**2a（MLA，提交 662a464 起）**：`models/deepseek_v2.py` 全模型端口（MLA + dense/MoE 混合 +
shared experts + routed scaling）+ `layers/attention_mla.py`（fused cache [c_kv|k̃_pe] 576
元素/token/层 + 吸收式 decode Triton 内核 + KV 布局泛化）+ 引擎集成。验证链：decode 内核 vs
稠密参考**位级 0 误差**（3 场景跨块）→ CPU 全模型 vs transformers 5.15 同权重 top-1 100%
（max 1e-6）→ 引擎 parity top-1 100%（mean 0.003-0.03）。上游怪癖：5.15 DeepseekV2 无缓存前向
不传因果掩码；experts 内存 3D 直挂 Parameter（state_dict 无 .weight）；flash varlen 要求 v
head_dim == k → 零填充 192 截断。**顺带修复**：RMSNorm fp32 下 x.float() 别名输入 → mul_ 原位
归一化残差中间张量（CPU fp32 参考错——逐层对照定位）。账本修正：论文共享 rope key → 576 元素
**7.11×**（旧口径 1536/2.7× 是"每头 rope key"的 naive 假设）。

**2b（SWA 滚动环，754cd22）**：`rolling_cache=True`（mistral 全层统一窗口 + bf16/fp8 KV +
ngram 投机可选）：块表 = 窗口内容清单（`Sequence.kv_j0` 行首逻辑块序号），驱逐
`(front+1)·B ≤ N−W−slack` 先释放再分配（净零 free 消耗）；refcount 守卫断言（环模型不发布/
不消费前缀缓存 → 块恒私有）；fp8 内核泛化 chunk_starts；自研 bf16 paged decode 内核 + varlen
环装配（flash 无法表达环表 key 位置）。验证：BM CPU 属性 pytest；引擎 e2e（mistral toy W=512
跨窗多轮）vs 稠密掩码参考 84 采样步 top-1 全一致。稳态内存 = 窗口 + B 余量/序列。

**2b-ext（组合解锁 + 真实模型验证，报告 `benchmarks/_stage2b_ext_report.md`）**：
① fp8 KV+环（decode 传 chunk_starts）与 fp8 KV+MLA（fused 行两段独立 scale：cal_c/cal_r、
clamp±448、内核与稠密装配双路反量化）；② 投机(ngram)+环：BlockManager 环 spec 账本修复
（表项 = 逻辑块 j0+i）、verify 行 key 集 [j0·B, end) 稠密装配喂 flash（段内相对下标 ⇒ 窗口
掩码精确）、slack=γ+2；③ 非统一窗口（gemma2 交替 local/global）**split 双池**：环池（local，
驱逐到 cap）+ full 池（global，普通分页永不驱逐）双 BM/双 GPU cache/Context full_* 侧 + 自研
内核 softcap（cap·tanh，flash 同语义）；④ 纯 int4/fp8 MLA decode：kv_b 保留反量化副本
（`is_mla_kv_b` 例外，占参 ~1%，V2-Lite 实测 113MB）→ 稠密兜底/强制 eager 消除（w8a8/sparse24
兜底仍留）；⑤ int4 组大小可配（`int4_group_size`）+ 内核尾 K 掩码与组粒度静态展开（128 路径
逐位不变）。
**验证摘要（真实模型）**：Mistral-7B int4 流式 5050-token（>W=4096）**fp8 KV 环 vs 掩码全程
逐位一致**（0.0 diff、轨迹含 EOS 全同；环表 ≤ cap 18）；gemma-2-2b-it 26 层 split 4800-token：
环池表长到 ring_cap(18) 封顶而 full/掩码继续线性增长，30 稀疏步（含越窗后）vs 手工 fp16 稠密
参考 top-1 失配 0；DeepSeek-V2-Lite（31.4GB 下载）：4 层真权重切片 fp16 引擎 vs 稠密参考
0 失配、int4(group64) 23/24，全量 27 层流式 int4 启动 55s、权重常驻 ~5GB（int4 + kv_b w_deq
113MB）、decode ~2.8 tok/s（3 并发、MoE eager、纯 int4）且中英文连贯。
**顺带修复的真实 bug**：MLA decode 内核真实尺寸共享内存超限（BLOCK_T 32→16/warps 8）；int4
内核 K 尾块漏算（10944=64×171 丢 64 列 → logits 漂移+NaN，掩码修复）；streaming 判定 meta
实建数参数（旧通用公式漏 MoE/MLA 专家 → 曾把 33GB 当小模型直建 OOM）。
**诚实边界（现行）**：Gemma-2 split 双池仅 auto KV、无投机/无 KV swap、decode eager（softcap 层禁
fp8 KV；双池 CUDA graph 未实现）；ring+medusa/eagle 断言关；滚动模型停用前缀缓存（重复 prompt 代价）；
KV swap×ring/split 未验证；fp8 KV×MLA 的组合路径有 toy 覆盖（真实 V2-Lite 尚未测 FP8 KV）；fp8 权重×MLA 与 kv_b
同机制但无单独探针；真实模型对照的采样步一致性受"异内核数值差在近并列处翻转"限制（同内核
位级、异内核 top-1 噪声带内，用稀疏步手工参考论证）；transformers 5.15 MoE `torch._grouped_mm`
仅 sm_90 → 本机无 HF-GPU 直连真权重对照（nano fp16 稠密参考替代）。

### 阶段 3：在线服务与延迟感知调度——✅ 核心功能已实现；新 decode 策略待目标硬件测量
**已交付**：OpenAI 风格 HTTP 服务、SSE 流式输出、断连取消、持续接收请求、进程内会话文本存储与
token-budget 截断/可选摘要；Top-W cache-affinity admission、aging 公平性、recompute-aware 抢占；按
TTFT slack 调整 prefill 分块与 decode/prefill 配额；request-level TPOT slack 排序与 decode 优先配额；
纯非投机 decode 的 bounded multi-step burst（默认最多 4 轮，`decode_burst_yield` 默认开：有新 prefill 等待即提前收尾；`--no-decode-burst-yield` 为消融）。
`benchmarks/scheduling_ablation.py` 使用共享请求 trace 做 baseline/all-on/逐项消融并输出 JSON，
包含 `without_tpot_aware`、`without_multi_step_decode` 与 `without_decode_burst_yield`，并单列 burst 轮数/提前让出次数/放弃槽位。
动态准入与背压也已实现：按 prompt token（扣除前缀命中预测）、请求最大输出 token 与已完成请求的全局输出长度 EWMA 估算 Prefill/Decode 工作量，并结合在线吞吐、队列和预计 KV 占用估算压力；低压力请求立即接收，超压请求进入有界 FIFO 等待，超时或达到硬上限时拒绝。前缀命中预测用 engine 空闲期物化的 `BlockManager.prefix_snapshot()`（按 `prefix_map_version()` 失效）做链式哈希估算，逐请求与实测一致（§10.3.9）。KV 容量仍按最大输出 token 保守预留。`benchmarks/admission_ablation.py` 用相同到达 trace 对比三档（关闭 / 开但不感知缓存 / 开且感知缓存），输出接收/延迟/拒绝、TTFT、E2E、吞吐与命中估算误差 JSON。
**待补**：把 multi-step 扩展到 mixed/spec decode 的安全调度边界；验证 multi-step 与 TPOT-aware 调度在
WSL2/RTX 5060 Ti 上的 TTFT、TPOT、吞吐和公平性影响（burst 让出已实现并有计数，收益仍待实测）；
**按模型/费率校准准入阈值**——命中估算已验证准确，但 §10.3.9 实测默认阈值在高费率下过度拒绝。
已有历史吞吐数字不能证明新增策略的收益，需运行当前 checkout 的消融脚本后再更新结论。

### 阶段 4：量化/稀疏算法层（GPTQ 误差补偿 + 剪枝感知）——**第四**
**为什么第四**：现有量化是应用层，补算法层才能答"为什么 AWQ 有效、2:4 怎么不丢精度、GPTQ 和
RTN 差在哪"。**动作**：①GPTQ（Hessian 逆 + 逐列误差补偿）在 0.6B 上与 RTN/AWQ 对比 ppl；
②SparseGPT 式误差补偿稀疏（把 2:4 的 KL 8.5 修到可用）；③精度方法论：校准集设计、离群通道
分析、误差传播曲线。

### 阶段 5：分布式推理理论（PP/DP/EP + NCCL 集体通信）——**最后**
**为什么最后**：单卡无法实测，性价比最低；作理论补强。**动作**：PP 的 1F1B 内存分析（bubble
比例 = (p-1)/(m+p-1)）与切分策略、EP 的路由 + 通信量、NCCL allreduce 的环/树带宽模型；paper
推导 + 数值模拟验证（无实机）。

### 穿插阅读（每阶段做一块）
- vLLM：attention backends（阶段1 对照内核）、scheduler（阶段3）、quant（阶段4）；
- llama.cpp：GGUF 量化与 kernel 设计（阶段1/4）；
- 论文：FlashAttention、MLA 原论文、PD 分离/Mooncake、GPTQ/AWQ/SmoothQuant、Megatron 1F1B/
  DeepSeek MoE。

## 9. 代码地图（功能 → 文件和核心符号 → 运行链；行号不作为稳定定位）

### 9.1 文件地图（按核心符号定位）

| 文件 | 职责 | 核心符号（行号不固定） |
|---|---|---|
| `nanovllm/llm.py` | 公共 API 入口 | `LLM`（纯别名，没逻辑） |
| `nanovllm/config.py` | 引擎配置解析 | `Config` （字段注释 = 功能词典；`__post_init__` 断言合法性） |
| `nanovllm/sampling_params.py` | 采样参数 | `SamplingParams`（禁 greedy） |
| `nanovllm/engine/llm_engine.py` | **引擎主循环** | `LLMEngine` ：`add_request`  / `_verify`  / `_medusa_drafts`  / `_eagle_drafts`  / `step`  / `generate`  / `collect_metrics`  |
| `nanovllm/engine/scheduler.py` | **调度器** | `Scheduler` ：`schedule`  / `_compute_draft`  / `_schedule_mixed`  / `_spec_rows`  / `_schedule_spec`  / `_schedule_mixed_spec`  / `_schedule_prefill`  / `_schedule_decode`  / `preempt`  / `swap_out`  / `swap_in`  / `_try_swap_in`  / `_maybe_finish`  / `postprocess`  / `postprocess_spec`  |
| `nanovllm/engine/sequence.py` | 序列状态（CPU 侧真源） | `SequenceStatus` 、`Sequence` （`kv_table` 滚动/全池表；`__getstate__` ） |
| `nanovllm/engine/block_manager.py` | **KV 块池 + 前缀缓存 + COW + 滚动环** | `Block` 、`BlockManager` ：`_t`  / `ring_cap`  / `_evict_front`  / `compute_hash`  / `can_allocate`  / `allocate`  / `allocate_private`  / `deallocate`  / `can_append`  / `can_append_spec`  / `may_append_spec`  / `cow_block`  / `may_append`  / `hash_blocks`  |
| `nanovllm/engine/model_runner.py` | **打包 + GPU 执行** | `ModelRunner` ：`call` （TP）/ `cow_block`  / `swap_out`  / `swap_in`  / `warmup_model`  / `quantize_int4_weights`  / `quantize_fp8_weights`  / `prune_sparse24`  / `quantize_awq_weights`  / `_decide_streaming` （meta 计数 + `cache_clear`）/ `_streaming_quant_hook`  / `_finalize_streaming`  / `calibrate_fp8_kv`  / `_finalize_mla_mode`  / `_finalize_rolling`  / `allocate_kv_cache`  / `_ring_rows`  / `prepare_prefill`  / `prepare_mixed`  / `prepare_spec`  / `_prepare_mixed_spec`  / `prepare_decode`  / `run_model`  / `_spec_graph_hidden`  / `capture_spec_graph`  / `run`  / `capture_cudagraph`  |
| `nanovllm/engine/ngram.py` | n-gram 投机（纯函数） | `find_ngram_draft` 、`verify_drafts`  |
| `nanovllm/models/registry.py` | 按 model_type 选模型 | `get_model_class` ；`_PLANNED_BLOCKERS`  |
| `nanovllm/models/*.py` | 7 个模型族（同构模板） | qwen3  · qwen2  · llama3  · mistral  · gemma2  · qwen3_moe  · deepseek_v2 （Attention/MLP/DecoderLayer/Model/ForCausalLM）；`compute_logits` 各 ForCausalLM 末尾（qwen3.py） |
| `nanovllm/layers/attention.py` | **注意力：写 KV + flash/自研内核路由** | `store_kvcache_kernel`  / `store_kvcache`  / fp8 decode 内核  / `kv_rows_gather`  / `paged_decode_attention_fp8`  / `paged_decode_attention_bf16`  / fp8 varlen 内核  / `paged_varlen_attention_fp8`  / `Attention` （`forward` 、`_ring_varlen` 、`_decode_rows` ） |
| `nanovllm/layers/attention_mla.py` | **MLA（DeepSeek）** | `mla_store`  / `mla_gather_dequant`  / `mla_decode_kernel`  / `mla_decode_attention`  / `MLAAttention` （`forward` 、`_decode_rows` 、`_decode_kernel` ） |
| `nanovllm/layers/linear.py` | **全部 GEMM + 量化** | int8 内核  / w8a8  / int4 内核  / int4_gemm  / sparse24 内核  / sparse24_gemm  / fp8 激活量化  / fp8 内核  / fp8_gemm  / `WeightQuantMixin` （quantize_int4 、_int4_forward 、quantize_fp8 、quantize_sparse24 ）/ `LinearBase` （quantize_w8a8 、forward ）/ Column  / Merged  / QKV  / Row  |
| `nanovllm/layers/moe.py` | **MoE（1.5）** | `ExpertFFN`  / `MoE` （`_route` 、forward 、`_forward_loop` 、`_forward_grouped` 、`reference` ） |
| `nanovllm/layers/medusa.py` / `eagle.py` | 投机草稿头 | `MedusaHeads`  / `EagleLayer`  |
| `nanovllm/layers/layernorm.py` / `rotary_embedding.py` / `activation.py` / `sampler.py` | 基础算子 | `RMSNorm`（含 weight_offset 变体）/ `RotaryEmbedding` （build_cache 、forward ）/ `get_rope` （**@lru_cache(1) 共享实例——meta 污染教训 §6 故事 18**）/ `SiluAndMul` / `Sampler`（Gumbel） |
| `nanovllm/layers/embed_head.py` | Embedding + LM Head | `VocabParallelEmbedding`  / `ParallelLMHead` （继承 WeightQuantMixin） |
| `nanovllm/utils/context.py` | **每步张量契约** | `Context`  / `get_context`  / `set_context` （位置传参，**新字段必须追加末尾**）/ `reset_context`  |
| `nanovllm/utils/loader.py` | 权重加载 | `default_weight_loader`  / `load_model`  / `_load_eager`  / `_materialize`  / `_load_streaming`  |

### 在线服务与新增调度组件

| 文件 | 职责 | 主要入口 |
|---|---|---|
| `nanovllm/server.py` | HTTP/SSE、请求队列、会话上下文与压缩 | `create_app`、`GenerationManager`、`ConversationStore` |
| `nanovllm/engine/kv_transfer.py` | Prefill/Decode 池之间的 KV 主机内存布局导出与导入 | `export_kv_cache`、`import_kv_cache` |
| `nanovllm/engine/scheduler.py` | cache-affinity、aging、抢占成本与 TTFT 配额 | `schedule`、`_order_waiting`、`_slo_prefill_controls`、`preempt` |
| `benchmarks/scheduling_ablation.py` | 连续请求 trace 下 baseline/all-on/逐策略消融 | `build_arrival_trace`、`run_arrival_trace` |
| `benchmarks/context_concurrency.py` | 并发 × prompt 长度矩阵、TTFT/TPOT/E2E/吞吐、GPU 显存与实际 MHA/MLA KV 池容量账本 | `run_scenario`、`kv_capacity_report` |
| `benchmarks/kv_fp8_calibrate.py` | 文本→校准 token-ID JSON、FP8 margin 扫描、held-out 首个 decode logits/range 对比 | `run_variant`、`compare_logits`、`range_utilization` |

### 9.2 运行链

**链 A：进程启动（只跑一次）**
```
LLM(...) → LLMEngine.__init__ [llm_engine.py]
 └─ ModelRunner.__init__ [model_runner.py]
     ├─ dist.init_process_group("nccl", ...)          # 无条件，TP=1 也初始化
     ├─ get_model_class(model_type)                   # registry.py
     ├─ _decide_streaming()  → load_model(...)   # meta 计数(cache_clear)→ 流式逐层物化+量化 / eager
     ├─ eager 量化：quantize_int4/fp8/w8a8/awq/sparse24  # model_runner.py（streaming 走 chunk_hook）
     ├─ _finalize_mla_mode()  / _finalize_rolling()
     ├─ warmup_model()                            # 真实形状：JIT 编译 + 峰值显存
     ├─ allocate_kv_cache()                       # 大块 KV（MLA fused / ring / full 双池）绑层
     ├─ capture_cudagraph()                      # decode 图族 [1,2,4,8,16..512]
     └─ capture_spec_graph()                     # 投机：stride 家族 × 行容量家族（MLA/滚动跳过）
```

**链 B：每步推理循环（`generate` 内 `while not is_finished()`）**
```
step() [llm_engine.py]
 ├─ scheduler.schedule() → (seqs, kind)                # scheduler.py
 │   ├─ _try_swap_in()                            # 先把换出的 KV 换回
 │   ├─ 投机：先给 running 全算草稿 (_compute_draft )
 │   └─ 分支：waiting+running→mixed | waiting→prefill | 其余→decode/spec
 ├─ COW 拷贝：cow_pairs → call("cow_block")   # run() 之前
 ├─ swap 拷贝：swap_pairs → call("swap_out"/"swap_in")
 ├─ model_runner.call("run", seqs, kind)
 │   ├─ prepare_prefill /mixed /decode /spec  → set_context() [context.py]
 │   ├─ run_model(input_ids, positions, kind)
 │   │   ├─ kind=spec → 支持的纯批次走 spec CUDA graph；MLA/rolling/spec-mixed eager
 │   │   ├─ kind=mixed → 普通 MHA 按精确形状懒捕获/重放 mixed CUDA Graph；不支持时 eager
 │   │   ├─ kind=decode 且 bs≤512 且非 eager → decode CUDA graph 重放
 │   │   └─ 否则 eager：model(input_ids, positions)
 │   ├─ model.compute_logits(hidden)                   # LM Head（图外）
 │   └─ Sampler(logits, temperatures) → token_ids      # Gumbel 采样
 │       └─ reset_context() [context.py]
 ├─ 投机：_verify() [llm_engine.py] → postprocess_spec() [scheduler.py]
 │         → _medusa_drafts /_eagle_drafts   # 用 hidden 生成下轮草稿
 ├─ 否则：postprocess()                           # 追加 token/EOS/rehash
 └─ 收集 finished 序列 → outputs
```

**补充：PD 分离模式**
```
LLMEngine(execution_mode="pd")
 ├─ Prefill runner/scheduler 处理等待请求（该角色禁用 KV swap）
 ├─ prompt KV 经 `kv_transfer.py` 导出到主机内存
 ├─ 释放 Prefill GPU 页，在 Decode pool 分配私有页并导入 KV
 └─ 同一 engine step 内随后调度 Decode runner；两个阶段串行，无远端通信或异步重叠
```

**补充：在线服务**：HTTP route → 动态准入估算与接收/FIFO 延迟/拒绝 → 有界 `GenerationManager` 队列 → 单 engine worker 持续提交/step → SSE token events；`conversation_id` 绑定主机内存文本历史、会话 prompt token 预算与 prefix hit 计数，KV 不跨对话轮次固定驻留，但相同 token 前缀仍可通过全局前缀缓存复用。

**链 C：单层前向（以 Qwen3 为例，锚点 = qwen3.py）**
```
Qwen3ForCausalLM.forward
 └─ Qwen3Model.forward
     ├─ embed_tokens(input_ids) → hidden
     ├─ for layer: Qwen3DecoderLayer.forward
     │   ├─ Qwen3Attention.forward
     │   │   ├─ qkv_proj(x) → q,k,v                    # QKVParallelLinear
     │   │   ├─ RotaryEmbedding(q,k)                   # 按 positions 旋转
     │   │   ├─ Attention.forward [attention.py]   # 链 D
     │   │   └─ o_proj(o)
     │   ├─ Qwen3MLP.forward （gate_up → SiluAndMul → down）
     │   └─ 残差相加（norm 走 add_rms_forward）
     ├─ norm(hidden, residual)                         # RMSNorm（残差融合）
     └─ compute_logits  → ParallelLMHead          # 词表映射（TP>1 gather）
```

**链 D：Attention 数据流（Context 契约——本项目最核心的接口设计）**
```
prepare_* 构建 GPU 张量 ──set_context()──> Context（全局单例）
   [cu_seqlens_q/k, max_seqlen_q/k, slot_mapping, context_lens, block_tables,
    chunk_starts/ring_*, full_*（滚动/split 侧）, n_prefill_tokens, is_mixed, is_spec]
                    │
Attention.forward [attention.py]  ← get_context()
 ├─ store_kvcache(k, v, k_cache, v_cache, slot_mapping)  # 本步 K/V 散写分页缓存
 └─ 读路由（按批次形态 / 模型形态）：
     ├─ is_spec / is_mixed → varlen（分块序列用缓存形状 K/V；环序列走 _ring_varlen 装配）
     ├─ MLA → MLAAttention._decode_rows/_decode_kernel（吸收式）或稠密兜底
     ├─ 滚动环 decode → bf16/fp8 paged 内核（chunk_starts 表达环位置）
     ├─ 纯 prefill → flash_attn_varlen_func（连续 K/V）
     └─ 纯 decode → fp16 flash kvcache / fp8 paged_decode_attention_fp8 / bf16 自研内核
```

**链 E：量化路由决策（以 int4 为例）**
```
LinearBase.forward [linear.py]
 └─ 已量化? → _int4_forward
     ├─ M≤128 且 N≥2048 → Triton int4_gemm （group_size 可配 + 尾 K 掩码）
     └─ 否则            → F.linear(x, w_deq)（bf16 反量化副本，cuBLAS）
 权重来源：quantize_int4  在 warmup 前一次性打包（dual-path 存 q/scale + w_deq；
 纯 int4 不存 w_deq；MLA kv_b 恒存 w_deq）
```

**链 F：投机解码完整链路**
```
Scheduler._compute_draft [scheduler.py] ─每步 CPU─> 写 seq.draft_tokens
 → schedule() → kind="spec" / "mixed"
 → prepare_spec/_prepare_mixed_spec [model_runner.py]
     verify 行 = query=[last_token, 草稿...] 的 chunked prefill，num_cached = len-1
     （滚动环：key 集 [kv_j0·B, end) 装配，spec 表项 = 逻辑块 j0+i）
 → run_model：spec CUDA graph（stride×容量家族）或 eager varlen
 → LLMEngine._verify [llm_engine.py]：γ+1 行采样 s_i ↔ 草稿 d_i 逐个验收；末行 bonus
 → postprocess_spec [scheduler.py]
     只提交接受 token；hash 范围 [num_tokens-n_acc-1, num_tokens-1)（被拒草稿不进前缀缓存）
 → medusa/eagle：_medusa_drafts/_eagle_drafts [llm_engine.py] 生成下轮草稿
```

## 10. 基准档案（指标口径 · 快速开始 · 归档实测结果 · profiling）

### 10.1 指标口径与计时

| 指标 | 定义 |
|---|---|
| Throughput | 总输出 token 数 / 总耗时（wall time），tok/s |
| TTFT | 每个请求从提交（加入调度队列）到生成第一个 completion token 的时间 |
| TPOT | 每个请求 (完成时间 − 首token时间) / (输出token数 − 1)，即稳态解码的单token延迟 |
| E2E | 每个请求从提交到完成的端到端延迟 |
| benchmark SLO 达成率 | TTFT ≤ `--slo-ttft-ms` 与 TPOT ≤ `--slo-tpot-ms` 的请求占比；TPOT 只统计至少输出 2 个 token、存在 token 间隔的请求；服务请求还可分别指定 `ttft_slo_ms` / `tpot_slo_ms` |
| preemptions | KV cache 块不足时调度器抢占（swap/回退重算）的次数，0 表示容量充足 |

计时插桩在引擎内部：`Sequence.t_submitted/t_first_token/t_completed`（driver 侧，不跨进程传输），
由 `LLMEngine.collect_metrics()` 统一导出，`benchmarks/bench.py` 统计。
**TTFT 语义注意**：`t_first_token` 在该序列 **prefill 完成的那次 postprocess** 记录，因此 TTFT ≈
请求自身 prefill 完成时刻（含排队），整批 TTFT 呈阶梯分布。在线 scheduler 可使用 TTFT deadline
slack 和 TPOT 目标；TPOT 控制依据请求级 token 间隔 EWMA， benchmark 达成率则按上面的请求平均
TPOT 定义计算。离线批处理下 TTFT<500ms 通常难达成。

### 10.2 环境与快速开始

硬件/软件：RTX 5060 Ti 16GB（sm_120，36 SM）、WSL2（11GB RAM + 4GB swap）、torch 2.8.0+cu128、
flash-attn 2.8.3.post1、triton 3.4.0。**conda 环境里 editable install 可能指向另一克隆
（如 ~/AI/nano-vllm）——运行前确认 `python -c "import nanovllm; print(nanovllm.__file__)"`；
`benchmarks/run_in_wsl.sh` 通过 PYTHONPATH 强制本工作区副本。**

```bash
# 1. 默认吞吐/延迟基准（256 seqs，in 128-1024 / out 64-512）
python benchmarks/bench.py --num-seqs 256
# 2. 共享前缀 workload（前缀缓存；非整块前缀 → 部分块共享 + COW）
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512
# 3. 前缀缓存跨批次演示：相同批次跑 3 遍，第 2/3 批 prefill 应大幅减少
python benchmarks/bench.py --num-seqs 64 --min-input-len 1024 --max-input-len 1024 \
    --min-output-len 32 --max-output-len 32 --repeat-batches 3
# 4. 与真实 vLLM 对比（隔离环境 vllm-compare；workload 一次生成两侧共享）
python benchmarks/compare_workload.py --tag small --num-seqs 128 --min-input-len 64 \
    --max-input-len 128 --min-output-len 64 --max-output-len 128
python benchmarks/compare_nanovllm.py --workload results/compare_workload_small.json \
    --kv-cache-dtype auto --output results/compare_nanovllm_small_fp16.json   # nano 侧（nano-vllm 环境）
python benchmarks/compare_vllm.py --workload results/compare_workload_small.json \
    --kv-cache-dtype auto --output results/compare_vllm_small_fp16.json       # vLLM 侧（vllm-compare 环境）
python benchmarks/compare_merge.py results/compare_*.json                     # → compare_report.md/.csv
# 5. 耗时分解（torch.profiler，prefill/decode 分开）
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64
# 6. Batch 缩放实验（吞吐-延迟权衡曲线，单引擎复用）
python benchmarks/batch_scale.py
# 7. 量化 / fp8 KV
python benchmarks/bench.py --num-seqs 256 --quantization fp8
python benchmarks/bench.py --num-seqs 256 --kv-cache-dtype fp8_e4m3
# 8. 投机（草稿质量按内容类型变化大，见 §10.3.5）
python benchmarks/spec_bench.py --speculative ngram
# 9. 连续到达策略消融（同一随机 trace，baseline/all-on/单策略关闭）
python benchmarks/scheduling_ablation.py --num-seqs 128 --arrival-rate 16 \
    --arrival-mode poisson --variants baseline all_on
# 10. 动态准入/前缀缓存感知消融（同一到达 trace，三种模式）
python benchmarks/admission_ablation.py --num-seqs 128 --arrival-rate 16 \
    --arrival-mode poisson --shared-prefix-len 512
# 11. 在线 HTTP/SSE 服务
nanovllm-serve ~/huggingface/Qwen3-0.6B --host 127.0.0.1 --port 8000
# 12. 并发 × 上下文长度矩阵 + 实际 KV 池容量/显存账本（JSON + CSV）
python benchmarks/context_concurrency.py --context-lengths 512,1024,2048,4096 \
    --concurrency 1,2,4,8,16 --repeats 3 --max-output-tokens 32
# 13. 真实文本 FP8 KV 校准、margin 扫描与 held-out 首个 decode logits 精度对比
python benchmarks/kv_fp8_calibrate.py --calibration-file calibration.jsonl \
    --eval-file heldout.jsonl --margins 0.9,1.0,1.1,1.25 --output results/kv_eval.json
# FP8 KV 并发测试引用上一步生成的 token-ID 校准文件
python benchmarks/context_concurrency.py --kv-cache-dtype fp8_e4m3 \
    --kv-calibration-path results/kv_eval.tokens.json --context-lengths 512,1024,2048 \
    --concurrency 1,2,4,8
```

结果 JSON → `results/bench_<workload>_<ts>.json`；并发矩阵同时输出同 stem CSV；FP8 校准输出报告 JSON、摘要 CSV 和 `.tokens.json` 校准数据；profiling → `profiles/{prefill,decode}.txt`。

`admission_ablation.py` 运行 `dynamic_off`、`dynamic_on_prefix_cache_unaware`、`dynamic_on` 三组；
第三组启用 prefix-cache 命中估算，并逐请求对比估算与实际命中 token。

混合 CUDA Graph 消融使用同一 workload 分别运行默认配置与 `--no-mixed-cudagraph`，例如：

```bash
python benchmarks/bench.py --num-seqs 256 --repeat-batches 2
python benchmarks/bench.py --num-seqs 256 --repeat-batches 2 --no-mixed-cudagraph
```

逐批 JSON 会给出 graph captures/replays/eager fallbacks 与 prefix feature parse/reuse；首次遇到形状的 capture 会计入该批耗时，优先比较已预热的后续 batch，并确认 `replays > 0` 才表示 workload 实际走过 mixed graph。调度消融 JSON 另含 generation 失效重解析、缓存淘汰、deferred-free 引用排队/提交数和峰值待释放块数；CPU 用例覆盖对应的分配、释放、LRU 淘汰、swap release 和重复共享引用延迟释放边界。

Multi-step Decode / TPOT 目标调度 / burst 提前让出用同一到达 trace 消融：

```bash
python benchmarks/scheduling_ablation.py \
  --variants baseline all_on without_multi_step_decode without_decode_burst_yield \
  --num-seqs 128 --arrival-rate 16 --arrival-mode poisson --slo-tpot-ms 20 --max-decode-steps 4
```

每组保留 request TPOT 目标达成率、TPOT/TTFT/E2E 分位数、输出吞吐、decode forward 数和 burst 产生的 token 数，以及新增的 burst 轮数/提前让出次数/放弃的 decode 槽位；`without_*` 结果分别对比 baseline 和 all-on。单次吞吐基准可用 `benchmarks/bench.py --default-tpot-slo-ms 20` 启用默认 TPOT 目标，并用 `--no-tpot-aware-scheduling`、`--no-multi-step-decode` 或 `--no-decode-burst-yield` 关闭单项功能。Multi-step 只进入纯非投机 decode，因此应结合目标硬件运行和多种到达模式评估，不把历史数据外推为效果结论。

准入与前缀命中估算用两个口径交叉验证（估算在响应里，实测在引擎里）：

```bash
python benchmarks/admission_ablation.py --num-seqs 32 --arrival-mode constant --arrival-rate 1 \
  --shared-prefix-len 512          # 低费率：观察估算法是否正确
python benchmarks/admission_ablation.py --num-seqs 64 --arrival-mode poisson --arrival-rate 8 \
  --shared-prefix-len 512          # 高费率：观察拒绝/延迟与吞吐代价
python benchmarks/prefix_cache_probe.py --prefix-len 512       # 进程内：scheduler 估算 vs 提交复用
python benchmarks/prefix_cache_verify.py --base-url http://127.0.0.1:8000   # HTTP：估算 vs 实测
python benchmarks/prefix_estimate_check.py                     # 三条最小对，直接读响应字段
```

并发矩阵每个点把一批请求同时交给在线 `GenerationManager`，默认关闭预测准入以测调度与 KV 容量；`--dynamic-admission` 可把接收/延迟/拒绝策略纳入压测。每个请求的输出长度固定，因此输出速率与 TPOT 分布可横向比较。JSON 保留逐请求结果，并汇总 TTFT/TPOT SLO 达成率、admission 估计与等待、吞吐、抢占、swap、prefix hit、KV pool 峰值和显存峰值；CSV 汇总每个矩阵点。容量账本直接从 MHA/MLA cache tensor 字节数和 BlockManager 页数计算：full-history 池按 `(prompt tokens + max output tokens)` 预留整块页数；rolling 池按 prefill 的完整 prompt 分配峰值与生成期 `ring_cap` 占用中的较大值估算，split rolling/full 模型分别报告两池容量。它不包含 CPU swap，也不代表同延迟服务能力。CUDA peak 包括 engine 初始化后当前分配基线和本次场景峰值，JSON 同时保留 peak delta。

FP8 校准输入支持 JSONL 的 `prompt`/`text` 字段或纯文本；省略语料时使用脚本内置小语料，并把每条短 prompt 重复到 1024 token 以覆盖长一些的 KV 历史；用户语料默认保留原始长度，可用 `--calibration-context-length` / `--eval-context-length` 显式扩展。Gemma-2 的 attention logit soft-cap 当前不兼容 FP8 KV。报告的 range utilization 是 held-out 每层/每段 activation 最大绝对值除以 `448 × calibrated_scale`；大于 1 表示观测到的最大值被 clamp，不是逐元素饱和比例。logit 精度指标只纳入首 token 一致的 prompt，确保对比 decode 时两侧历史相同；实际部署评估应覆盖真实请求分布，并把校准与验证语料分开。

### 10.3 归档实测结果（截至各表日期；每张表自带 workload，跨表数字只作量级参考）

以下性能记录没有覆盖当前新增的在线服务、TTFT SLO 调度、Top-W/aging/recompute-aware 组合；请勿将
旧数字当成新功能的 benchmark。当前策略的可比基线由 `benchmarks/scheduling_ablation.py` 生成。

#### 10.3.1 与真实 vLLM 对比（2026-08-18，Qwen3-0.6B）

**环境**：隔离 conda 环境 `vllm-compare`（`benchmarks/setup_vllm_compare.sh`）。vLLM **0.10.2** +
torch 2.8.0+cu128 + **同一份 flash-attn 2.8.3.post1 wheel**（注意力后端同源）；transformers 4.57.6；
`XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`。两侧配置对齐：`gpu_memory_utilization=0.9`、
`max_model_len=4096`、`max_num_batched_tokens=16384`、chunked prefill、CUDA graph、prefix
caching、无 CPU offload。差异如实记录：**block size 256 vs 16**（KV 容量 nano 107,776 vs
vLLM 97,440 token）。
**指标口径（诚实声明）**：nano = 逐请求精确时间戳；vLLM 0.10.2 V1 离线 API **不暴露逐请求指标**
（`RequestOutput.metrics` 恒 None）→ 用 `LLM.get_metrics()` 聚合直方图（**avg = sum/count 精确；
p50/p99 = 桶内线性插值近似**）。另：**decode 单步耗时（`_step_timing.py`）才是两侧可比的
kernel 级指标**。

| workload | 指标 | nano-vllm | vLLM 0.10.2 | nano/vllm |
|---|---|---|---|---|
| **small**（128 seqs，in 64-128，out 64-128，容量内） | throughput | **6587 tok/s** | 4624 | **1.42×** |
| | TTFT p50 / p99 | 353.1 / 353.3 ms | 372.0 / 497.4 | 0.95 / 0.71 |
| | TPOT p50 / p99 | **13.5 / 13.7 ms** | 17.4 / 24.9 | **0.78 / 0.55** |
| | E2E avg / p99 | **1.62 / 1.89 s** | 2.27 / 4.92 | **0.71 / 0.38** |
| **clean**（256 seqs，in 128-1024，out 64-512，超容双方都抢占） | throughput | **2552 tok/s** | 1888 | **1.35×** |
| | TTFT p50 / p99 | 2520 / 21862 ms | 3246 / 39034 | 0.78 / 0.56 |
| | TPOT p50 / p99 | **45.1 / 73.6 ms** | 56.0 / 149.2 | **0.81 / 0.49** |
| **long fp8**（128 seqs，1024 in + 128 out，147k 总上下文） | throughput | **1854 tok/s**（fp8 KV，0 抢占） | 1150（fp16） | **1.61×** |
| long fp16 | throughput | 1421 tok/s（2 抢占） | 1150 | 1.24× |

**decode 单步耗时（引擎墙钟，同口径）**：small ~13.5ms（nano TPOT）vs vLLM ~17ms；clean fp16
36.4ms vs vLLM ~56ms（含抢占）；long fp16 29.2ms / fp8 32.2ms vs vLLM ~33ms——**fp16 打平或略快，
fp8 已达 vLLM fp16 水平**。
**结论与诚实修正**：①吞吐全面领先 1.35-1.61×（prefill 阶段快 ~2×）；②vLLM 0.10.2 在 sm_120
上无法跑 fp8 KV：V1 不支持 `kv_cache_dtype`（回退 V0），V0 fp8 路径选 XFormers → xformers 0.0.32
把 fp8 派发到 FA3（Hopper sm_90 专属）→ `CUDA error: invalid argument`；vLLM 0.11-0.13 V1 fp8
也是 FA3 路线（`flash_attn_supports_fp8()` 要求 capability.major==9）——**nano 自研 fp8 KV 是
这张卡上唯一可跑的实现**；③所有数字条件：单卡 WSL2；表内 vLLM 数字来自 0.10.2 V1（fp16），
V0 的 FP8 回退路径在本卡无法完成推理，因此没有 V0 性能数据；
flash-attn 2.8.3.post1、无 FlashInfer、WSL `pin_memory=False`。

#### 10.3.2 混合调度 + 前缀缓存 + batch 缩放（Qwen3-0.6B）

- **混合调度收益**（同 workload 同 seed 方案对比旧"先全 prefill 后 decode"）：吞吐全档
  **+7.2%~+21.3%**（256 档 4854→5840）；TTFT p50 全档下降（早完成者下一步即出 token，不再空等）；
  抢占 384 档 31→20、512 档 **141→71（近半）**；TPOT p50 同步下降（256 档 33.07→27.44ms）。
  长期 workload 视角：long fp16 +11.3%（抢占 21→2）、clean fp16 +10.2%（85→68）。**诚实说明**：
  span 口径的 TPOT 受"总工作量"下界约束，混合批次的真正收益是消除死等 + 抢占压力（不体现在
  span 指标里）。
- **前缀缓存**：跨批次 1024-token 相同 prompt（64 seqs×3 遍）：冷缓存 prefill 65536 tok/4 步；
  满块复用后 batch 1/2 prefill **0 tok / 0 步**。**部分块共享 + COW**（300-token 批次，含 44-token
  部分块）：batch 1 prefill 2816 tok（64×44，非缺陷——batch 0 的 decode 把共享部分块写成了 76
  token，缓存内容已变，只有满块可复用；**前缀缓存只对"内容真正一致"的部分生效**）。共享前缀 512
  （128 seqs）：批次内前缀块建立后 prefill 只算尾部（25,393 vs 77,824 tok，67% 跳过）。E2E 不是
  前缀缓存收益的正确视角——用重复批次或 prefill 计算量看。
- **batch 缩放**（混合调度重测，干净 workload in 64-256/out 32-128）：

| num_seqs | throughput (tok/s) | TTFT p50 | TPOT p50 | preemptions |
|---|---|---|---|---|
| 16 | 1639 | 62.4ms | 5.16ms | 0 |
| 32 | 2857 | 135.0ms | 6.03ms | 0 |
| 64 | 4524 | 243.7ms | 7.80ms | 0 |
| 128 | 5325 | 414.2ms | 13.27ms | 0 |
| 256 | **5840（峰值）** | 845.0ms | 27.44ms | 0 |
| 384 | 5029 | 839.3ms | 50.09ms | 20 |
| 512 | 5245 | 1565.1ms | 54.08ms | 71 |

  结论：吞吐 256 附近见顶——KV 容量（421 块）被 384+ seqs 超出后抢占侵蚀收益，但混合调度让
  decode 提前释放块、容量压力缓解（512 档反超 384 档的"单调回落"消失）；TTFT 随 batch 近似线性；
  512 档 TTFT p50 跳升是单次运行噪声（p99 5462ms 长尾），需多次取中位数。

#### 10.3.3 FP8 KV cache（`--kv-cache-dtype fp8_e4m3`，Qwen3-0.6B）

自研 Triton paged 内核直接读 FP8(E4M3)：decode v6（直接 fp8 load + 硬件 cvt 反量化 + MMA，
QPAD=16、GQA 融合、BLOCK_T=32/warps=1）、varlen v7（多查询扩展，逐列因果掩码 `<=`）；写路径
warmup 校准每层 scale、**显式 clamp±448**（§6 故事 1）。

当前实现新增真实 token-ID 校准输入与 scale-margin 参数：`kv_calibration_path` / `kv_fp8_scale_margin`；
`benchmarks/kv_fp8_calibrate.py` 可从文本/JSONL 生成输入，扫描多个 margin，并用 held-out 首个
decode logits、KL/top-k/logit 误差和 activation range utilization 比较。下面的 KL/吞吐数值是此前
归档测试的结果，不是此次校准脚本在当前环境测得的数据。

- 容量：421 → **802 块**（1.9×）；精度：**KL 0.0073、top-1 100%**（首个 decode 步对齐 logits）
- 引擎级 decode 单步：long 32.2ms、clean 35.0ms（vs vLLM fp16 ~33/~56ms，§10.3.1）；吞吐以
  §10.3.1/§10.3.2 同批表为准
- **大 tile 假设被数据否定**（BLOCK_T 64/128 因寄存器压力 1.1-13× 更慢）
- **诚实记录**：短上下文（<256 token）仍有 ~15-25% 差距（内核开销未摊薄）；split-K/持久内核是
  进一步路线；SWA 窗口（Mistral/gemma2-local）由 WINDOW 掩码/chunk_starts 支持（2b 起滚动环走
  bf16/fp8 自研内核，见 §8 阶段 2）

#### 10.3.4 W8A8（`--quantization w8a8`，Qwen3-0.6B）

per-group（K 维 128，AWQ 标准）int8 权重 + per-token int8 激活 + Triton int8 GEMM（int32 块内
累加、乘组 scale 后 fp32 跨组累加）；SmoothQuant 校准折权重（恒等变换）。精度：**KL 0.064→
0.0379（-41%），top-1 87.5%→100%**（per-group 把 scale 粒度细 8×，残余误差来自激活 per-token
量化）；性能：5382→4497 tok/s（-16%，int8 GEMM 未调优）；权重显存减半。路线：GPTQ 式舍入、
int8 GEMM tile/warp 调优。

#### 10.3.5 投机解码（Qwen3-0.6B；`spec_bench.py`）

**ngram（verify CUDA graph 化后的最终版）**：

| 风格 | bs | α | 吞吐 baseline→spec | TPOT p50 |
|---|---|---|---|---|
| repeat（echo，最好情况） | 8 | 1.0 | 1435→**5561（+3.87×）** | 5.2→**1.2ms** |
| repeat | 256 | 0.99 | 7749→**11076（+1.43×）** | 27.5→**15.5ms** |
| json（结构化续写） | 8 | 0.42 | 856→**982（+1.15×）** | 7.2→3.7ms |
| free（随机 token） | 8 | 0.34 | 1180→**1236（+1.05×）** | 5.1→5.1ms |
| json / free | 256 | 0.26 / 0.46 | 0.64× / 0.85×（亏） | — |

**为什么从"打平"变"赢"**：verify 路径本身高效（47µs/tok vs decode 110µs/tok）——初版打平的真凶
是 **eager 启动税 ~10ms/步**（~300 次 launch）；spec CUDA graph（容量族 × 双 stride + 零长填充行，
bit-exact）后 repeat bs=8 从 1.40× → **+3.87×**。剩余亏损场景 = 草稿质量低（α<0.4），是草稿源
问题不是 verify 路径问题。**fp8+spec 解锁**：v7 varlen 内核免"逐层全缓存反量化"（~18GB/步）→
bs=8 repeat 0.15×→**+3.92×**、free +1.31×；bs=256 repeat +1.53×（与 fp16 同量级，容量优势不变）。
**Medusa**：bs=8 repeat **+1.57×**（TPOT 5.4→4.4ms）；bs=256 repeat 0.61×、free/json 大 batch
0.49-0.70×（α 0.02-0.27——0.6B 可预测性天花板，head_0 达模型 top-1 的 87%）。
**EAGLE-1**：γ=2 repeat **+3.26×**（α 0.525）；γ=4 只有 +0.63×；free γ=2 +0.70×（α 0.19）。
条件：WSL2 单卡、flash-attn 2.8.3、0.6B（大模型 + 结构化内容 α 更高，结论会右移）；vLLM 同款
ngram 对照测试留作后续。

#### 10.3.6 INT4/AWQ/2:4 稀疏（Qwen3-0.6B）

**精度（真实文本 ppl——决定性指标）**：语料 = 12 条模型自生成续写（每次运行重新采样 → 有 run
间波动）。**2026-09-07 复测**：fp16 **3.23** / fp8 **3.25**（+0.6%）/ int4(RTN) **4.17**（+29%）/
awq **3.70**（+15%，把 int4 差距砍半）。历史 run（同脚本不同语料）：fp16 3.32 / int4 4.38 /
awq 3.76（2026-08-16 报告），fp16 3.60 / fp8 3.60 / int4 4.81 / awq 4.22（2026-08-24 报告）——
**跨 run 只比同批内**。引擎级对齐（决定性、无 run 波动）：fp8 权重 KL 0.017 top-1 100%；fp8 KV
KL 0.0073；w8a8 KL 0.0379；int4 8-prompt KL 1.08（**注意：8-prompt KL 与 ppl 曾方向相反——被
尾部单点主导，ppl 才是真相**）。
**7B 级补充（Mistral-7B-v0.1，2026-09-07，3060 token，`_ppl_7b.py`）**：7B 的 fp16 引擎在本机
装不下（14.5GB 权重 + KV），评测/校准改用 **fp16 裸模型直构 CUDA**（逐 key 加载，峰值 ≈ 权重
+1GB）；语料由 **fp8 引擎**生成（fp8≈fp16，流式可装载）。**方法学坑**：首轮曾用 int4 引擎生成
语料——自生成文本对被评模型 in-distribution，测得 int4 5.195 < fp16 5.285 的**伪优势**，该批
弃用重跑；教训：**语料生成器不能与被评量化级同源**。校准同 0.6B 的 α 网格 + llm-awq 误差目标，
但语料共用 + 每层 512 行样本（0.6B 为引擎 prefill 激活 1024 行）→ awq 结论保守读。scales 存
`results/awq_scales_Mistral-7B-v0.1.pt`（128 层），引擎 `awq_scales_path` 可直接用。

| 模式 | Mistral-7B ppl | vs fp16 | 0.6B 参照（各自语料，仅比方向） |
|---|---|---|---|
| fp16 | **4.05** | — | 3.23 |
| fp8 | 4.05 | **+0.2%** | +0.6% |
| int4(RTN) | 4.23 | **+4.5%** | +29% |
| awq（α 搜索） | 4.15 | **+2.7%（砍掉 int4 差距 ~40%）** | +15% |

结论：**7B 对 RTN int4 的鲁棒性远强于 0.6B**（+4.5% vs +29%，符合低比特量化随模型规模稳健的
文献规律）；awq 仍有效、收益比例略降；fp8 ≈ 无损。α 分布：qkv 0.2-0.3 / gate_up 0-0.2（常选
RTN 基线）/ o 0.1-0.3 / down 0.1-0.5；个别层出现 α=1.0 且 s 动态范围极大（样本量不足的过拟合
信号，保守读）。结果 JSON `results/ppl_7b.json`。

**吞吐**（干净 workload in 64-256/out 32-128；int4/awq 为双路径配置，与 fp16 同批测量）：

| 模式 | bs=256 | bs=8 | 说明 |
|---|---|---|---|
| fp16 | 4792 tok/s | 1099 tok/s | 基线（该批次） |
| fp8 权重 | **5825 tok/s（1.22×）** | — | 全模式峰值（K=4096 的 8B 上 prefill TTFT 210ms vs int4 813ms） |
| int4（双路径） | 5057（1.06×） | 1488（**1.35×**） | 显存 1.73GB（w_deq 定价，比 fp16 大 15%） |
| awq（双路径） | 4932（1.03×） | 1305（1.19×） | ppl 更好 |
| int4（纯） | 3073（0.64×）→ tile 搜索后 **3916.5（0.82×）** | 1033 | 显存 0.85GB；int4 组 64/尾 K 掩码后 128 路径逐位不变 |
| sparse24 | 2357（0.49×） | 385（0.35×） | 内核 bit-exact；一次性剪枝 KL 8.5（丢 35% 权重质量） |

**内核形态结论**：软件 int4 只赢权重带宽主导的小 M GEMM（gate_up M=8 4.36×、lm_head 3.42×、
qkv 1.6-2.0×；down_proj 0.40×，M≥128 全输 0.2-0.6×）——**双路径按形态路由**是正面解法；
cuSPARSELt 每调用 0.3-0.5ms、CUTLASS 仅 sm_8x（sm_120 全废），软件 2:4 无稀疏 MMA 只是带宽优化。

#### 10.3.7 新模型端口（Mistral-7B SWA / Gemma-2-2B，2026-08-24 首测 + 2026-09-07 复测）

**正确性（HF 参考 prefill logits，`_parity.py`）**：Mistral-7B-v0.1 top-1 100%（mean 0.014，
首跑即过；SWA 窗口数学由 `_swa_probe.py` 独立验证）；gemma-2-2b-it top-1 100%（mean 0.022，
三个隐藏架构细节修完后）。**真实长解码**（阶段 2b/2b-ext）：见 §8 阶段 2 验证摘要。

**吞吐**（`benchmarks/bench.py`，in 128-1024 / out 64-512，0 抢占；JSON 存档于 results/）：

| 模型 | 日期 | seqs | 吞吐 | decode | TPOT avg/p50 | TTFT avg | 权重 | KV |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B（int4 流式纯 int4） | 2026-08-24 | 32 | 311.4 tok/s | 475 tok/s | 46.8 / 49.0ms | 10.7s | 4.14GB | 213 块 |
| gemma-2-2b（int4 双路径） | 2026-08-24 首测 | 64 | 1094.7 tok/s | 1211 tok/s | 35.9 / 34.6ms | 2.47s | — | 220 块 |
| gemma-2-2b（int4 双路径） | 2026-09-07 复测 ×2 | 64 | 782.7 / 836.6 tok/s | 841 / 906 tok/s | 46.2-53.0 / 44.8-52.1ms | 2.56-2.67s | **7.45GB**（双路径实测；纯 int4 3.40GB） | 220 块（56,320 tok） |

要点：Mistral TTFT 10.7s = 32×~576 token 预填充批在 7B int4 上的真实成本（吞吐 311 tok/s 与
Llama-3.1-8B 303.5 tok/s（bs=16）同量级，两代 7B+ 端口互相印证）；SWA 长上下文（4876-token
prompt 跨 sliding_window=4096）分块 prefill + decode 全路径跑通（957 tok/s prefill），无
NaN/崩溃；attn soft-cap 量级（cap=50 的 tanh 在层 0 原始 logits ±11 时最大只改 1.7%——flash
原生 softcap 精确实现；final cap=30 压 logits ±30+，必须实现）。**诚实记录**：gemma2 双路径
09-07 两轮复测一致地比 08-24 首测慢 ~25-30%（同 workload 同 seed、KV 220 块逐 token 复现）；
原因未定位（环境波动或 2b 解码路径改动），跨日期比较以 09-07 为准、首测数字仅作对照。mistral
行仍为 08-24 数据（未复测）。gemma2 softcap 层不支持 fp8 KV（断言拦截）；全部条件：WSL2 单卡、
bf16、flash-attn 2.8.3.post1。

#### 10.3.8 阶段 2 组合（MLA / 滚动环 / 2b-ext）关键数字

完整证据表与复现命令见 `benchmarks/_stage2b_ext_report.md`；本文 §4.8/§8 阶段 2 已摘要。
速查：MLA fused 576 元素/token/层（7.11×，修正口径）；MLA decode 内核位级 0 误差、引擎 parity
top-1 100%；fp8 KV×环/×MLA 与 spec×环 toy 全绿（同内核位级、异内核 top-1 噪声带内）；
真实 Mistral-7B fp8 环 vs 掩码 5050-token **逐位一致**；真实 gemma-2-2b-it split 30/30 稀疏步、
环池 cap 18 封顶；DeepSeek-V2-Lite 4L parity 0 失配 / 全量流式 int4 ~2.8 tok/s（3 并发、MoE
eager）、kv_b w_deq 113MB。

#### 10.3.9 在线准入与前缀命中估算（2026-10-06，Qwen3-0.6B，RTX 5060 Ti / WSL2，mixed 单卡）

**目的**：验证"准入估算的前缀命中"和"引擎实际复用"是否吻合，并做准入三档消融
（关闭 / 开但不感知缓存 / 开且感知缓存）。所有数字来自本 checkout 的实跑，JSON 见
`results/prefix_cache_*`、`results/admission_ablation_*`。

**① 估算 vs 实测（逐请求）**

| 探针 | 场景 | 结果 |
|---|---|---|
| `prefix_cache_probe.py`（进程内，scheduler Top-W 估算 vs `num_prefix_cached_tokens`） | 首发 4 条 + 复用 6 条（精确重复 / 新后缀 / 半前缀） | **10/10 完全相等**，命中档 0 / 256 / 512 token，绝对误差 0，全部块对齐（block_size=256） |
| `prefix_cache_verify.py`（HTTP `nanovllm-serve`） | 8 请求三波（首发 / 精确重复 / 新后缀），冷启动服务 | **8/8 `admission.estimated_cache_hit_tokens == context.prefix_cache_hit_tokens`，Δ=0**；首发两条 0/0（缓存为空），复用六条 512/512（预热过的服务上首发也会命中 512，同样 Δ=0） |
| `prefix_estimate_check.py`（HTTP 三条最小对） | 同前缀重复 + 换尾部 | 三条都是 est=512 / actual=512，但 `predicted_ttft_ms` 从缓存未命中场景的 ~65-77 ms 降到 ~22-24 ms，说明命中确实进入了准入的 TTFT 预测 |
| `/health` 累计口径 | 一次 8 请求验证后 | 准入累计估算 6144 vs 引擎累计复用 3072；差异来自**并发波次内后到请求共享同一批块**（估算按提交时快照预测 1024，实际只新复用 512），属口径差异而非估算错误——逐请求 `context` 字段是权威对照 |

**② 准入三档消融（`admission_ablation.py`，每档新引擎 + 同一到达 trace，共享前缀 512）**

低费率（32 请求 @1 req/s constant）：

| 变体 | 接受/拒绝 | 延迟(defer) | 吞吐 tok/s | TTFT p50/p99 ms | E2E p99 ms | 估算/实际命中 token | 平均估算误差 |
|---|---|---|---|---|---|---|---|
| 关闭准入 | 32/0 | 0 | 264.9 | 313.3/336.6 | 4745 | 0 / 15872 | -496.0 |
| 开·不感知缓存 | 32/0 | 0 | 264.5 | 312.9/352.9 | 4662 | 0 / 15872 | -496.0 |
| 开·感知缓存 | 32/0 | 0 | 264.6 | 319.0/337.0 | 4702 | 15872 / 15872 | **+0.0** |

高费率（64 请求 poisson；"相对关闭准入"= 吞吐变化）：

| 场景 | 变体 | 接受/拒绝 | 吞吐 tok/s | 相对关闭准入 | TTFT p50/p99 ms | 平均估算误差 |
|---|---|---|---|---|---|---|
| 4 req/s | 关闭准入 | 64/0 | 828.7 | — | 481.0/752.8 | -504.0 |
| 4 req/s | 开·不感知缓存 | 45/19 | 667.8 | -19.4% | 536.7/1199.4 | -500.6 |
| 4 req/s | 开·感知缓存 | 51/13 | **725.7** | -12.4% | 535.8/1218.3 | **-10.0** |
| 8 req/s（seed 0） | 关闭准入 | 64/0 | 1130.5 | — | 563.6/1044.1 | -504.0 |
| 8 req/s（seed 0） | 开·不感知缓存 | 30/34 | 725.7 | -35.8% | 609.6/2084.8 | -494.9 |
| 8 req/s（seed 0） | 开·感知缓存 | 34/30 | **776.9** | -31.3% | 617.6/2083.8 | **-15.1** |
| 8 req/s（seed 2） | 关闭准入 | 64/0 | 993.1 | — | 607.1/905.8 | -504.0 |
| 8 req/s（seed 2） | 开·不感知缓存 | 28/36 | 700.5 | -29.5% | 655.3/2007.7 | -493.7 |
| 8 req/s（seed 2） | 开·感知缓存 | 41/23 | **872.1** | -12.2% | 630.4/1898.9 | **-12.5** |
| 8 req/s（seed 3） | 关闭准入 | 64/0 | 1161.2 | — | 584.9/943.8 | -504.0 |
| 8 req/s（seed 3） | 开·不感知缓存 | 19/45 | 533.7 | -54.0% | 605.0/2005.3 | -485.1 |
| 8 req/s（seed 3） | 开·感知缓存 | 32/32 | **805.8** | -30.6% | 612.0/1630.4 | **+0.0** |
| 16 req/s | 关闭准入 | 64/0 | 1408.4 | — | 1062.8/1577.1 | -504.0 |
| 16 req/s | 开·不感知缓存 | 17/47 | 791.5 | -43.8% | 708.5/1188.6 | -481.9 |
| 16 req/s | 开·感知缓存 | 17/47 | 775.0 | -45.0% | 699.9/1193.7 | -30.1 |

**读数（诚实结论）**：
- **缓存感知准入确实有用，而且稳定**：在 4 / 8(×3 seed) req/s 全部 5 个对照点上，"感知缓存"比"不感知"多接受 4~13 个请求、吞吐高 4.9%~51%（seed 2 +24.5%、seed 3 +51.0%），命中估算误差从约 -490 token 收到 -10~0 token；16 req/s 饱和点上接受数相同（各 17），吞吐打平（791.5 vs 775.0，噪声内），说明收益来自"把命中算进工作量"从而少误拒，而不是排序本身。
- **但默认准入阈值整体是净损失**：4/8/16 req/s 下"感知缓存"相对"关闭准入"分别 -12.4% / -12.2~-31.3% / -45.0% 吞吐；越是高费率越亏（拒绝 47/64 请求）。同时 TTFT p99（16 req/s：1577 → 1194 ms）与 E2E p99 明显改善——因为被拒的请求根本不进系统，这是"少做事换好看的尾延迟"，不能当作收益。
- **1 req/s 下准入完全不触发**（0 拒绝 0 延迟），三档在噪声内，缓存感知的唯一可见效果是把估算误差从 -496 打到 0：这是"估算正确性"实验，不是性能实验。
- 口径提醒：`/health` 的 `estimated_cache_hit_tokens` 是**准入累计潜在命中**，`actual_cache_hit_tokens` 是引擎累计实际复用；同批并发请求共享同一批块时后者小于前者（一次 8 请求验证里 6144 vs 3072），**逐请求** `context.estimated_cache_hit_tokens` vs `context.prefix_cache_hit_tokens` 才是权威对照口径。

**④ 准入校准：必须用"会打穿 SLO 的负载"才看得出收益（2026-10-06）**。8 req/s + 2 s TTFT 目标时 64 个请求全部达标（TTFT p50 ~0.6 s），准入自然只能靠拒绝"省事"——那不是收益。换 24 req/s / 128 请求 / 共享前缀 512（`results/admission_ablation_r24_overload.json`）：

| 变体 | 接受率 | 拒绝率 | 吞吐 tok/s | TTFT p50/p99 ms | **被服务请求的 TTFT 达标率** | 估算/实际命中 |
|---|---|---|---|---|---|---|
| 关闭准入 | 100% | 0% | 1611.2 | 2918/4738 | **26.6%** | 0 / 65024 |
| 开·不感知缓存 | 12.5% | 87.5% | 617.4 | 810/2042 | **87.5%** | 0 / 7680 |
| 开·感知缓存 | 15.6% | 84.4% | 777.1 | 916/2038 | **95.0%** | 9216 / 9728（误差 -25.6） |

结论：**baseline 确实会打穿 SLO（只有 26.6% 的请求在 2 s 内拿到首 token，p99 4.7 s）**，准入把它换成了"少接收、但接收的都达标"（95%），缓存感知在这一档比不感知多接收 25% 的请求（20 vs 16）、吞吐 +26%（777 vs 617 tok/s）。同一批被拒请求的 `predicted_ttft_ms` 在 2.7 s 量级、`estimated_prefix_cached_tokens=512`——拒绝理由是"排队预测超目标"而非误判。仍要保留的诚实边界：准入是用吞吐换尾延迟，**被服务的比例（15.6%）就是它付出的代价**；且 `admission_target_ttft_ms` 是唯一阈值旋钮，2 s 这个数字本身没有按引擎实测 service rate 校准。

**③ decode burst 提前让出**：本 checkout 原先只有"PD decode 让出独立 prefill 队列"，mixed 模式下 burst 期间新到的 prefill 只能等 burst 跑完。本次新增 `decode_burst_yield`（默认开，`--no-decode-burst-yield` 消融）与 `decode_bursts / decode_burst_rounds / decode_burst_yields / decode_burst_skipped_slots` 计数，`scheduling_ablation.py` 增加 `without_decode_burst_yield` 变体与 "Decode burst behaviour" 报告表；实测见 §10.3.10——该让出条件在 mixed 模式下结构上恒不成立（0/7021）。

#### 10.3.10 调度策略消融：TPOT 配额与 decode burst 让出（2026-10-06，Qwen3-0.6B，RTX 5060 Ti / WSL2）

命令（用户给定口径，全量跑完）：

```bash
python benchmarks/scheduling_ablation.py --variants baseline all_on \
  without_multi_step_decode without_decode_burst_yield \
  --num-seqs 128 --arrival-rate 16 --arrival-mode poisson \
  --slo-tpot-ms 20 --max-decode-steps 4
```

补充跑了一次 5 变体（加 `without_tpot_aware`）以定位根因；两次运行的 baseline/all_on 在 1~5% 内一致（Poisson trace 噪声）。JSON：`results/scheduling_ablation_128_r16.json`、`results/scheduling_ablation_128_r16_5v.json`。

> **2026-10-06 复跑修订（第二轮）**：上一版表里的 all_on/without_multi_step 数字**不可比**——per-request 指标对"在 burst 内结束的序列"重复落盘（`_run_decode_burst` 与外侧批次都报同一 seq_id），驱动侧 `completed` 计数被重复项抬高后**提前退出跑批**：128 请求的 all_on 只统计到 **95 个不同 seq_id + 38 条重复**，吞吐是在残缺 workload 上算出来的。修法：`LLMEngine._record_once()` 按 seq_id 幂等落盘 + `reset_benchmark_metrics()` 统一重置 + harness 断言"记录数 == 请求数且无重复"。修复后同一命令（`results/scheduling_ablation_128_r16_fixed.json`）：baseline 1826.1、all_on 350.5、without_tpot_aware 1771.3、without_multi_step_decode 353.2 tok/s，**每个变体都是 128/128 唯一记录**——结论不变（TPOT 配额是 collapse 根因），但修复前的重复/残缺数字不要再引用。

| 变体 | 吞吐 tok/s | 相对 baseline | TTFT p50/p99 ms | TPOT p50 ms | E2E p99 ms | request TTFT SLO | prefill 步数 / token / tok/s |
|---|---|---|---|---|---|---|---|
| baseline（全关） | 1848.3 | — | 468/772 | 41.6 | 18213 | 60.2% | 25 / 76474 / **10098** |
| all_on（全开） | 365.8 | **-80.2%** | 43738/81343 | 170.2 | 92922 | 0.8% | **355** / 76474 / **1106** |
| without_tpot_aware | 1688.2 | -8.7% | 1133/1680 | 44.5 | 20066 | 2.3% | 23 / 76474 / 8536 |
| without_multi_step_decode | 354.4 | -80.8% | 44721/84344 | 175.2 | 96036 | 0.8% | 355+ |
| without_decode_burst_yield | 356.0 | -80.7% | 45158/83822 | 173.8 | 95608 | 0.8% | 355+ |

**读数（诚实结论）**：
- **collapse 的根因是 TPOT-aware 配额，不是 multi-step，也不是 burst**：`all_on` 与 `without_multi_step_decode` 都是 ~355 步 prefill（吞吐 -80%），而单独移除 `tpot_aware_scheduling` 就把吞吐拉回 1688 tok/s（-8.7%）、prefill 回到 23 步。机制在代码里可追：`_slo_prefill_controls` 的 `token_budget = min_budget + pressure·(max_budget−min_budget)·(1−tpot_pressure)`，当请求带 20 ms TPOT 目标而实测 TPOT ~170 ms 时 `tpot_pressure → 1`，配额被压到 **`prefill_reserve_tokens` 下限（256）**，于是 128 条 ~600-token prompt 被切成 355 个 ~215-token 的碎片步——prefill 从 10.1k tok/s 掉到 1.1k tok/s，队列 p50 等待 43 s。**触发条件是"存在活跃 TPOT 目标且实测超过目标"**：不设 `--slo-tpot-ms`/`default_tpot_slo_ms` 时 `_has_active_tpot_target()=False`，配额不收缩（baseline 正常的原因）。

**TPOT 饥饿的定量定位与修复（2026-10-06 第三轮，`tpot_starvation` 指标）**：给调度器加了收缩因子观测 `budget_scale = prefill_pressure·(1−tpot_pressure)`（→0 即 prefill 被压到下限），实测 128 请求 / 16 req/s / 20 ms 目标：**budget_scale 平均 0.001、最小 0.000，355 个 prefill 步里平均配额恰好 256 token**。再按目标值扫描（`--slo-tpot-targets 10,20,40,80,160`，64 请求）：**10/20/40/80/160 ms 全部落在 256 token**（scale 0.002~0.006），`req-TPOT-SLO` 随目标升高而升高（0%→76.6%）但实测 TPOT 恒定在 80~87 ms——**目标越松也没用**，说明这不是"目标设得太紧"，而是压力信号的形状问题：

- 旧式 `1 − slack/target` = `observed/target`，**只要超过目标就直接饱和到 1.0**，超出多少都一样 → 配额立刻掉到地板；
- 地板的 256-token 碎片步又让实测 TPOT 保持高位 → 压力持续为 1 → 自锁（closed loop）。

修法两条，都做成配置项（默认开启）：

| 配置 | 作用 | 关键实测 |
|---|---|---|
| `tpot_prefill_throttle_margin=0.5` | 目标超出 50% 以内视为噪声（死区），超出后按超出量线性升压而非直接饱和 | 160 ms 目标：386.9 → **1238.8 tok/s**，TTFT p50 18.6 s → **2.5 s**，配额 258 → **2010 token** |
| `tpot_throttle_max_waiting=16` | 等待队列 ≥16 时**停用** TPOT 对 prefill 的压缩（此时瓶颈是 prefill 吞吐，继续压会自锁） | 20 ms 目标：350 → **1140 tok/s**，TTFT p50 44.5 s → **1.6 s**，TPOT p50 173 → **82 ms**，配额 256 → **1251 token** |

对照（证明不是"换个目标就好"）：`--tpot-prefill-throttle-margin 0` 复现旧行为时，**连 160 ms 目标也照样 collapse**（363.7 tok/s、TTFT 19.7 s、scale 0.007）——即旧曲线的自锁与目标是否可达无关。修复后同一 128 请求命令（`results/scheduling_ablation_128_r16_gated.json`，全部 128/128 唯一记录）：

| 变体 | 吞吐 tok/s | 相对 baseline | TTFT p50/p99 ms | TPOT p50 ms |
|---|---|---|---|---|
| baseline | 1858.6 | — | 477.6/730.4 | 41.8 |
| all_on | 1139.9 | -38.7% | 1598.0/12127.8 | 82.4 |
| without_tpot_aware | 1814.4 | -2.4% | 678.7/1190.0 | 42.3 |
| without_multi_step_decode | 1354.5 | -27.1% | 1336.4/7999.7 | 65.1 |

**边界（诚实标注）**：修复让"可达/略超"的目标不再塌方，但若目标远低于该负载下引擎能交付的 TPOT（这台卡 + 16 req/s 大约是 ~42 ms 无压缩、~82 ms 带节流），**TPOT 目标本身仍然不会被满足**（20 ms 目标实测 82 ms）——配额现在在 1251 token/步附近震荡而不是钉死在地板，因此是正确的折中而不是"达标"。`without_tpot_aware` 仍是 128 变体里最快的（1814 tok/s），所以"要不要开 TPOT 调度"取决于是否真的需要 decode 优先。
- **multi-step decode 在无 TPOT 目标时的收益很轻**：5 变体里 `without_multi_step_decode` 相对 `all_on` 吞吐 -3.1%、TTFT p99 +3.7%；两次运行符号一致但幅度在噪声量级，只能算"轻微正向"。
- **`without_decode_burst_yield` 在 mixed 下是近似空操作，但原因不是"没法做"，而是当时"没接上"**：让出条件是"burst 期间 waiting 非空 → 提前收尾"，mixed 模式只在两次 `engine.step()` 之间提交请求，burst 又只在 `scheduler.waiting` 为空时开始，所以**这个条件确实结构上恒不成立**（实测 `yields = 0 / 7021 bursts`，`rounds/burst = 4.00`）。真正能在 burst 中途拿到"新请求已到达"的路径是**到达压力信号**：服务端 `_sync_decode_burst_pressure()`（HTTP 请求进 `pending`/`incoming` 时置位）、trace 侧的 `set_decode_burst_yield_callback()`，以及 `decode_burst_yield_on_arrival` 开关。首次运行时这三者都缺失（见上方复跑修订），所以那一列必然为 0——修复后的服务端实测见 §10.3.11：**让出确实会在后续轮次发生**（8/8 轮），迟到请求尾延迟从 68.3 ms 降到 53.6 ms（-21%），最差轮 676 ms → 329 ms。- decode burst 计数随策略变化很大（baseline 0、all_on 7021 次 × 4.00 轮），说明多步复用在发生；TPOT 目标下的低吞吐不是多步造成的。
- `prefix_feature_cache` 命中率在 all_on 下只有 6.8~7.3%（parses 5254 / reuses 416），stale reparse 5126 次与 parses 同量级——与 §4.1 记录的"每步重新解析"一致，属已知成本，未单独消融。

#### 10.3.11 burst 中途到达的让出（2026-10-06，服务端实测）

**场景**：先发一条长输出请求（128 token，持续 decode → 反复进入 burst），`--late-delay-ms` 之后再发一条**极小**的迟到请求（5 句前缀 + 8 输出 token，自身只需 ~40 ms）。两条请求由两个线程并发提交，迟到请求在**服务端收到**时正处在 burst 中途——这是 offline trace 驱动复现不了的路径（trace 只能在两次 step 之间 add_request）。

命令：`python benchmarks/decode_burst_server_probe.py --base-url http://127.0.0.1:8000 --prefix-sentences 5 --first-output 128 --late-output 8 --late-delay-ms 300 --rounds 8`（服务端 `--max-decode-steps 8`）。

| 服务端配置 | arrival-yield 计数 | 迟到请求尾延迟 p50 | 均值 | 最差轮 | 备注 |
|---|---|---|---|---|---|
| `decode_burst_yield_on_arrival=True`（默认） | **8 / 8 轮** | **53.6 ms** | 110 ms | 329 ms | 每轮都命中，burst 在第 2 轮收尾 |
| `--no-decode-burst-yield-on-arrival` | 0 | 68.3 ms | 219 ms | 676 ms | burst 跑满 8 轮，迟到请求等完整个 burst |

**读数**：①让出机制在 burst 的**后续轮次**确实生效（8/8），迟到请求不必等整个 burst；②steady-state 差异 -21%（53.6 vs 68.3 ms），最差轮 -51%（329 vs 676 ms），前两轮的差距更大（含 JIT/图捕获）；③效应量级 ≈ `max_decode_steps × 单轮耗时`（8×~14 ms ≈ 110 ms 上界），所以 `max_decode_steps` 越大、单轮越慢，这个让出越值钱；④offline trace 驱动的消融表里这一列仍会是 0，因为那条路径无法在 step 内提交请求——**要衡量它必须走服务端并发提交**（本表）或以后加 "step 内可提交" 的驱动方式。


### 10.4 Profiling

**torch.profiler（CPU 侧；WSL 下 CUPTI 不可用，无 CUDA kernel 时间）**：
`profiler.py` 产出 `profiles/prefill.txt`/`decode.txt`——prefill：`aten::copy_`（锁页→GPU 输入
搬运）Self CPU 93%+；decode (eager)：`aten::mm` 22.96%、TorchDynamo Cache Lookup 7.01% +
Pregraph bytecode 4.63%（torch.compile 图查找开销）；decode (CUDA-graph)：`aten::copy_`（往 graph
静态输入拷贝）96.22%（CPU 开销集中在输入搬运，单次 replay 内部不可见）。
**CUDA 内核级统计（nsys/ncu）**：本机 CUPTI 不可用（torch.profiler CUDA activity 报
`CUPTI_ERROR_INVALID_DEVICE`、ncu 报 `ERR_NVGPUCTRPERM`）——只能拿 CPU 侧与 wall-clock；在
CUPTI 可用环境执行：

```bash
nsys profile -o /tmp/nanovllm_kernels -t cuda python benchmarks/bench.py --num-seqs 8 \
  --min-input-len 128 --max-input-len 128 --min-output-len 16 --max-output-len 16 --enforce-eager
nsys stats -r cuda_gpu_kern_sum /tmp/nanovllm_kernels.nsys-rep
ncu --set basic --launch-count 20 python benchmarks/bench.py --num-seqs 8 \
  --min-input-len 128 --max-input-len 128 --min-output-len 16 --max-output-len 16 --enforce-eager
```

### 10.5 读数要点与诚实规则

- decode 单 token 延迟由 KV cache 带宽决定（memory-bound）；prefill 由矩阵运算决定（compute-bound）。
- 重复批次 batch 1 的 prefill tokens 若不为 0，先查"缓存内容与 prompt 是否真的一致"
  （如 300-token 场景 2816 = 缓存块已被 decode 改写）。
- TPOT p99-p50 差距反映批大小波动/抢占影响；`preemptions > 0` 时所有延迟指标恶化。
- **基准的可信度来自"同 workload、同 seed、指标口径一致 + 探针可复现"**；随机语料类指标
  （ppl/生成文本）只比同批；所有数字带日期与环境条件。

## 11. 附录：合并映射与 2026-09-07 变更记录

### 11.1 旧文件 → 本档章节映射（已删除的 BENCHMARKS.md / LEARNING.md 内容去向）

| 旧位置 | 去向 |
|---|---|
| LEARNING.md 阶段 0-7 | §0/§1（1.1-1.7 对齐原阶段编号），工具箱 → §1.8 |
| LEARNING.md 面试对照（尾注） | §1 各表 + §4 深水区 + §6 故事 |
| INTERVIEW.md §0-§7 | §2（电梯）、§3（主线）、§4（深水区）、§5（速查）、§6（故事）、§7（方法论）、§8（路线图，原 §6）、§9（代码地图，原 §7） |
| BENCHMARKS.md 指标定义/快速开始 | §10.1 / §10.2 |
| BENCHMARKS.md §1-§6 | §10.3.1 / §10.3.2（过程性旧 run 数字已按"只保留现行结论"删除，保留演进结论） |
| BENCHMARKS.md §7 / §8 / §9+9b / §10 / §11 | §10.3.3 / §10.3.4 / §10.3.5 / §10.3.6 / §10.3.7 |
| note.md 故事编号 | §6 故事 1-18 同编号一一对应 |

### 11.2 2026-09-07 合并时的修正清单（保证正确性）

1. **代码行号全部实测校准**（2b/2b-ext 后 scheduler/block_manager/model_runner/attention 大幅
   位移，旧 §7 行号大面积失效；§9.1 为实测值）。
2. **ppl 口径统一**：三批互不一致的历史数字（fp16 3.32 / 3.60 / 3.63 等）实为同脚本不同随机
   续写语料——2026-09-07 复测 fp16 3.23 / fp8 3.25 / int4 4.17 / awq 3.70 为现行口径，并明示
   run 间波动。
3. **gemma2 int4 权重数字裁定**：`_quant_mem` 实测双路径 **7.45GB**、纯 int4 **3.40GB**——
   BENCHMARKS §11 原表"~4GB"是纯 int4 的数、双路径行标注错误；note.md 的 7.46GB 正确。
4. **gemma2 int4 dual 基准复测**（2026-09-07 ×2）：783-837 tok/s，比 08-24 首测慢 ~25-30%；
   两轮一致，原因未定位（环境/2b 路径改动），跨日期以新测为准。
5. **修复引擎回归（commit 本档提交）**：`_decide_streaming` 的 meta 计数构造污染 `get_rope`
   lru_cache 共享实例 → eager 量化路径（gemma2 int4 实测复现）首前向崩溃；加 `cache_clear()`。
   修复后 qwen3-0.6B int4 dual 实测 1.730GB 与文档一致（§6 故事 18）。
6. **pytest 计数更新**：41 → **57**（2026-09-07 实测）。
7. **陈旧状态更新**："SWA 不做滚动复用（TODO）" → 滚动环已实现（2b）并附边界；"fp8 KV×MLA
   断言关 / 纯 int4 MLA 兜底 eager" → 2b-ext 已解锁；streaming 限制补 MLA kv_b w_deq 例外。
8. 删除三合一前的重复段落（电梯陈述×2、故事×2、数字速查×2、学习路线×2、文件地图×2 等），
   保留各自角色定位下的最小重叠（速查表/深水区/基准档案分属"背数字/答追问/查证据"三种用途）。

### 11.3 相关文件

- `note.md`（未跟踪的个人时间线，截止 2026-08-24；§6 故事同编号，头部有指向本档的说明）
- `benchmarks/_stage2b_ext_report.md`（阶段 2b-ext 完整证据表 + 复现命令）
- `benchmarks/_kernel_roofline.md` / `_cuda_gemm_report.md`（阶段 1 交付物）
- 结果存档：`results/bench_*.json`、`results/compare_*.json`、`results/batch_scale.{csv,png}`
- 本文档内相对引用若失配，以 git 历史对应提交（`git log --oneline`）为准。

<!-- EOF -->
