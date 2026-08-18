# BENCHMARKS

nano-vllm 的性能基准与 profiling 工具。所有脚本在 WSL2 conda 环境（`nano-vllm`，Python 3.12，RTX 5060 Ti 16GB）中运行。硬件/软件栈：Qwen3-0.6B，torch 2.8.0+cu128，flash-attn 2.8.3，CUDA-graph decode。

## 指标定义

| 指标 | 定义 |
|---|---|
| Throughput | 总输出 token 数 / 总耗时（wall time），tok/s |
| TTFT | 每个请求从提交（加入调度队列）到生成第一个 completion token 的时间 |
| TPOT | 每个请求 (完成时间 − 首token时间) / (输出token数 − 1)，即稳态解码的单token延迟 |
| E2E | 每个请求从提交到完成的端到端延迟 |
| SLO 达成率 | TTFT < 500ms 与 TPOT < 10ms 的请求占比（阈值可用 `--slo-*` 调整） |
| preemptions | KV cache 块不足时调度器抢占（回退重算）的次数，0 表示容量充足 |

计时插桩位于引擎内部：`Sequence.t_submitted / t_first_token / t_completed`（driver 侧，不跨进程传输），由 `LLMEngine.collect_metrics()` 统一导出，`benchmarks/bench.py` 负责统计。

## 快速开始

```bash
# 在 WSL 中激活环境后（或直接通过 benchmarks/run_in_wsl.sh 从 Windows 侧调用）
conda activate nano-vllm
cd /path/to/nano-vllm

# 1. 默认吞吐/延迟基准（256 seqs，随机 token，input 128-1024，output 64-512）
python benchmarks/bench.py --num-seqs 256

# 2. 共享前缀 workload：测前缀缓存（前缀长度不限，非整块也能命中部分块）
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512
python benchmarks/bench.py --num-seqs 64 --shared-prefix-len 300   # 非整块前缀 → 部分块共享 + COW

# 3. 前缀缓存跨批次演示：相同批次跑 3 遍，第 2/3 批 prefill 应大幅减少
python benchmarks/bench.py --num-seqs 64 --min-input-len 1024 --max-input-len 1024 --min-output-len 32 --max-output-len 32 --repeat-batches 3

# 4. 与真实 vLLM 对比（隔离环境 vllm-compare，见 §5；workload 一次生成两侧共享）
python benchmarks/compare_workload.py --tag small --num-seqs 128 --min-input-len 64 --max-input-len 128 --min-output-len 64 --max-output-len 128
python benchmarks/compare_nanovllm.py --workload results/compare_workload_small.json --kv-cache-dtype auto --output results/compare_nanovllm_small_fp16.json
python benchmarks/compare_vllm.py --workload results/compare_workload_small.json --kv-cache-dtype auto --output results/compare_vllm_small_fp16.json
python benchmarks/compare_merge.py results/compare_nanovllm_small_fp16.json results/compare_vllm_small_fp16.json

# 5. 耗时分解（torch.profiler，prefill 与 decode 分开出表；CUDA 内核级统计见下文 nsys）
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64

# 6. Batch 缩放实验（吞吐/TTFT/TPOT vs num_seqs，出图）
python benchmarks/batch_scale.py

# 7. 量化模式：FP8 KV cache（容量翻倍，自研Triton decode内核）
python benchmarks/bench.py --num-seqs 256 --kv-cache-dtype fp8_e4m3
# 8. W8A8 权重量化（int8 GEMM + SmoothQuant校准）与组合
python benchmarks/bench.py --num-seqs 256 --quantization w8a8
python benchmarks/bench.py --num-seqs 256 --quantization w8a8 --kv-cache-dtype fp8_e4m3
```

结果 JSON 输出到 `results/bench_<workload>_<ts>.json`，profiling 表输出到 `profiles/{prefill,decode}.txt`。

> 注意：conda 环境里的 editable install 可能指向另一个克隆（如 `~/AI/nano-vllm`），运行前确认 `python -c "import nanovllm; print(nanovllm.__file__)"` 指向你要测的目录；`benchmarks/run_in_wsl.sh` 已通过 `PYTHONPATH` 强制使用本工作区副本。

## 实测结果（2026-08-16，RTX 5060 Ti 16GB，Qwen3-0.6B）

KV cache 容量：**421 块 × 256 token = 107,776 token**（Qwen3-0.6B 单块 28MB = 2×28层×256×8 KV头×128 head_dim×2B）。

### 1. 干净 workload（256 seqs，input 64-256，output 32-128，容量内，CUDA-graph decode）

```
throughput (output)    5014.8 tok/s
prefill                41629 tok in 3 steps (31183 tok/s)
decode                 20469 tok in 127 steps (7339 tok/s)
preemptions            0
TTFT                   avg 921.6ms | p50 1060.7ms | p99 1335.1ms
TPOT                   avg 32.8ms  | p50 31.6ms  | p99 53.2ms
E2E                    avg 3.40s   | p50 3.54s   | p99 4.13s
```

JSON: `results/bench_n256_i64-256_o32-128_20260816T064401Z.json`

### 2. 超容 workload（256 seqs，input 128-1024，output 64-512，超出 KV 容量 → 抢占风暴）

```
throughput (output)    2287.5 tok/s   (vs 5014.8 干净 run，-54%)
prefill                173526 tok in 111 steps   (vs ~10 步预期)
decode                 75039 tok in 900 steps    (批量被抢占打散)
TTFT                   avg 7.95s | p50 2.60s | p99 24.4s
TPOT                   avg 49.7ms | p50 51.8ms | p99 85.1ms
```

**结论**：workload 总 token 需求（约 25 万）超过 KV 容量（10.8 万）时，调度器频繁抢占（每抢占一次释放块并回退重算），TTFT/TPOT 与吞吐全面恶化。这定量展示了 paged KV cache 容量规划的重要性；用 `--gpu-memory-utilization` 或缩短 output 可避免。

### 3. 前缀缓存：跨批次命中（64 seqs，1024-token 相同 prompt，跑 3 遍）

```
batch 0 (冷缓存): prefill 65536 tok in 4 steps | wall 3.12s
batch 1 (命中 75%): prefill 16384 tok in 1 step | wall 1.51s   (2.1x 加速)
batch 2 (再次命中): prefill 16384 tok in 1 step | wall 1.50s
```

JSON: `results/bench_n64_i1024-1024_o32-32_20260816T064940Z.json`

**说明**：本数据为阶段 1（末块复用）前的旧行为——每序列 1024 token = 4 块，只复用 3 块（768 token）。末块复用落地后（`can_allocate` 在末块满时也检查它），batch 1/2 的 prefill 降为 **0 tok / 0 步**（见 3.5）。本组数据保留作对照。

### 3.5 部分块共享 + COW（阶段 2）

`BlockManager` 现在会发布/复用**部分块**的哈希；写者要写共享的部分块时触发 **copy-on-write**（`cow_block`：CPU 换表记账 + `ModelRunner` 一次 `kv_cache[:, :, new] = kv_cache[:, :, old]` 全层复制），避免污染其他共享者。改动包含脏哈希守卫式删除（双胞胎块场景）与调度器 `num_tokens` 实际缓存长度记账。

（1）300-token 重复批次（64 seqs，含 44-token 部分块，decode-COW 路径）：

```
batch 0 (冷缓存): prefill 19200 tok in 2 steps | wall 1.06s
batch 1 (命中):   prefill 2816 tok  in 1 step  | wall 0.50s
```

JSON: `results/bench_n64_i300-300_o32-32_20260818T003053Z.json`

**2816 = 64×44 而不是 0，是正确语义而非缺陷**：batch 0 的 decode 把共享的 44-token 部分块写成了 76 token（44+32 个 completion），缓存内容已变；batch 1 的 prompt 只要求 44-token 内容，不匹配 → 只有满块 block 0 可复用。对比 1024-token 场景 batch 1 = 0，因为 4 个全是满块，completion 写进**新块**、旧块内容永不变。**前缀缓存只对"内容真正一致"的部分生效**——这就是"满块优先、部分块要 COW"设计合理性的直接证据。

（2）共享前缀 300 + 各自尾部（64 seqs，prefill-COW 路径）：

```
prefill 23618 tok in 2 steps (31415 tok/s) | wall 1.61s | 0 抢占
TTFT p50 506.8ms | TPOT p50 17.6ms
```

JSON: `results/bench_n64_i128-256_o32-64_prefix300_20260818T003114Z.json`

两条 COW 路径（decode 写共享末块 / prefill 尾部跨入共享块）均端到端跑通。附带收益：单测从 6 条增至 **8 条**（`test_copy_on_write`、`test_twin_hash_no_keyerror`），后者锁死了"双胞胎哈希 + 无条件 del → KeyError"的回归。

### 4. 共享前缀 512（128 seqs，input 64-128 + 前缀 512）

批次内前缀块建立后，后续序列 prefill 只计算尾部：prefill 实际计算 **25,393 tok** vs 无缓存需算 77,824 tok（**67% 跳过**）。但 decode 因上下文更长（~600 vs ~100 token）单步变慢，E2E 不是衡量前缀缓存收益的正确视角——用上面的"重复批次"或"prefill 计算量"看。

### 5. 与真实 vLLM 对比（2026-08-18，RTX 5060 Ti 16GB，Qwen3-0.6B）

**环境**：隔离 conda 环境 `vllm-compare`（`benchmarks/setup_vllm_compare.sh`），**不污染 nano-vllm 环境**。vLLM **0.10.2** + torch 2.8.0+cu128 + **同一份 flash-attn 2.8.3.post1 wheel**（保证注意力后端同源）；transformers 固定 4.57.6（5.x 与 vLLM 0.10 的 tokenizer 接口不兼容）；`XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`（xformers 0.0.32 的版本门禁拒绝 flash-attn>2.8.2，但我们故意两侧用同一构建）。两侧配置对齐：`gpu_memory_utilization=0.9`、`max_model_len=4096`、`max_num_batched_tokens=16384`、chunked prefill 开、CUDA graph 开、prefix caching 开、无 CPU offload（`swap_space=0`）。差异如实记录：**block size 256 vs 16**，KV 容量 nano 107,776 vs vLLM 97,440 token（vLLM 固定 16 字节/元素，容量受 `max_num_batched_tokens` 与图捕获峰值影响）。

**跑法**（三件套，workload 由 `compare_workload.py` 一次生成、两侧共享同一批 token id）：

```bash
# nano 侧（nano-vllm 环境）
python benchmarks/compare_nanovllm.py --workload results/compare_workload_small.json \
    --kv-cache-dtype auto --output results/compare_nanovllm_small_fp16.json
# vLLM 侧（vllm-compare 环境，见 benchmarks/run_vllm_compare.sh）
python benchmarks/compare_vllm.py --workload results/compare_workload_small.json \
    --kv-cache-dtype auto --output results/compare_vllm_small_fp16.json
# 合并出表
python benchmarks/compare_merge.py results/compare_*.json   # → results/compare_report.md/.csv
```

**指标口径（诚实声明）**：nano 侧为逐请求精确时间戳；vLLM 0.10.2 的 V1 引擎离线 API **不暴露逐请求指标**（`RequestOutput.metrics` 恒为 None，见 `vllm/v1/engine/output_processor.py`），改用 `LLM.get_metrics()` 的聚合直方图：**avg = sum/count 精确**，**p50/p99 = 桶内线性插值（近似）**。另有引擎级差异：nano 的调度器"先跑完全部 prefill 再 decode"，早期完成 prefill 的请求会空等（见下），因此 **decode 单步耗时（`benchmarks/_step_timing.py`）才是两侧可比的 kernel 级指标**。

#### 5.1 fp16 × fp16（同 dtype 直接对比，merge 报告原表）

| workload | 指标 | nano-vllm | vLLM 0.10.2 | nano/vllm |
|---|---|---|---|---|
| **small**（128 seqs，in 64-128，out 64-128，容量内） | throughput | **6587 tok/s** | 4624 | **1.42×** |
| | TTFT p50 / p99 | 353.1 / 353.3 ms | 372.0 / 497.4 | 0.95 / 0.71 |
| | TPOT p50 / p99 | **13.5 / 13.7 ms** | 17.4 / 24.9 | **0.78 / 0.55** |
| | E2E avg / p99 | **1.62 / 1.89 s** | 2.27 / 4.92 | **0.71 / 0.38** |
| **clean**（256 seqs，in 128-1024，out 64-512，超容 → 双方都抢占） | throughput | **2552 tok/s** | 1888 | **1.35×** |
| | TTFT p50 / p99 | **2520 / 21862 ms** | 3246 / 39034 | **0.78 / 0.56** |
| | TPOT p50 / p99 | **45.1 / 73.6 ms** | 56.0 / 149.2 | **0.81 / 0.49** |
| | preemptions | **68** | n/a（vLLM 不暴露） | — |

#### 5.2 长上下文（1024 in + 128 out，147k token 总上下文）：nano fp8 vs vLLM fp16

| 引擎 | kv dtype | 抢占 | throughput | TTFT p50/p99 | TPOT p50/p99（含等待口径） | decode 单步 p50 |
|---|---|---|---|---|---|---|
| nano-vllm | **fp8_e4m3** | **0** | **1854 tok/s** | 2734 / 4748 ms | 47.6 / 64.2 ms | 32.2 ms |
| nano-vllm | fp16 | **2** | **1421 tok/s** | 2512 / 8862 ms | 50.3 / 67.9 ms | 29.2 ms |
| vLLM 0.10.2 | fp16 | n/a | 1150 tok/s | 2500 / 19634 ms | 33.4 / 648 ms | ~33 ms（直方图 p50） |

**decode 单步耗时（引擎墙钟，`_step_timing.py`）——同口径对比**（v6 MMA 内核 + 饱和修复 + 混合调度后）：

| 场景 | nano fp16 | nano fp8 | vLLM fp16 |
|---|---|---|---|
| small（128 seqs，~100 tok） | — | — | ~17 ms |
| clean（256 seqs，128-1024 tok） | 36.4 ms | **35.0 ms** | ~56 ms（含抢占效应） |
| long（128 seqs，1024-1152 tok） | 29.2 ms | **32.2 ms** | ~33 ms |

#### 5.3 结论与诚实修正

1. **吞吐全面领先 1.35-1.61×**：small fp16 1.42×；clean fp8 **1.97×**（3715 vs 1888）；long fp8 **1.61×**（1854 vs 1150）；prefill 阶段快 ~2×；E2E 领先 ~25-35%。
2. **fp16 decode 单步与 vLLM 打平或略快**（long: 29.2 vs 33 ms；small: nano TPOT 13.5 vs 17.4 ms）。
3. **fp8 decode 单步已达 vLLM fp16 水平**：long 32.2 vs 33.4 ms（**打平**）、clean 35.0 vs ~56 ms（**1.6× 领先**，vLLM 的 56ms 含抢占效应）。此前的 1.6× 差距**不是 CUDA graph 问题**（fp8 内核本来就在图内，graph/eager 对照实验证实），而是内核效率：v6 用「直接 fp8 load（硬件 cvt 替代 LUT gather）+ MMA（QPAD=16 满足 dot 的 N≥16，8× 计算浪费换内存效率）」把逐层内核时间从 ~1.9ms 降到 ~1.15ms。大 tile（BLOCK_T 64/128）实验被数据否定（寄存器压力 1.1-13× 更慢）。**同时发现并修复了一个潜伏 bug**：torch 的 fp32→fp8 cast 溢出不饱和而是产生 NaN 位模式（实测 500→0x7F），v4 的 LUT 把 NaN 位模式读成 0.0 掩盖了它；写路径加 `clamp(-448, 448)` 后精度反而提升（KL 0.0077→0.0073，top-1 一致率 87.5%→100%）。
4. **调度缺陷已修复（混合批次，vLLM V1 同款）**：原"先跑完全部 prefill 再 decode"让早完成 prefill 的请求空等（long workload 中最早批次空等 ~4s）。现 `schedule()` 在 waiting 与 running 都非空时返回 **mixed 批次**（prefill 行在前 + decode 行在后，共享 `max_num_batched_tokens` 预算，一步内同时推进）；ModelRunner 新增 `prepare_mixed`，attention 混合路由（fp16：prefill 组 varlen + decode 组 flash_with_kvcache；fp8：prefill 组 varlen + 自研 fp8 内核；分块 prefill 序列经缓存形状 K/V 读自己上一 chunk）。**实测效果（所有 workload 吞吐上升、无一步额外开销）**：small fp16 +6.4%、clean fp16 +10.2%（抢占 85→68）、clean fp8 +2.9%、long fp16 **+11.3%（抢占 21→2，早完成请求的块提前释放）**、long fp8 +1.6%（TPOT 含等待口径 52.5→47.6ms）。早期请求的 decode token 从"等 ~4s"变为"下一步即出"（流式体验）。期间修了三个实现坑：Context 字段顺序（位置传参）、混合批次 LM head 取每序列末行、分块 prefill 序列在 varlen 中需缓存形状 K/V。**诚实说明**：span 口径的 TPOT 受"总工作量"下界约束，混合批次对它改善有限（long fp8 47.6 vs 52.5 主要来自抢占减少）；真正的收益是消除死等间隔 + 抢占压力下降，这两点不体现在 span 指标里（见 §5.2 decode 单步对照）。
5. **vLLM 0.10.2 在 RTX 5060 Ti（sm_120）上无法运行 FP8 KV cache**：V1 引擎不支持 `kv_cache_dtype`（自动回退 V0）；V0 的 fp8 路径选 XFormers 后端，而 xformers 0.0.32 把 fp8 注意力派发到 FA3（Hopper sm_90 专属内核）→ `CUDA error: invalid argument`；flashinfer 此环境无可用 wheel。vLLM 0.11-0.13 的 V1 fp8 也是 FA3 路线（`flash_attn_supports_fp8()` 要求 capability.major==9），同样无法在 sm_120 运行。**nano-vllm 的 fp8 KV + 自研 Triton 内核是这张卡上唯一可跑的 fp8 KV 实现**（容量 1.9×、零抢占、KL 0.0073）。
6. 所有数字的条件：单卡 WSL2、vLLM 0.10.2 V1（fp16）/V0（不可达）、flash-attn 2.8.3.post1、无 FlashInfer（vLLM 的 top-p 采样回落 PyTorch 原生实现）、WSL `pin_memory=False`。换版本/换环境后需重测。

### 6. Batch 缩放实验（吞吐-延迟权衡曲线）

`python benchmarks/batch_scale.py`（单引擎复用，每个 batch 独立 seed，CSV + PNG 输出到 `results/batch_scale.{csv,png}`；**下表为混合调度重测版**，旧调度器数据备份在 `results/batch_scale_old_scheduler.{csv,png}`）：

| num_seqs | throughput (tok/s) | TTFT p50 | TPOT p50 | preemptions |
|---|---|---|---|---|
| 16 | 1639 | 62.4ms | 5.16ms | 0 |
| 32 | 2857 | 135.0ms | 6.03ms | 0 |
| 64 | 4524 | 243.7ms | 7.80ms | 0 |
| 128 | 5325 | 414.2ms | 13.27ms | 0 |
| 256 | **5840（峰值）** | 845.0ms | 27.44ms | 0 |
| 384 | 5029 | 839.3ms | 50.09ms | 20 |
| 512 | 5245 | 1565.1ms | 54.08ms | 71 |

**vs 旧调度器（同 workload 同 seed 方案）**：吞吐全档提升 **+7.2% ~ +21.3%**（256 档 4854→5840，峰值仍在 256）；TTFT p50 全档下降（256 档 997.4→845.0ms）——早完成 prefill 的请求下一步即出 token，不再空等；抢占 384 档 31→20、512 档 **141→71（近半）**，对应 prefill 总 token 数下降（512 档 98716→89930，即重跑 prefill 的工作量减少）；TPOT p50 随抢占减少同步下降（256 档 33.07→27.44ms、512 档 68.97→54.08ms）。

**结论**：吞吐在 256 附近见顶——KV 容量（421 块）被 384+ seqs 的 workload 超出，抢占（20/71 次）侵蚀收益，但混合调度让 decode 提前释放块，容量压力明显缓解（512 档吞吐反超 384 档，旧调度器下 384 > 512 的"单调回落"形态消失）；TTFT 随 batch 近似线性增长，TPOT 反映单步 batch 增大。**吞吐-延迟权衡 + 容量悬崖（被调度缓解但未消除）**一张图讲完。512 档 TTFT p50（1565ms）比 384 档（839ms）跳升是单次运行噪声（p99 5462ms，长尾大），需多次重复取中位数才可信。

### 7. FP8 KV cache 量化（`--kv-cache-dtype fp8_e4m3`）

自研 Triton paged decode attention 内核直接读 FP8(E4M3) 缓存（v6：直接 fp8 load + 硬件 cvt 反量化 + MMA 计算），写路径用 warmup 校准的每层固定 scale 量化、**显式 clamp 到 ±448 饱和**（见下"潜伏 bug"）。`benchmarks/_fp8_kernel_check.py` 独立验证内核与 flash-attn/torch 参考一致（误差 = 量化噪声）。

- **容量**：421 → **802 块**（107,776 → 205,312 token，1.9×）
- **精度**（`benchmarks/accuracy_check.py`，首个 decode 步对齐 logits）：**KL=0.0073，top-1 一致 100%**，logits max diff 0.945——修复饱和 bug 后比旧数字（KL 0.0077 / top-1 87.5%）更好
- **内核优化历程**（独立逐层微基准 `_kernel_bench.py` + 变体实验 `_kernel_v6_exp.py`，RTX 5060 Ti）：
  - v1 位解码 3.2× flash → v2 LUT 查表 ~1.95× → v3 MMA pad 失败（M=2<16）→ v4 LUT+GQA 融合 + `[G,T]/[G,T,D]` 广播：seqlen=160 1.26×、256 打平、1024 0.85×（逐层口径）
  - v5 直接 fp8 load（Triton cvt 替代 LUT gather，去掉每元素两次 gather）：全面 0.77-0.86×（v4 的 LUT 是短/长上下文共同瓶颈）
  - **v6 MMA（现役）**：QPAD=16 满足 `tl.dot` 的 N≥16（GQA 组 G=2 不够），8× 计算浪费换内存效率；`s=dot(k16,q16)` / `acc=dot(v_t^T,p16)`；BLOCK_T=32/warps=1；**再快 ~25%**（vs v5 0.74-0.71×，vs v4 0.53×）
  - **大 tile 假设被数据否定**：BLOCK_T∈{64,128} 因寄存器压力 1.1-13× 更慢（v3 失败同因）
- **引擎级**（decode 单步，`_step_timing.py`）：long **54.8 → 45.1 → 32.2 ms**；clean **54.4 → 47.9 → 35.0 ms**。长上下文 fp8 已达 vLLM fp16 水平（§5.3）；吞吐 clean 2492 → 3612、long 1400 → 1825 tok/s
- **潜伏 bug 修复（重要）**：torch 的 fp32→fp8 cast **溢出不饱和而是产生 NaN 位模式**（实测 500→0x7F、1e30→0x7F）；写路径 `(k*inv_scale).to(fp8)` 在校准数据外的真实激活超 448 时写入 NaN → v5/v6 直接 cvt 会把 NaN 传播到 logits（v4 的 LUT 把 e=15/m=7 读成 0.0 掩盖了它——旧 KL 0.0077 其实含"溢出静默归零"畸变）。修复：写路径 `clamp(-448, 448)` 饱和。教训：**位模式解码 vs 硬件 cvt 在 NaN 语义上不等价，内核换解码方式必须重跑引擎级精度检查**
- **诚实记录**：短上下文（<256 token）仍有 ~15-25% 差距（内核开销未摊薄）；长上下文已打平 vLLM fp16（32.2 vs 33 ms）；split-K/持久内核仍是进一步路线

### 8. W8A8 权重量化（`--quantization w8a8`）

**per-group（K 维 128 分组，AWQ 标准）** int8 权重 + per-token int8 激活 + 自研 Triton int8 GEMM（int32 块内累加、乘组 scale 后 fp32 跨组累加，`benchmarks/_w8a8_check.py` 独立验证 per-group 与 per-channel 两种模式）；SmoothQuant 式校准（激活逐通道 amax → s 向量折进权重 `W'=W·s, X'=X/s`，恒等变换压平激活离群值）。内核 `group = (k//BLOCK_K) % num_groups` 处理 num_groups=1（per-channel）特例防越界。

- **精度**（w8a8 + fp16 KV）：**KL=0.064 → 0.0379（-41%），top-1 一致率 87.5% → 100%**——per-group 把权重 scale 粒度从整行（K=1024）细化到 128 一组（8×），精度缺口的主体被修复；残余误差来自激活的 per-token 量化（embedding 离群值，SmoothQuant 只压平到一定程度）
- **性能**（256 seqs，in 64-256 / out 32-128）：5382 → 4497 tok/s（**-16.4%**，int8 GEMM 未调优 + 激活量化/平滑开销）；权重显存减半（释放 fp16 权重）
- **内核实现要点**：`BLOCK_K=128=GROUP`，每个 K 块恰好一组，块内 int8 MMA→int32 后乘本组 scale 以 fp32 跨组累加（组间 scale 不同，不能像 per-channel 那样最后统一乘）
- **路线图**：GPTQ 式舍入（量化误差反馈补偿）是进一步的精度手段；int8 GEMM 调优（tile/warp 扫描）可收回吞吐损失

### 9. 投机解码——n-gram / prompt-lookup（`--speculative ngram`）

**思路**：草稿源 = 序列自身历史（vLLM PromptLookupWorker 同款，`--speculative-model "[ngram]"`）。取末尾 w 个 token 作窗口（w: 4→1 递减回退），在历史中找**严格位于当前窗口之前**的最近一次出现，把"上次出现之后紧跟着的 token"抄出来当草稿（γ ≤ 4，EOS 前截断，按剩余输出预算封顶）。无模型、零显存、零训练——草稿的唯一成本是 CPU 搜索。

**核心架构决策：verify 步 = 带前缀复用的 varlen prefill 步**。每序列 query = `[末token, 草稿...]`（γ+1 个 token，位置从 len-1 起），num_cached = len-1，直接复用已有的"分块 prefill + 前缀复用"路径（缓存形状 K/V + block_tables + cu_seqlens）。logits 语义：位置 len-1+i 预测位置 len+i → 样本 s_i 验证草稿 d_i；最后一行是全接受时的 bonus。**接受规则（Leviathan et al. 2023）**：草稿是点质量分布，"接受 iff 目标采样 == 草稿，拒绝则输出该采样"严格保持目标分布——每位置输出分布与不做投机时逐 token 相等。**KV 提交语义**：verify 写 span [len-1, len+γ) 含被拒草稿的槽位——不回滚（下一步覆盖），只截断逻辑长度；**前缀缓存哈希只发布到接受长度**（被拒 token 永不进 hash；哈希范围 `[num_tokens-n_acc-1, num_tokens-1)`）。COW 覆盖跨块写 span（`can_append_spec/may_append_spec`）；混合批次下 verify 行与 prefill 行同走全批次 varlen（LM head：prefill 组取末行 + verify 组保留全行）。全部无草稿时回落纯 decode（CUDA graph 路径不受影响）。实现：`nanovllm/engine/ngram.py`（草稿搜索 + 验收纯函数）、`scheduler.py`（spec 调度 + postprocess_spec）、`model_runner.py`（prepare_spec / _prepare_mixed_spec）。

**正确性验证**（`benchmarks/_spec_equiv_check.py`，三层）：
1. **verify 前向 logits 对齐**（决定性）：同 seed 同流跑 plain/spec（单 prompt），首个 spec 步的**全部 γ+1 行** logits 与 plain 逐 decode 步对齐（迭代重建 seq 状态，取全接受步）——7 个 prompt 全过，max|Δlogit| = 0.12~0.44（varlen vs flash-kvcache 内核级噪声），top-1 一致 100%，fp16/fp8 双路径。
2. **分布一致性**（统计）：temp=0.6 下 spec 与 plain 自一致率同量级（弱检验，小样本噪声大，正确性以第 1 层为准）。temp=1e-4 诊断：plain 自翻转 1.4% vs spec 翻转 3.8%（spec 多出的是 verify/decode 内核噪声在近并列位置的翻转——验收逻辑本身精确）。
3. 冒烟：temp=0.6 批量跑通、α 报告、fp8 路径正确。
4. 纯 Python 单测 `tests/test_spec_decode.py`（草稿搜索/验收/调度/跨块 COW/哈希范围），pytest 30/30。

**性能（`benchmarks/spec_bench.py`，Qwen3-0.6B，三种风格 × spec on/off；**下表为 verify CUDA graph 优化后的最终版**）：**

| 风格 (bs=256) | α | avg γ | 吞吐 baseline→spec | TPOT p50 |
|---|---|---|---|---|
| free（随机token） | 0.46 | 1.7 | 3553→3024（**0.85×**） | 48.0→41.8ms（**-13%**） |
| json（结构化续写） | 0.26 | 2.2 | 3246→2067（**0.64×**） | 72.3→104.2ms |
| repeat（echo 5s，最好情况） | 0.99 | 3.9 | 7749→**11076（+1.43× 赢）** | 27.5→**15.5ms（-44%）** |

| 风格 (bs=8) | α | 吞吐 baseline→spec | TPOT p50 |
|---|---|---|---|
| repeat（echo 5s） | 1.0 | 1435→**5561（+3.87× 大赢）** | 5.2→**1.2ms（-77%）** |
| json | 0.42 | 856→**982（+1.15× 赢）** | 7.2→**3.7ms（-49%）** |
| free | 0.34 | 1180→**1236（+1.05× 打平转赢）** | 5.1→5.1ms |

**verify 路径优化（本项工作，为什么从"打平"变"赢"）**——定位过程（`benchmarks/_verify_probe.py` + `_spec_step_timing.py`）：
1. **探针分离变量**：分页 K/V vs 连续 K/V 只差 6%（**分页不是瓶颈**）；短查询（5 tok）每 token 47µs vs 长查询 prefill 27µs（只贵 1.7×，不是初测的 7.5×——v1 的数字被首次调用的 torch.compile 污染）；draining 形状变化不掉效率。
2. **步级计时定位真凶**：spec 步墙钟 38ms = GPU ~25ms + **CPU 逐 kernel 启动税 ~10ms**（~300 次 eager launch，小步时无法被 GPU 时间隐藏）。baseline 的 CUDA graph 重放 CPU 只需 0.63ms。
3. **修复：verify 前向也捕获 CUDA graph**（`capture_spec_graph`）：行容量族 [8..256] × 双 stride（5 覆盖任意 γ；3 覆盖低 γ 步，容量=3×行数减少填充浪费），捕获时全行满长度（flash varlen 的 grid 按容量烘焙），重放时真实行 + **零长度填充行**（cu_seqlens 尾部重复末值 → flash 按空行跳过，`_graph_pad_probe.py` 验证 **bit-exact**）；`max_seqlen_q=γ+1`/`max_seqlen_k=max_model_len` 作为标量参数捕获时固定（key 循环由 cu_seqlens 驱动，烘焙上限无开销，probe 验证）；LM head 在图外对真实行切片计算（省掉填充行的词表 GEMM）。
4. **效果**：repeat@bs=256 从 1.00× 打平 → **1.43×**；bs=8 从 1.40× → **3.87×**（TPOT 1.2ms）；低 γ 的 free/json 小 batch 从 0.44×/0.53× 翻到 1.05×/1.15×（填充浪费被 stride-3 家族压住）。大 batch 低 α 场景（json@256，α=0.26）仍亏损——那是草稿质量问题，不是 verify 路径问题。

**诚实结论**：
1. **verify 路径本身效率没问题**（47µs/tok vs decode 110µs/tok，2.3× 更优）；初版打平的真相是 **eager 启动税**。CUDA graph 化后 **α≈1 时全面赢**（bs=256 +43%、bs=8 +3.87×），延迟（TPOT）大改善（-44% ~ -77%）。
2. **剩余亏损场景 = 草稿质量低**（α < 0.4 时 γ 个草稿大半浪费）：0.6B 模型在自由文本/非 echo 结构上的续写重复度低。这是草稿源的问题（Medusa 头的用武之地），verify 路径已就绪。
3. **fp8 KV + spec 仍不可用（0.15×）**：fp8 的前缀复用 prefill 路径每层每次**全缓存反量化**（`k_cache.to(k.dtype)*k_scale`，802 块 × 28 层 ≈ 18GB/步 内存搬运）——这是 GPU 侧成本，graph 也救不了。修复需自研 fp8 varlen 内核（或缓存镜像，但会抵消 fp8 的容量收益）。正确性不受影响（fp8 对齐全过）。
4. **对路线图的含义（更新）**：Medusa 的 verify 形状与 n-gram 相同，**verify 路径已 graph 化、直接受益**；下一步瓶颈转移到草稿质量（Medusa 头）+ fp8 verify 内核。
5. 条件：WSL2 单卡、flash-attn 2.8.3、bs=8/256、Qwen3-0.6B（0.6B 模型的续写重复度低，α 天然偏低；大模型 + 结构化内容 α 更高，结论会右移）。vLLM 同款 n-gram 的对照测试（vllm-compare 环境）留作后续。

## Profiling

### torch.profiler（CPU 侧，WSL 下 CUPTI 不可用，无 CUDA kernel 时间）

`python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64` 产出 `profiles/prefill.txt` / `decode.txt`：

- **prefill**：`aten::copy_`（锁页内存 → GPU 的输入搬运）占 Self CPU 93%+，其次 `aten::mm`（GEMM）~1%，`TorchDynamo Cache Lookup` ~0.5%——CPU 侧瓶颈是数据搬运而非计算。
- **decode (eager)**：`aten::mm` 22.96%（2260 次调用，GEMM 总耗时最高）、`TorchDynamo Cache Lookup` 7.01% + `Pregraph bytecode` 4.63%（torch.compile 图查找开销）、triton kernel（RMSNorm 等）6.48%——decode 的 CPU 开销分散，GEMM 与框架层开销明显。
- **decode (CUDA-graph)**：`aten::copy_`（往 graph 静态输入张量拷贝）96.22%，单次 graph replay 内部不可见——即 graph 路径的 CPU 开销集中在输入搬运。

### CUDA 内核级统计（nsys / ncu）

本机的 WSL 环境 **CUPTI 不可用**（torch.profiler CUDA activity 报 `CUPTI_ERROR_INVALID_DEVICE`，ncu 报 `ERR_NVGPUCTRPERM`），因此当前只能拿到 CPU 侧调用表与 wall-clock 数据。在 CUPTI 可用的环境（Windows 原生运行、或配置好 WSL 的 GPU 权限/驱动）中执行：

```bash
# nsys：内核耗时汇总
nsys profile -o /tmp/nanovllm_kernels -t cuda python benchmarks/bench.py --num-seqs 8 \
  --min-input-len 128 --max-input-len 128 --min-output-len 16 --max-output-len 16 --enforce-eager
nsys stats -r cuda_gpu_kern_sum /tmp/nanovllm_kernels.nsys-rep

# ncu：按内核看吞吐/占用（需性能计数器权限，如 sudo 或设置 NVreg_RestrictProfilingToAdminUsers=0）
ncu --set basic --launch-count 20 python benchmarks/bench.py --num-seqs 8 \
  --min-input-len 128 --max-input-len 128 --min-output-len 16 --max-output-len 16 --enforce-eager
```

## 阅读结果时关注什么

- decode 单 token 延迟由 KV cache 带宽决定（memory-bound）；prefill 由矩阵运算决定（compute-bound）。
- 共享前缀/重复批次的 prefill 计算量下降 = 前缀缓存命中；部分块命中会触发 COW（一次 28MB 的全层复制），命中收益小于 COW 开销时不划算——这就是"满块优先共享"的取舍。
- 重复批次中 batch 1 的 prefill tokens 若不为 0，先查"缓存内容与 prompt 是否真的一致"（如 300-token 场景的 2816 = 缓存块已被 decode 改写）。
- TPOT 的 p99 与 p50 差距反映批大小波动/抢占影响；`preemptions > 0` 时所有延迟指标都会恶化。
- SLO 的 TTFT<500ms 在离线批处理下通常难以达成。注意语义：`t_first_token` 在该序列 **prefill 完成的那次 postprocess** 记录（最后一个 prefill chunk 的前向已产出首个 completion token），因此 TTFT ≈ 该请求自身 prefill 完成时刻（含排队等前面 prefill 批次的时间），整批请求的 TTFT 呈阶梯分布（按落在哪个 prefill 步）。该指标更贴近 online serving 的语义。
