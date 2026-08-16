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

# 2. 共享前缀 workload：测前缀缓存（前缀必须是整块数的倍数，如 256/512/768）
python benchmarks/bench.py --num-seqs 256 --shared-prefix-len 512

# 3. 前缀缓存跨批次演示：相同批次跑 3 遍，第 2/3 批 prefill 应大幅减少
python benchmarks/bench.py --num-seqs 64 --min-input-len 1024 --max-input-len 1024 --min-output-len 32 --max-output-len 32 --repeat-batches 3

# 4. 与真实 vLLM 对比（需先 pip install vllm）
python benchmarks/bench.py --num-seqs 256 --compare-vllm

# 5. 耗时分解（torch.profiler，prefill 与 decode 分开出表；CUDA 内核级统计见下文 nsys）
python benchmarks/profiler.py --num-seqs 64 --max-input-len 512 --max-output-len 64
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

**说明**：每序列 1024 token = 4 块，复用 3 块（768 token），最后一块仍被重算——`BlockManager.can_allocate` 按"最后一块未完成"设计跳过它（`range(num_blocks - 1)`）。因此前缀必须是整块（256）的倍数才能复用；期望的优化方向是 COW + 部分块共享（对齐 SGLang RadixAttention）。

### 4. 共享前缀 512（128 seqs，input 64-128 + 前缀 512）

批次内前缀块建立后，后续序列 prefill 只计算尾部：prefill 实际计算 **25,393 tok** vs 无缓存需算 77,824 tok（**67% 跳过**）。但 decode 因上下文更长（~600 vs ~100 token）单步变慢，E2E 不是衡量前缀缓存收益的正确视角——用上面的"重复批次"或"prefill 计算量"看。

### 5. vLLM 对比

待装 `pip install vllm` 后运行 `python benchmarks/bench.py --num-seqs 256 --compare-vllm` 填入。vLLM 无 Ray 时 `--tp` 保持 1。

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
- 共享前缀/重复批次的 prefill 计算量下降 = 前缀缓存命中。
- TPOT 的 p99 与 p50 差距反映批大小波动/抢占影响；`preemptions > 0` 时所有延迟指标都会恶化。
- SLO 的 TTFT<500ms 在离线批处理下通常难以达成。注意语义：`t_first_token` 在该序列 **prefill 完成的那次 postprocess** 记录（最后一个 prefill chunk 的前向已产出首个 completion token），因此 TTFT ≈ 该请求自身 prefill 完成时刻（含排队等前面 prefill 批次的时间），整批请求的 TTFT 呈阶梯分布（按落在哪个 prefill 步）。该指标更贴近 online serving 的语义。
