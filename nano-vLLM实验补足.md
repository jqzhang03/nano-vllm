# nano\-vLLM实验补足



# nano\-vLLM：RTX 5060 Ti 16GB 单卡补充实验方案

目标：把项目中目前缺少真实请求负载、准入与 SLO 阈值未校准、KV 容量与并发边界不清、算子收益波动、投机解码效果随内容变化等问题，转成可复现的实验。所有数值都应由这轮实验实测后再写入简历；本文中的历史数据只作实验起点，不作为新实验结论。

## 本轮执行结果与结论（2026-10-15）

本轮已按方案补做调度、准入、KV、INT4 MoE 和 n-gram 投机解码实验。这里记录实际跑过的条件；真实 ShareGPT prompt 配合的是合成 Poisson 到达过程，不是生产请求日志。原始 JSON/CSV 留在本机被 Git 忽略的 `results/supplemental/`，汇总与图表见 [`benchmarks/_supplemental_report.md`](benchmarks/_supplemental_report.md) 和 [`benchmarks/_supplemental_figures/`](benchmarks/_supplemental_figures/)。

- **真实 prompt 与调度**：清理后的 500 条 ShareGPT 用户 prompt，8 req/s、5 个配对 seed。trace 中的 `max_tokens` 由对应 assistant 回复 token 长度截断生成，参考回复文本不作为模型输入；到达时间由 Poisson 合成，没有线上到达日志。seed0 结果的 500 个 request ID、输出预算和到达时间均与生成 trace 匹配。`all_on` 相对 baseline 的输出吞吐均值下降 6.50%（配对 bootstrap 95% CI：-7.99%～-4.52%），联合 SLO 输出 goodput 均值下降 59.09%（-68.30%～-49.89%）；两项都在 5/5 个 seed 下降。当前结论不支持默认启用整组策略。
- **准入**：完成 400 请求、96 req/s 的 12 点 target/work 网格（TPOT≤20 ms），以及 24 点 soft-pressure/max-wait 细化（TPOT≤80 ms）；两阶段 SLO 不同，结果不能合成一条 Pareto 曲线。候选 `500/5000/1.0/500` 在 3-seed、TPOT≤80 ms 校准中位拒绝率/goodput 为 58.2%/4458 token/s，admission-off 为 33.8%/5217 token/s；TPOT≤20 ms 时候选 goodput 为 1607 token/s、admission-off 为 0，TPOT≤40/80/120 ms 时候选均低于 admission-off。Held-out 只有 100 条 prompt、384 req/s、2 个 seed，与 calibration 负载不同；该候选中位 goodput 为 3774 对 4507 token/s，拒绝率 16.5% 对 0%。没有产品拒绝率预算，因此不推荐新的默认阈值。
- **KV 容量与 swap**：当前 Qwen 配置的 synthetic pool 实测 BF16 为 425 blocks / 108,800 token slots，FP8 E4M3 为 835 / 213,760；这只是该配置的容量账本，没有测到 OOM 或物理容量极限。默认成本策略下观察到的 2 次抢占都选择重算。关闭重算成本比较后做的 3 次路径诊断中，swap-on 为 470.2 token/s（12 次 swap），no-swap 为 511.4 token/s（0 次 swap），不能据此宣称 swap 带来性能收益。FP8 held-out 比较仅覆盖 16 条 prompt 的首个 decode logits，不代表长文本质量。
- **INT4 MoE**：DeepSeek-V2-Lite、TP=1、group64 的 split/fused 在 26 层 × 2 seed × 5 row count 中都对 loop bitwise 匹配（每种实现 260/260）；microbenchmark fused 为 11.90×～42.95×，仅代表 kernel 局部。36 组在线初筛每组一次，吞吐中位变化 +0.91%（-14.94%～+23.07%），只有 19/36 组输出完全一致。另一个 8 req/s、128 输出、100 ms TPOT 条件完成 5 seeds × 5 repeats：吞吐中位变化 +0.51%（-12.87%～+16.46%，均值 95% CI -0.35%～+5.11%），输出完全一致 14/25 组。该重复只覆盖一个 workload，仍不证明稳定服务收益，fusion 保持实验选项。MLA query range 修正后的复现仍有 1/16 请求输出不同。
- **投机解码**：完成 n-gram 的内容分层与 batch/γ 扩展（报告共 102 次运行，固定 96-token 输出）。重复文本接受率高于 ShareGPT、开放问答和结构化内容；α 与样本中的 4-gram 重复率相关（Spearman ρ=0.929），不是对总体请求的因果结论。没有测试自然停止，也没有可用的 Medusa/EAGLE draft 权重。

因此，本轮交付的是按上述 workload 可复核的实测结果，而不是对全部计划维度的通用结论。后续若要推荐准入阈值，仍需业务拒绝率预算和匹配 calibration/held-out 负载；若要验证 serving 优化，仍需在更多 MoE workload 上重复配对，并取得真实到达日志及可用的 Medusa/EAGLE 权重。

## 0\. 统一实验口径与环境

- 硬件基线：单张 RTX 5060 Ti 16GB，TP=1；优先使用仓库已覆盖的 Qwen3\-0\.6B BF16 跑调度、准入、容量和投机实验。MoE 算子实验使用仓库现有的 DeepSeek\-V2\-Lite INT4 group64 配置。切换模型、量化或 KV dtype 后要重新测量，不能把一个模型的结论外推到另一个模型。

- 记录软件与硬件状态：仓库 commit、Python/PyTorch/CUDA/Triton/flash\-attn 版本、GPU 型号、显存、GPU 温度/核心时钟、模型路径和量化配置；确认运行的 nanovllm 源码来自当前 checkout。实验期间关闭其他 GPU 任务，不锁频、不改功耗参数。

- 成对对照：baseline 与优化版本使用完全相同的 prompt token IDs、输出上限、到达时间、采样配置和 seed；保存 workload fingerprint。两边交替运行顺序，避免温度和时钟漂移总偏向某一组。

- 预热与重复：每个配置先预热；探索阶段至少 3 个 seed，候选结论至少 5 个配对重复。冷启动/CUDA Graph 首次 capture 和稳定态分开计时。每次开始前清空或重置指标与 prefix KV，避免前一轮缓存污染后一轮。

- 统一报告：请求数、输入/输出 token 分布、到达模式、TTFT/TPOT/E2E 的 p50/p95/p99、输出 token/s、接收率/拒绝率、KV 抢占与 swap、GPU 峰值显存。SLO goodput 定义为同时满足 TTFT 与 TPOT 目标的请求所生成 token 数除以测试墙钟时间；同时报告请求 goodput 和被接收请求比例，避免只看被服务请求的尾延迟。

- 原始数据保存在 JSON，汇总表另存 CSV；记录每次命令、配置、seed 和 fingerprint。对有波动的优化用配对差值和 bootstrap 95% 置信区间报告，不挑单次最好结果。

## 1\. 补真实请求负载：从合成 prompt 到 ShareGPT trace

### 已完成 trace loader 与剩余边界

**bench\.py**、**scheduling\_ablation\.py** 和 **admission\_ablation\.py** 已支持 JSONL request trace，包括 `request_id`、`max_tokens`、`arrival_time_s`、prefix 元数据与请求级 SLO；prompt-only 格式仍兼容。**sharegpt\_trace.py** 生成脱敏 ShareGPT trace，使用参考 assistant 回复 token 长度派生输出预算但不保存回复文本，按 Poisson 或 constant 等模式合成到达；**derive\_trace\_variants.py** 派生 smoke、calibration、held-out 和共享前缀变体。当前配对调度/准入筛查使用 Poisson；共享前缀变体采用合成的 256-token 公共模板，没有使用真实 system prompt。当前没有生产到达日志，因此不能外推真实到达间隔分布。

### 负载准备

1. 从 ShareGPT/LMSYS 公开数据中抽取 100 条作为冒烟集，再准备 300–500 条用于正式统计。每条只保留用户输入；去除个人信息、超长异常样本和空请求。将原始 assistant 回复长度仅作为输出预算的来源，不把参考回复直接喂给模型。

2. 按 tokenizer 统计 prompt 长度，分桶为 1–256、257–512、513–1024、1025–2048 tokens；正式 workload 尽量覆盖真实比例，同时保留每个长度桶单独跑的结果，避免总体均值掩盖长 prompt。

3. 已生成 request trace JSONL，包含 **request\_id**、**prompt**、**max\_tokens**、**arrival\_time\_s**、**prefix\_group**、**ttft\_slo\_ms**、**tpot\_slo\_ms** 和 tokenized prompt hash。workload loader 已支持这些字段，并保留老格式 **\{"prompt":"\.\.\."\}** 兼容。

4. trace builder 和 benchmark 支持 burst、constant（2/4/8/16 req/s）及 Poisson。当前 ShareGPT 调度/准入配对报告使用 Poisson；没有在相同配对矩阵中覆盖 burst/constant，也没有线上到达日志，结论表述为“真实 prompt + 合成到达过程”。

5. 已完成 0%/20%/50% 共享前缀准入对照；20%/50% 条件使用合成的 256-token 公共模板，没有验证真实 system/template 前缀。合成前缀用于单独验证 prefix cache，不能与真实公共前缀结果混报。

### 执行与对照

已完成 prompt-only 输入和 request trace 两种路径。正式筛查使用保存的请求 ID、ShareGPT prompt 与参考回复派生的 `max_tokens` 预算；到达时间使用可复现的 Poisson 合成 trace。所有策略在同一 workload fingerprint 上配对，输入输出预算固定，到达过程不代表线上日志。

```bash
mkdir -p results/supplemental
MODEL_DIR=~/huggingface/Qwen3-0.6B
python benchmarks/scheduling_ablation.py --model "$MODEL_DIR" --prompts-file datasets/sharegpt_100.jsonl --num-seqs 100 --arrival-mode poisson --arrival-rate 8 --variants baseline all_on --seed 0 --output results/supplemental/sharegpt_poisson_8rps_seed0.json
```

- 初始对照用 **baseline**、**all\_on**；之后针对差异最大的策略做单项消融，如 **without\_tpot\_aware**、**without\_slo\_adaptive**、**without\_multi\_step\_decode**。

- 每种到达模式至少跑 3 个 seed；正式结果增加到 5 个 seed。固定同一 trace fingerprint，核对每个变体都完成或明确记录全部 100% 请求，避免重复 request 记录造成提前结束。

### 观察指标与判定

- 按长度桶报告 TTFT/TPOT/E2E 分位数、输出 token/s、抢占数、队列等待时间和 SLO goodput；对共享前缀场景额外报告 prefix hit tokens/requests。

- 对比 prompt\-only 合成输出长度与 trace replay 的差异，说明“真实 prompt”本身不足以代表真实 serving 负载，输出长度和到达过程也需要匹配。

- 交付物：脱敏 JSONL trace、trace builder、各变体 JSON/CSV、fingerprint 校验结果、分长度和分到达率图表。

## 2\. 校准准入策略与 TTFT/TPOT SLO

### 实验目标与起点

目标是在给定拒绝率/延迟要求下找到合理的 admission 参数，而不是只让接收请求的 p99 变好。已有 24 req/s、128 请求过载结果显示：无准入吞吐约 1611 tok/s、TTFT 目标达标率 26\.6%；cache\-aware 准入接受率 15\.6%、接受请求 TTFT 达标率 95%、吞吐约 777 tok/s。这说明需要同时展示吞吐、达标率和拒绝代价，且 2 s 准入目标仍需按 workload 校准。

### 校准步骤

1. 使用第 1 节的 ShareGPT trace，先关闭准入在 2/4/8/12/16/24 req/s 上做容量扫描；找到联合 SLO 开始明显失守的 offered rate。低于饱和点且 admission 从不触发的结果只作为正确性对照，不用于选阈值。

2. 选“刚开始失守”和“明显过载”两档压力，固定同一请求 trace，对比 **dynamic\_off**、**dynamic\_on\_prefix\_cache\_unaware**、**dynamic\_on**。测试 0%/20%/50% 共享前缀，区分准入估算本身和 prefix\-cache 感知的影响。

3. 分两阶段扫参数以控制成本：先扫 **admission\_target\_ttft\_ms** 500/1000/2000/3000 ms 与 **admission\_work\_budget\_ms** 2000/5000/10000 ms；再围绕表现最好的两组扫 **admission\_soft\_pressure** 0\.5/0\.7/0\.85/1\.0 和 **max\_admission\_wait\_ms** 500/2000/5000 ms。TPOT SLO 用 20/40/80/120 ms 形成可行性曲线，不能只挑容易达到的目标。

4. 用 seed 0–2 做调参，seed 3–4 和未见过的 prompt 子集做 held\-out 验证。准入参数在 validation 集上不再继续调。

### 运行示例

```bash
MODEL_DIR=~/huggingface/Qwen3-0.6B
python benchmarks/admission_ablation.py --model "$MODEL_DIR" --prompts-file datasets/sharegpt_100.jsonl --num-seqs 100 --arrival-mode poisson --arrival-rate 16 --shared-prefix-len 256 --slo-ttft-ms 2000 --slo-tpot-ms 80 --admission-target-ttft-ms 2000 --admission-work-budget-ms 5000 --admission-soft-pressure 0.85 --max-admission-wait-ms 2000 --seed 0 --output results/supplemental/admission_16rps_seed0.json
```

示例只表示命令入口；正式矩阵需分别改变到达率、SLO 和 admission 参数，并在不同 seed 上复跑。

### 指标与选点

- 报告 arrived/accepted/deferred/rejected 数、接受率、拒绝原因、准入等待 p50/p95、预测 TTFT、实际 TTFT/TPOT/E2E、接受请求 SLO 达成率、全到达请求 SLO goodput、输出 token/s、prefix 估算误差和 KV 压力。

- 先画“请求拒绝率—联合 SLO goodput—p99 TTFT”的 Pareto 曲线。若没有业务给定的最大拒绝率，不宣称某一组参数是最优默认值；展示可选 operating points，并把 held\-out 结果作为校准有效性的证据。

- 交付物：参数 sweep 表、Pareto 图、held\-out 验证结果、推荐参数及其适用负载范围。

## 3\. 测 KV 容量与并发边界

### 实验矩阵

**context\_concurrency\.py** 已能在每个并发/上下文点读取实际 KV tensor、block pool 容量、GPU 峰值显存、请求延迟与抢占；它当前使用确定性 synthetic token IDs，适合先测容量，再用第 1 节的真实 prompt 验证性能边界。历史 Qwen3\-0\.6B BF16 配置记录约 421 个 KV blocks，约 108K token slots；重新以当前模型、版本和配置实测为准。

- 上下文长度：512、1024、2048、4096 tokens；模型位置编码与显存允许时再加 8192。每条请求保留 32 或 64 个输出 token。

- 并发：1、2、4、8、16、32，容量尚有余量时再加 48、64。每点 3 次 burst；接近边界的点做 5 次。

- 配置：先测 BF16 weights \+ auto KV，关闭 KV swap 得到 GPU KV pool 的物理边界；再测 FP8 KV。FP8 先用真实文本校准 scale，并以 held\-out prompt 做首个 decode logits/KL/top\-1 检查。最后单独开启 swap，报告 host offload 的服务边界，不把 swap 容量记作 GPU KV 容量。

### 运行命令

```bash
MODEL_DIR=~/huggingface/Qwen3-0.6B
python benchmarks/context_concurrency.py --model "$MODEL_DIR" --context-lengths 512,1024,2048,4096 --concurrency 1,2,4,8,16,32 --repeats 3 --max-output-tokens 64 --no-swap-kv --output results/supplemental/kv_auto_no_swap.json
```

达到最大上下文后，另跑 **\-\-kv\-cache\-dtype fp8\_e4m3** 并提供校准文件；再去掉 **\-\-no\-swap\-kv** 单独测试 swap。不要把三种配置混在同一张“最大容量”数字里。

### 边界定义

- 物理容量边界：实际 KV blocks/token slots、CUDA peak memory、首次 OOM 或无法分配的点。

- 无抢占容量边界：首次出现 KV preemption/recompute 的并发点。

- 服务容量边界：首次出现联合 SLO goodput 明显下降、TPOT p95 失守或请求无法完成的点。物理上能装入显存，不代表在目标延迟下可服务。

- 输出“最大并发 vs prompt 长度”曲线，并分开报告 BF16/FP8 KV、swap on/off、冷/热 prefix。burst 矩阵之后再用 Poisson 到达 trace 验证在线服务边界。

- 交付物：容量矩阵 CSV/JSON、显存与 KV block 账本、抢占/SLO 失守点、FP8 held\-out 精度报告。

## 4\. 让算子收益在目标 workload 上稳定可复现

### 优先选择可控消融

先选仓库已有明确开关的 INT4 MoE segmented GEMM gate/up 融合作为案例：DeepSeek\-V2\-Lite、纯 INT4 group64、TP=1，比较 loop、segment\-split、segment\-fused。现有固定输入正确性覆盖很强，但端到端融合收益会随输出长度、到达率与 TPOT 目标变化，配对波动区间仍跨过零。因此本实验需要分别证明内核正确、固定形状收益、在线收益。

### 阶段 A：微基准与数值正确性

1. 在真实模型权重上测 token batch 1/4/16/64/128；每种配置预热，使用 CUDA events 计时，增加重复样本数量。固定输入逐位比对 loop、segment split、segment fused 输出。

2. 交替 split/fused 运行顺序；每个形状做至少 10 组配对重复。保存输入 fingerprint、kernel latency 分布、GPU clock/温度和显存。

```bash
MOE_MODEL=~/huggingface/DeepSeek-V2-Lite
python benchmarks/moe_int4_ablation.py "$MOE_MODEL" --int4-group-size 64 --token-batches 1,4,16,64,128 --profile-token-batches 1,16,64 --warmup 5 --iterations 10 --repeats 10 --output results/supplemental/moe_int4_micro.json
```

### 阶段 B：在线服务配对

- 固定 TP=1、INT4 group64、eager、KV swap off；输入先用 64 tokens 的固定形状隔离算子，再增加真实 ShareGPT prompt 验证外部有效性。输出长度 16/64/128，arrival rate 4/8/16/24 req/s，TPOT 目标 100/125/150 ms；先用较小矩阵找差异，再细化候选区间。

- 对 split/fused 使用相同到达 trace 和采样 seed，交替后端顺序；保存完整 model\-call shape histogram、SM clock、首个分歧 token 和输入 fingerprint。固定输入重放用于性能配对；端到端 token 差异不能直接判定 kernel 不正确。

- 每个候选格至少 5 个 seed × 5 次配对；若 95% 配对区间跨零，结论写成“未观察到稳定收益”或限定到具体负载条件。不要用 microbenchmark 的大倍数替代全模型 serving speedup。

### 指标与判定

- 内核：median/p95 latency、配对 speedup、max absolute/relative error、寄存器/显存信息（可取得时）。

- 端到端：output tok/s、joint\-SLO output goodput、请求达标数、TTFT/TPOT p50/p95、GPU clock、batch shape 和每次 trace 的方差。

- 只有固定输入正确性通过，且关键目标负载下配对改善稳定、置信区间不跨零时，才把该算子写成已验证优化；否则描述为条件性收益并保留可选开关。

## 5\. 补投机解码的内容依赖实验

### 先修正基准公平性

当前 **spec\_bench\.py** 的 free 输入在每次 run\_case 内重新抽随机 token，因此 baseline 与 spec 可能不是同一批输入；json/repeat 则把一个完全相同的 prompt 复制给整个 batch。正式扩展前应一次性生成 tokenized prompts，并在所有模式中复用；每个请求使用不同但同类的 prompt，避免相同 prompt/prefix cache 混入 decode 收益。脚本的 **\-\-seed** 目前主要控制 prompt 构造，GPU sampler 仍是随机采样；记录每次 seed 并重复多轮，不能依赖单个随机输出。

### 内容分层与运行顺序

1. 建立四类各 50–100 条不同请求：重复/echo 正对照；JSON、SQL 或代码等结构化输出；ShareGPT 真实对话/总结；开放式问答或推理等低重复生成。训练 Medusa/EAGLE draft 权重的 prompt 与 held\-out 评估 prompt 按来源隔离。

2. 第一阶段只比较 baseline 与 n\-gram，batch=8，统一 96 或 128 输出 token、prompt 长度分桶、同一组输入；每类至少 3 个 seed。先用 **ignore\_eos=True** 做固定输出成本对比，再增加 **ignore\_eos=False** 的自然停止实验。

3. 第二阶段对高接受率和低接受率两类扩大到 batch 1/8/32，扫描 **max\_draft\_len** γ=2/4。EAGLE/Medusa 只在对应模型的 draft/head 权重已训练且验证通过时加入；n\-gram 作为无额外模型的主对照。

4. 按内容类别分别跑，不把不同类别混成一个 batch 后只报告总 α。若希望做请求级相关性，给每条 sequence 增加 accepted/proposed 计数，再与生成文本的 4\-gram 重复率、distinct\-4、token entropy 关联。

### 采集指标与解释

- 记录 proposal 数、accepted draft 数、α、每个 verify step 的实际接受 token 数、spec verify/decode steps、decode tokens/s、全请求吞吐、TTFT/TPOT p50/p95、E2E、GPU peak memory。

- 输出内容统计：4\-gram 重复率、distinct\-4、输出长度和类别；用“内容特征—接受率 α—加速比”散点图判断哪些工作负载处于 break\-even 以上。加速比不能只由 α 推断，草稿成本、verify 成本、batch 和 γ 都要同时记录。

- 至少 5 组配对重复后报告 median 与 95% 区间。精度/分布保持检查单独做；temperature\>0 时不同路径可能消费不同随机数，不能要求普通跑分输出逐 token 完全相同。

- 已有 EAGLE γ=2 重复文本加速、自由文本退化的数据只能作为高/低接受率示例，不代表真实请求总体。新简历数字必须写清模型、类别、batch、γ 和硬件。

## 6\. 本轮实验执行顺序与交付状态
1. 已完成 ShareGPT trace builder/loader 与 fingerprint 配对；生成 500 请求 trace。seed0 结果的 500 个 request ID、max_tokens 预算和到达时间均与该 trace 匹配；100 请求 smoke 结果已留档。
2. 已完成 BF16/FP8 KV 容量矩阵、真实 prompt 点和 swap 路径诊断。未触发 OOM 或测到物理 KV 容量极限；swap 结果只用于确认路径。
3. 已完成准入 target/work、soft-pressure/max-wait 筛查，3-seed calibration 和 2-seed held-out 检查。两阶段 target/work SLO 不同，held-out workload 较短且到达率不同；没有业务拒绝率预算，不推荐默认阈值。
4. 已完成 n-gram 内容分层与 batch/γ 扩展，以及 INT4 MoE 固定输入正确性和在线配对。自然停止、Medusa/EAGLE 权重评估未完成；MoE 5×5 重复仅覆盖一个条件。
5. 已保留 trace builder、benchmark/report 脚本、运行配置、原始 JSON/CSV 和图表。汇总结论见 [`benchmarks/_supplemental_report.md`](benchmarks/_supplemental_report.md)；原始实验数据保存在本机被 Git 忽略的 `results/supplemental/`。

建议简历只写实验最终支持的结论，例如“基于真实 prompt 与到达 trace 对调度/准入策略做配对评测，按 TTFT/TPOT 联合 SLO goodput 校准准入阈值”；具体加速率、接受率和容量数字等测完后再填写。

