# nano\-vLLM实验补足



# nano\-vLLM 下一阶段实验工作与详细步骤（截至 2026\-10\-16）

**目标：**把已经补做的实验收敛成可复现、能区分功能正确性与服务收益的下一轮计划。以下优先补齐现有结论的证据缺口，不把单次筛查或 kernel 微基准写成端到端加速。

|方向|已有证据|目前缺口|
|---|---|---|
|多模型支持|RTX 5060 Ti 16GB、INT4 streaming-load 下，Qwen2.5-7B、Mistral-7B-v0.1、Llama-3.1-8B 均通过 16/16 请求生成 smoke。|尚未完成三模型的并发 × 上下文矩阵；smoke 不代表长时间稳定服务边界。|
|调度优化|2026-10-16 当前工作树 5-seed 配对中，关闭 TPOT-aware 相对 all_on 的联合 SLO 输出 goodput 增加均值 213.9 token/s（95% CI +47.8 至 +405.7）；相对 baseline 的 CI 跨零。10-15 的历史负结果单独保留。|本轮只完成 baseline/all_on/without_tpot_aware；其它策略及更多到达率仍需消融。Poisson 到达是合成负载。|
|准入|250-request calibration 与 held-out trace 各 3/2 个 seed、96 req/s；admission-off 每次零拒绝且 goodput 高于 cache-blind/cache-aware。|两种动态准入在 calibration 中拒绝约 31%–34%；没有产品拒绝预算，不能据此推荐默认阈值。|
|算子优化|修正 MLA query range 为逐行 start + qlen 后，seed42、8 req/s、16 请求的 5 对诊断均 token parity，focus logits 5/5 相同，行置换 replay 10/10 通过。|历史 32 req/s、24 请求的分歧条件本轮未重跑；仍未定位首个不同 KV 状态。本轮是诊断而非性能测量。|
|量化与投机解码|FP8 已补 250 calibration + 250 held-out 的 decode position 1/8/32/64/128 teacher-forced 精度评测；BF16/FP8 在线容量分别 108,800/211,712 token slots。N-gram 已有自然停止及在线 A/B；gamma4 Medusa/EAGLE checkpoint 均可加载。|FP8 margin 无全面胜者且未提高各点 SLO；N-gram 在线收益只在重复合成文本较稳定；gamma4 Medusa/EAGLE greedy 输出 parity 均为 0/3。|

## 0\. 统一实验口径与复现门槛（P0）

1. 复跑关键结论。2026-10-16 的调度、准入、量化与投机结果记录为 HEAD a5a2e0e71d91753e2384285b079aaaf0535b84c6，manifest 标记 git_worktree_dirty=true；这些是当前改动工作树上的复测，不能称为干净提交复现。提交最终选定改动后，仍需在 clean tree 上复跑代表性对照。结果 manifest 应保留 commit、dirty 状态、模型 revision、Python/PyTorch/CUDA/Triton/flash-attn 版本、GPU 时钟/温度和完整命令。

2. 固定请求 trace。沿用 benchmarks/sharegpt\_trace\.py 生成的脱敏 user prompt；assistant 回复只用于派生截断的 max\_tokens，不作为模型输入。保存 trace 文件 SHA256、逐请求 prompt token hash、max\_tokens、arrival\_time\_s、prefix\_group 和 SLO。把合成 Poisson 到达明确标作合成流量；若使用合成公共前缀，也单独标记为 synthetic。

3. 成对运行必须使用同一批 tokenized prompts、到达时间、采样配置和 workload fingerprint。每个变体使用新 engine，预热与计时分开；交替执行 A/B 顺序，运行前清空指标和 prefix KV。每个 seed 的记录数必须和去重后的 request\_id 数一致。

4. 校准与 held\-out 分开。screen 使用 3 个 seed；候选结果至少 5 个配对 seed。held\-out 使用未参与选参数的 prompt 子集，但 request 数、到达率、prompt 长度分布、输出预算、前缀比例和 SLO 与 calibration 一致。配对比较按 seed 计算差值并给出 95% bootstrap CI。

5. 统一口径：同时报告 TTFT/TPOT/E2E p50、p95、p99，输出 token/s、满足 TTFT 与 TPOT 两个 SLO 的输出 token goodput、请求达标率、接收/拒绝率、KV 抢占与 swap、峰值显存。goodput 分母使用测试墙钟时间；拒绝请求不能从到达请求统计中消失。

## 1\. 多模型支持：从“能加载”补到“能稳定服务”

**要回答的问题：**已测的 INT4 streaming\-load 显存值是否能在三个 checkpoint 上转化为端到端可用的上下文与并发范围。

1. 固定 Qwen2\.5\-7B、Mistral\-7B、Llama\-3\.1\-8B 的 checkpoint revision、tokenizer、权重格式、量化方式和 group size。不同 checkpoint 若量化格式不同，分别记录配置，不把不同量化格式的结果当成同条件比较。

2. 先做每模型 16 条 prompt 的 smoke：短/中/长 prompt 各覆盖，greedy 生成 32 tokens；检查模型加载、首 token、EOS、输出长度、异常退出、峰值显存。失败时保留完整错误和模型配置，不只记录成功模型。

3. 通过 smoke 后，使用同一组 100 条脱敏用户文本，按各模型 tokenizer 重新分为 128–256、257–512、513–1024、1025–2048 token 桶；输出上限固定 64。分别测并发 1/2/4/8，每点 3 次独立进程重复。上下文总长度不得超过模型位置上限。

4. 每个模型记录 load time、权重与 KV 显存、TTFT/TPOT/E2E 分位数、输出 token/s、OOM/抢占数和完成率。若要评估量化质量，额外用相同 teacher\-forced token 前缀比较 BF16 与量化 logits；只测显存不能得出质量或吞吐结论。

**通过标准：**三次重复均能加载并完成所有请求，报告每个模型的可服务边界。模型之间只报告各自结果；优化收益必须在同一模型、同一 checkpoint、同一负载下做成对比较。

## 2\. 调度优化：定位 all\_on 回退，并在真实前缀上验证缓存价值

**现有结论：**all\_on 在 500 请求、8 req/s 的 5 个配对 seed 上同时降低输出吞吐和联合 SLO goodput，不应继续作为整体优化结论。固定 512\-token 合成前缀的 128 请求测试曾把 prefill 工作量从 77,824 降至 25,393 tokens（约 67%）；下一步要验证实际重复 system prompt 的效果。

### 2\.1 调度策略逐项消融

1. 使用固定 500 请求 ShareGPT trace，Qwen3\-0\.6B BF16、Poisson 8 req/s、TTFT≤2,000 ms、TPOT≤80 ms；同一轮运行 baseline、all\_on、without\_tpot\_aware、without\_multi\_step\_decode、without\_slo\_adaptive、without\_cache\_affinity、without\_aging。

2. 先以 3 个 seed 筛查；只把有改善迹象的策略扩展到 5 个配对 seed。每个 seed 变体之间校验 prompt/arrival fingerprint 完全一致。筛查通过后，再在 4/8/16/24 req/s 的负载点复测；如要研究 TPOT 目标影响，20/40/80 ms 分开成组比较，不把不同 SLO 的数据拼成同一条曲线。

3. 结果表至少包含吞吐、联合 SLO goodput、TTFT/TPOT p50/p95/p99、排队等待、抢占、prefix hit tokens 和各策略计数。若 all\_on 回退，结合单项消融和队列/配额计数说明主要回退来自哪条策略；若 CI 跨零，结论写“未观察到稳定收益”。

```bash
MODEL_DIR=/path/to/Qwen3-0.6B
TRACE=datasets/sharegpt_500.jsonl
python benchmarks/scheduling_ablation.py --model "$MODEL_DIR" --prompts-file "$TRACE" --num-seqs 500 --arrival-mode poisson --arrival-rate 8 --variants baseline all_on without_tpot_aware without_multi_step_decode without_slo_adaptive without_cache_affinity without_aging --slo-ttft-ms 2000 --slo-tpot-ms 80 --seed 0 --output results/supplemental/schedule_seed0.json
```

### 2\.2 Prefix Caching 与准入策略

1. 若目标服务有固定 system prompt，构造真实重复前缀 trace：0%、20%、50%请求共享相同 system prompt，前缀长度按实际 tokenizer 结果及 block 对齐；用户正文保持不同。没有可用线上 prompt 时继续用合成模板，但结论限定为 synthetic prefix。

2. 准入只保留已筛出的候选，不再重复大网格：dynamic\_off、dynamic\_on\_no\_cache、dynamic\_on\_cache\_aware；候选参数比较 500/5000/1\.0/500 与 2000/10000/1\.0/500（依次为 target TTFT ms / work budget ms / soft pressure / max wait ms）。在 24 req/s 的 0/20/50% prefix 条件及无共享前缀的 48/96 req/s 压力点分别测量。

3. 先用 calibration 子集 3 个 seed，再用不重叠的 held\-out prompt 子集 2 个 seed。calibration 和 held\-out 都使用 400 个请求、相同 offered rate、到达分布、输出上限和 SLO；不要再用 400 请求/96 req/s calibration 对比 100 请求/384 req/s held\-out。

4. 按“拒绝率—全到达请求联合 SLO goodput—p99 TTFT”画 Pareto 图，同时报告接收请求达标率、defer/reject 数、拒绝原因和估算 cache hit 与真实复用 token 的误差。不同 TPOT SLO 分开画。

**决策规则：**目前没有业务拒绝率预算，所以不推荐新的默认拒绝阈值。只有某个候选在 held\-out 上提升联合 goodput、满足明确的拒绝率上限且置信区间不跨零时，才把它作为限定负载下的可选 operating point。

## 3\. 算子优化：先定位在线分歧，再决定是否保留融合

**现有结论：**INT4 segment kernel 固定输入正确性和微基准结果较强，但在线收益仍在噪声范围内。2026-10-16 修正 MLA query range 后，在 seed42、8 req/s、16 请求的独立诊断中 5 对均无 token 分歧，行置换 replay 也通过；历史 seed42、32 req/s、24 请求的分歧条件没有重跑，根因仍未定位。性能筛查前仍需在原 workload 上追踪首个不同状态，避免误归因到 fused GEMM。

1. 在 DeepSeek\-V2\-Lite、TP=1、INT4 group64、eager、固定 greedy 输入上复现 seed42、8 req/s、16 请求、64\-token 输入/输出的在线差异，至少重复 5 次 split/fused 配对。保存每个请求的 prompt/output fingerprint、model\-call shape histogram、request row order、block table、slot mapping 和逻辑 KV 行序。

2. 对首个不同请求与 token 做逐层对齐：比较输入 hidden、router top\-k 与 expert assignment、gate/up/down 输出、attention 输入/输出、KV 写入位置和 logits；沿 prefill 到该 decode position 回溯，报告最早不同的 tensor、层、token position 与误差。先判断是数值差、缓存索引/生命周期还是请求重排，再针对根因修改。

3. 根因修复后重新跑固定 shape parity。之后才执行在线性能筛查：8/16/24 req/s、64/128 输出 token、TPOT 100/125/150 ms；筛查 3 seeds × 2 repeats，选中的负载扩大为 5 seeds × 5 repeats。split/fused 交替运行，固定输入 trace 和采样配置。

4. 同时报告 kernel latency、端到端输出 token/s、joint\-SLO goodput、TTFT/TPOT、token parity、形状直方图和配对置信区间。若端到端收益区间跨零，融合保留为实验选项；不能用 11\.90×–42\.95× 的 kernel\-local speedup 替代服务收益。

```bash
python benchmarks/moe_int4_fuse_sweep.py --model /path/to/DeepSeek-V2-Lite --seeds 7,17,42 --arrival-rates 8,16,24 --output-lengths 64,128 --num-seqs 24 --input-length 64 --int4-group-size 64 --slo-tpot-targets 100,125,150 --repeats 2 --sampling-mode greedy --output results/supplemental/moe_fuse_screen.json
```

## 4\. 量化与投机解码：补多步精度、在线容量和自然停止

### 4\.1 FP8 KV 与 swap

1. 校准集与 held\-out 完全分离。held\-out 扩到至少 200 条 prompt，覆盖短/长文本、代码、数字和不同上下文长度；保持现有 scale margin 1\.0/1\.1/1\.25 对照。

2. 当前 kv\_fp8\_calibrate\.py 只比较首个 pure decode logits。新增 teacher\-forced 多位置评测：先用 BF16 生成并固定一段目标 continuation，再把同一 token 前缀输入 BF16 与 FP8 KV，比较 decode position 1/8/32/64/128 的 top\-1、top\-5 overlap、KL、logit error 和 E4M3 饱和计数。这样避免两个生成路径在首 token 分歧后比较到不同上下文。

3. 在 Qwen3\-0\.6B 上做 BF16 KV 与 FP8 E4M3 KV 在线矩阵：prompt 512/1024/2048/4096、并发 1/2/4/8/16/32、输出 64；确保输入加输出不超过模型长度上限。两种 dtype 都关闭 swap，逐级增加并发，记录实际 blocks/token slots、峰值显存、preemption、完成率、TTFT/TPOT 与 joint\-SLO goodput。需要时为压测脚本增加逐点捕获 CUDA OOM 并写出部分 JSON 的逻辑，避免一次 OOM 丢掉整批结果。

4. 只在确实出现 KV 抢占的点比较 swap\-on、swap\-off 和 recompute\-aware 默认策略，记录交换字节数、传输耗时、重算耗时与完成请求数。此前 swap\-on 低 8\.1%，因此目标是确定何种压力下它能避免失败或改善 SLO，而不是预设其能提高吞吐。

**判定：**分别报告容量账本与在线可服务并发。只有在同一请求 trace、相同 SLO 下 FP8 增加了达标并发且多位置精度达到预先约定阈值，才能写容量收益；约 1\.96× pool slots 不等于 1\.96× 在线并发。

### 4\.2 N\-gram / Medusa / EAGLE 投机解码

1. 沿用已有分层 prompt，补成每类至少 50 条互不相同的 held\-out prompt：重复文本正对照、JSON、代码/结构化输出、ShareGPT 对话、开放式问答。baseline 与 speculative 模式必须复用相同 tokenized prompts、采样参数和输入 fingerprint。

2. 先用 greedy、固定输出长度检查 none 与 ngram 的输出正确性；再用 temperature=0\.6 分别跑 ignore\_eos 固定长度与 natural\-stop 两种模式。屏幕矩阵为 batch 1/8/32、γ=2/4；先 3 seeds 筛查，再对代表性高/低接受率类别做 5 个配对 seed。

3. 记录 proposed/accepted token 数、α、TTFT/TPOT p50/p95、E2E、输出 token/s、输出长度、EOS 比例、4\-gram 重复率和 distinct\-4。分内容类别画 α 与 speedup；自然停止的结果和固定 token 成本结果分开汇总。

4. 若 n\-gram 在某些类别通过离线筛查，再将该类别放入同一连续到达 trace 做 speculative on/off 服务配对，报告 joint\-SLO goodput。离线 decode speedup 不能直接当作 serving speedup。

5. 本机已有可用于 Qwen3 的 Medusa/EAGLE gamma4 权重；已完成加载 smoke 和 gamma4 greedy 配对筛查。gamma2 Medusa 因 checkpoint/head 数量不匹配未通过加载，EAGLE gamma2 smoke 可生成。gamma4 token 输出 fingerprint 与 baseline 均未匹配（0/3），Medusa alpha 约 0.010–0.014、EAGLE 约 0.047–0.051。需先核对数值/verify 路径并提高接受率，再讨论模型投机收益；checkpoint 能加载不等于质量或性能已验证。

```bash
python benchmarks/spec_bench.py --model /path/to/Qwen3-0.6B --prompts-file datasets/spec_heldout.jsonl --speculative none,ngram --batch-sizes 1,8,32 --max-draft-lens 2,4 --max-output-len 256 --seeds 7,17,42 --repeats 3 --sampling-mode random --temperature 0.6 --natural-stop --output results/supplemental/spec_natural_screen.json
```

## 5\. 执行顺序与交付物

1. **第一步：**在当前干净提交复跑代表性结果，并生成 manifest；完成三模型端到端 smoke，确认权重加载数字可以对应到真实生成。

2. **第二步：**完成调度逐项消融和准入 held\-out 配对；优先解释现有负收益，并用真实 system prompt 或明确标注的 synthetic prefix 验证缓存。

3. **第三步：**扩展 FP8 多位置精度与在线 KV 压力；只有在实际出现抢占的点测 swap。

4. **第四步：**追踪 MoE 在线首个分歧，修复并复测 parity 后再扩大融合性能矩阵。

5. **第五步：**为 n\-gram 加入 natural\-stop 和在线 serving 对照；有匹配权重时再加入 Medusa/EAGLE。

- 每项实验保留原始 JSON/CSV、prompt/arrival fingerprint、命令与配置、绘图脚本和一段结论；结果区分“稳定收益、负收益、未确认”。

- 简历只采用当前干净版本复跑后有同负载配对证据的指标；限定模型、GPU、请求数、到达率、SLO 和配置。固定形状微基准、容量折算与单次 screening 分别标注，不能混作端到端性能结论。

本计划依据仓库 benchmarks/_supplemental_report.md 与 INTERVIEW.md §10.3.19 整理。下列 2026-10-16 进展仅代表标明的模型、GPU、工作负载和 dirty 工作树；“仍未完成”部分保持为计划，不代表已实测。

## 6. 2026-10-16 实测进度与优化消融

### 环境与复现口径

Qwen3-0.6B 结果使用 RTX 5060 Ti 16GB、Python 3.12.13、PyTorch 2.8.0+cu128、CUDA 12.8、Triton 3.4.0；模型路径为本机 checkpoint。运行记录的 HEAD 为 a5a2e0e71d91753e2384285b079aaaf0535b84c6，且 git_worktree_dirty=true。ShareGPT trace 使用脱敏用户 prompt；到达过程为合成 Poisson，不能称为生产流量。原始 JSON/CSV 位于 Git 忽略的 results/supplemental/。

### 调度优化前后对照

在 500 请求、8 req/s、TTFT≤2000 ms、TPOT≤80 ms 的 5-seed 配对 trace 上，报告的是按 seed 配对差值的均值与 bootstrap 95% CI：

| 配对 | 输出吞吐差 (token/s) | 联合 SLO 输出 goodput 差 (token/s) | 解释 |
|---|---:|---:|---|
| all_on − baseline | +158.8（+9.4 至 +308.1） | −59.6（−163.0 至 +37.9） | goodput 区间跨零，未显示稳定改善。|
| without_tpot_aware − all_on | +113.5（+61.8 至 +165.2） | +213.9（+47.8 至 +405.7） | 本负载下关闭 TPOT-aware 相对 all_on 有配对收益，吞吐 5/5 seed 增加，goodput 4/5 增加。|
| without_tpot_aware − baseline | +272.3（+111.8 至 +432.7） | +154.3（−40.2 至 +323.1） | 输出吞吐改善，但 joint-SLO goodput CI 跨零，不足以改默认策略。|

这组数据与 2026-10-15 HEAD 143ce1b 的 all_on 负结果属于不同代码状态/实验批次，分别保留；不能合并成同一组统计。其它 scheduler 单项和负载点尚未在本轮完成。

### 准入、FP8 与 KV 容量

- 准入在 calibration / held-out 各用 250 个请求、96 req/s、TTFT≤2000 ms、TPOT≤80 ms；每个 split 分别使用 3 个和 2 个 seed。每一 seed 的三个变体共享 workload fingerprint。admission-off 每次零拒绝且 goodput 高于两个动态准入变体；calibration 中 cache-blind −893 token/s（95% CI −1173 至 −723），cache-aware −685（−817 至 −596）。动态策略 calibration reject rate 约 31%–34%。没有业务拒绝预算，仍不推荐默认阈值。
- FP8 teacher-forced 评估用 250 calibration 和 250 held-out prompts，固定 BF16 continuation，并检查 decode position 1/8/32/64/128。margin 1.0/1.1/1.25 没有单一全面胜者；margin 1.25 的 held-out E4M3 饱和计数为 K=0、V=1，margin 1.0 为 K=338、V=53。饱和减少不等同于 logits 或生成质量全面更优。
- KV 容量矩阵 BF16、FP8 各 24 个点、每点 3 次，无失败。BF16 池为 425 blocks / 108,800 slots；FP8 为 827 / 211,712 slots（1.946×）。4096-token prompt、并发4时 BF16 12/12 请求同时满足 TTFT/TPOT SLO，FP8 为 4/12；并发32时两者都没有请求同时满足联合 SLO。容量增加没有转化成稳定的 SLO 收益。

### N-gram 与 Medusa/EAGLE 消融

- 固定长度 greedy none/ngram 校验在 repeat、JSON × batch1 × gamma2/4 × seed42 上 4/4 输出 token fingerprint 匹配；单 seed 只作正确性检查。自然停止 screen 完成 90/90 配对（5 类内容 × batch 1/8/32 × gamma 2/4 × 3 seeds），但 128-token 输出上限下 EOS 计数为 0；初次进程在 78/90 后因 WSL 内存和 swap 用尽中断，保留中断 JSON/CSV 并从 checkpoint resume 完成。不能把这 90 对称为有效 EOS 停止测试。
- Qwen3 chat template 的 ShareGPT、batch=8、512-token cap 触发真实 EOS。离线 gamma2 吞吐中位差 +9.44%（2/3 胜，95% CI −8.57% 至 +15.41%）；gamma4 +5.21%（3/3 胜，CI +2.15% 至 +9.39%）。这是 3-seed 小样本筛查。
- N-gram 在线重复合成 prompt、100 请求、cap96、合成 Poisson 到达：8 req/s 的三个 seed 为 +2.57%、+2.64%、+2.47%；16 req/s 为 +3.14%、+5.74%、+2.99%。两侧每次都是 100/100 请求满足联合 SLO。该结果只支持高度重复合成内容上的小幅收益。chat-template ShareGPT、4 req/s、cap512 的三组变化为 +0.19%、−5.03%、+2.47%，中位 +0.19%、均值 −0.79%，没有稳定在线吞吐提升。
- Medusa/EAGLE gamma4 greedy screen 使用同一 chat-template prompt fingerprint、batch8、固定 128-token 输出、3 seeds。Medusa 吞吐中位差 +9.27%（2/3 胜，95% CI −6.22% 至 +10.86%），EAGLE −35.79%（0/3 胜，CI −36.62% 至 −34.00%）；两者的输出 token fingerprint 都是 0/3 与 baseline 一致。因此 Medusa 的正向吞吐筛查不作为等质量加速证据，EAGLE 在此条件下为负收益。

### 多模型与 MoE 诊断

- 多模型 INT4 smoke：Qwen2.5-7B、Mistral-7B-v0.1、Llama-3.1-8B 均完成 16/16 请求；并发 × 上下文矩阵尚未执行。
- MLA query assembly 已调整为逐行 query_end = query_start + query_length。更新后 DeepSeek-V2-Lite INT4 在 seed42、8 req/s、16 请求条件下做 5 对：token parity 5/5，focus logits 5/5 相同，固定 KV 的行置换回放检查 10/10 通过。历史 seed42、32 req/s、24 请求的差异条件没有重跑，首次不同 KV 写入位置仍未定位；诊断耗时不是性能结果。

### 尚未完成

三模型并发/上下文矩阵、全量调度策略与到达率筛查、MoE 历史差异首因定位，以及提交最终代码后的 clean-tree 复跑仍待完成。所有服务吞吐结论继续限定于上面列出的合成/held-out workload 和 GPU。

