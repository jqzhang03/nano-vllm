## 2026-10-16 补充实验

本节记录 10 月 16 日在 Qwen3-0.6B / RTX 5060 Ti 16GB 上完成的后续实验。运行 manifest 的 HEAD 为 a5a2e0e71d91753e2384285b079aaaf0535b84c6，git_worktree_dirty=true；数据对应当时未提交的工作区，不能称为干净提交复现。ShareGPT trace 是脱敏用户 prompt，Poisson 到达为合成流量，不代表生产请求。

### 调度和准入

调度配对使用 500 请求、8 req/s、TTFT≤2,000 ms、TPOT≤80 ms、5 个 seed。下表为按 seed 配对差值均值及 bootstrap 95% CI，吞吐和 goodput 单位为 token/s：

| 配对 | 输出吞吐差 | 联合 SLO 输出 goodput 差 | 结果 |
|---|---:|---:|---|
| all_on − baseline | +158.8（+9.4 至 +308.1） | −59.6（−163.0 至 +37.9） | goodput CI 跨零，未显示稳定改善 |
| without_tpot_aware − all_on | +113.5（+61.8 至 +165.2） | +213.9（+47.8 至 +405.7） | 输出吞吐 5/5 seed 增加，goodput 4/5 增加 |
| without_tpot_aware − baseline | +272.3（+111.8 至 +432.7） | +154.3（−40.2 至 +323.1） | goodput CI 跨零，不足以改变默认策略 |

关闭 TPOT-aware 在此工作负载上改善了 all_on 的结果，但相对 baseline 的联合 SLO goodput 仍未达到稳定差异。10 月 15 日 HEAD 143ce1b 的负结果是不同代码状态和实验批次，继续作为历史数据单独保留。

准入在 calibration 与 held-out 各使用 250 个请求、96 req/s、TTFT≤2,000 ms / TPOT≤80 ms；每组分别 3 和 2 个 seed。三个策略共享每个 seed 的 workload fingerprint。admission-off 在每次运行中零拒绝且 goodput 高于 cache-blind 与 cache-aware；校准配对的 goodput 差分别为 −893 token/s（95% CI −1,173 至 −723）和 −685（−817 至 −596）。动态策略校准拒绝率约 31%–34%。由于没有产品拒绝预算，不推荐默认准入阈值。

### KV 容量与 FP8 精度

BF16 与 FP8 E4M3 的在线容量矩阵各有 24 点，每点 3 次，无失败。BF16 为 425 blocks / 108,800 token slots；FP8 为 827 / 211,712 slots（1.946×）。4096-token prompt、并发 4 时，BF16 有 12/12 请求同时满足 TTFT/TPOT SLO，FP8 为 4/12；并发 32 时两者均没有请求同时满足联合 SLO。扩大 KV 容量没有转化为稳定的在线 SLO 改善。

FP8 teacher-forced 对照使用 250 条 calibration 与 250 条 held-out prompt，以 BF16 continuation 固定输入，并比较 decode position 1/8/32/64/128。margin 1.0/1.1/1.25 没有全面胜者。margin 1.25 的 held-out E4M3 饱和计数为 K=0、V=1，margin 1.0 为 K=338、V=53；饱和减少本身不能代表 logits 或生成质量全面更优。

### N-gram、Medusa 与 EAGLE

单独的固定长度 greedy none/ngram 校验覆盖 repeat 与 JSON、batch 1、gamma 2/4、seed 42；4/4 配对的输出 token fingerprint 完全匹配。这是一个 seed 的正确性检查，不作为吞吐结论。自然停止 screen 完成 90/90 配对，覆盖 5 类内容、batch 1/8/32、gamma 2/4、3 个 seed；128-token 输出 cap 下 EOS 计数为零，因此该 screen 不能当作自然 EOS 结果。首个 screen 进程在 78/90 后因 WSL 内存和 swap 用尽退出；中断 JSON/CSV 已保留，后续从 checkpoint resume 完成剩余配对。

对 Qwen3 chat template 后的 ShareGPT、batch 8、512-token cap，模型实际产生 EOS。gamma2 吞吐中位差 +9.44%（2/3 胜，95% CI −8.57% 至 +15.41%）；gamma4 +5.21%（3/3 胜，CI +2.15% 至 +9.39%）。每轮只有 8 个请求，属于小样本离线筛查。

在线 n-gram 重复文本 trace 为 100 请求、max_tokens=96、合成 Poisson 到达。8 req/s 三个 seed 的吞吐变化为 +2.57%、+2.64%、+2.47%；16 req/s 为 +3.14%、+5.74%、+2.99%。两侧每次均有 100/100 请求满足联合 SLO。这只支持高度重复的合成文本上存在小幅吞吐收益。chat-template ShareGPT、4 req/s、100 个 held-out prompt、512-token cap 的三组变化为 +0.19%、−5.03%、+2.47%，中位 +0.19%、均值 −0.79%，没有稳定在线吞吐提升。

gamma2 的权重 smoke 中，EAGLE 可生成，Medusa 因 checkpoint 与构造的 head 数量不匹配而无法加载；gamma4 的 Medusa/EAGLE 均可加载。gamma4 配对使用同一 chat-template prompts、greedy、batch 8、3 个 seed。Medusa alpha 为 0.010–0.014，吞吐中位差 +9.27%（2/3 胜，95% CI −6.22% 至 +10.86%）；EAGLE alpha 为 0.047–0.051，吞吐中位差 −35.79%（0/3 胜，CI −36.62% 至 −34.00%）。两者与 baseline 的输出 token fingerprint 均为 0/3 匹配。Medusa 表面吞吐正值不作为等质量加速证据，EAGLE 在此条件下为负收益。

### 多模型与 MoE 诊断

Qwen2.5-7B、Mistral-7B-v0.1 与 Llama-3.1-8B 的 INT4 streaming-load smoke 均完成 16/16 请求；并发与上下文矩阵尚未执行。

MLA query assembly 改为逐行计算 query_end = query_start + query_length 后，DeepSeek-V2-Lite INT4 在 seed42、8 req/s、16 请求的独立条件做 5 对诊断，均有 token parity，focus logits 5/5 相同，固定 KV 的行置换 replay 检查 10/10 通过。历史 seed42、32 req/s、24 请求的 divergence 条件没有重跑，首次不同 KV 写入位置仍未定位；诊断运行时间不是性能结果。

原始结果在被 Git 忽略的 results/supplemental/ 中，包括 scheduling_clean_a5a_paired_summary.json、admission_matched_cal_*_20261016.json、admission_matched_holdout_*_20261016.json、kv_fp8_teacher_forced_20261016.json、context_kv_*_20261016.json、spec_natural_screen_20261016.json、spec_natural_eos_chat_sharegpt_b8_g2_seed012_512cap_20261016.json、spec_natural_eos_chat_sharegpt_b8_g4_seed012_512cap_20261016.json、admission_spec_*_20261016.json、spec_medusa_eagle_greedy_sharegpt_chat_b8_g4_seed012_20261016.json、multimodel_smoke_20261016_cleanup.json，以及 moe_int4_arrival_diag_seed42_r8_n16_i64_o64_focus5_14_rep5_20261016.json。
