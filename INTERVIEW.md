# INTERVIEW.md — nano-vllm 面试串讲（全部功能一张纸）

> **定位**：面试前最后一天/一小时过一遍的"表演脚本"——给"怎么讲"，不给"全部细节"。
> 细节在 `note.md`（工作全景/踩坑故事）、`BENCHMARKS.md`（数字来源）、`LEARNING.md`（学习顺序）。
> **串讲三原则**：①先讲成本模型与上界，再讲实现；②主动交代"哪里亏、为什么"（比吹嘘可信）；
> ③所有数字带条件（单卡 RTX 5060 Ti 16GB / WSL2 / bf16 / flash-attn 2.8.3 / 模型量级）。

---

## 0. 电梯陈述（30 秒 / 2 分钟 / 5 分钟）

### 30 秒
> 我从零实现了一个 vLLM 风格的推理引擎（约 1200 行 Python + Triton，不依赖任何推理框架）：
> 连续批处理调度（混合 prefill/decode）、paged KV cache + 前缀缓存 + COW、CUDA graph、
> 五条量化路径（w8a8/int4/AWQ/fp8/2:4 稀疏）、三种投机解码（n-gram/Medusa/EAGLE）、
> KV swap 抢占、按层流式加载，并移植了 Qwen/Llama/Mistral/Gemma-2 五个模型家族。
> 每个功能都与 HF 参考逐 token 对齐、与真实 vLLM 同 workload 对比过，并诚实记录了
> 哪些场景是亏的。

### 2 分钟
> 主线：**调度 → 内存 → 内核 → 量化 → 投机 → 系统 → 多模型**，每段 2-3 句 + 一个数字。
> - **调度**：先做"先全 prefill 再 decode"，实测发现早完成者死等 → 改成 vLLM V1 同款混合
>   批次（prefill 行在前、decode 行在后共享 token 预算），吞吐全档 +7~21%、抢占减半。
> - **内存**：paged KV cache（块 256）+ 链式哈希前缀缓存 + COW 安全写共享块 + 分块 prefill；
>   后来加的 KV swap 抢占用 CPU 缓冲换出 decode 序列，**bit-exact 0 误差**——但诚实结论是
>   本机（0.6B+WSL2）上 swap 比重算慢，价值在 7B+ 与真实 Linux。
> - **内核**：flash-attn 的 fp8 KV 路径在 sm_120（Blackwell 消费卡）不可用（FA3 是 Hopper-only），
>   自己写 Triton 内核（decode + verify 两套），fp8 KV 容量 1.9×、精度 KL 0.0073；这是
>   自研内核里最能打的点——"vLLM 在这张卡上跑不了的东西我能跑"。
> - **量化**：五条路径里 **fp8 是唯一大 M 不输 cuBLAS 的**（decode 走权重-only Triton 内核、
>   prefill 走硬件 FP8 MMA），ppl 3.60 vs fp16 3.60（+0.1%）近乎无损，bs=256 吞吐 5825 tok/s
>   是全模式峰值；2:4 稀疏的结论是"内核 bit-exact 但一次性剪枝丢 35% 权重质量"——技术
>   可行性评估型交付，知道哪里亏比什么都做更有信息量。
> - **投机**：n-gram/Medusa/EAGLE 三种都做通。核心理解是成本模型 γ·T_draft+T_verify vs
>   收益 (1+αγ)，而上界 α ≤ 模型 top-1 可预测性（实测自由文本 35%）——EAGLE γ=2 在
>   重复内容 +3.26×，free 文本只有 +0.70×，数字与理论互相印证。
> - **系统**：16GB 卡跑 7B+ 靠按层流式加载（meta 构造 → 逐层物化 → 加载即量化，峰值
>   10.7~11.6GB）；meta 物化踩了三个坑（to_empty 丢 weight_loader、RoPE 缓存全零、tie 重绑）。
> - **多模型**：注册表按 model_type 分发，移植 Qwen2.5/Llama-3.1/Mistral（SWA）/Gemma-2
>   （soft-cap），全部与 HF 参考 prefill logits top-1 100% 一致；Gemma-2 有"读源码才能发现"
>   的三个细节（embed ×√d、RMSNorm (1+w)、双残差四 norm），是逐层二分定位出来的。

### 5 分钟
> 2 分钟版本 + 三个"证伪"故事 + 两个诚实结论：
> - **证伪 1（大 tile 假设）**：int4 内核最初按"大 tile 更优"直觉调优，实测小 M 大 N 的
>   权重带宽主导形态才赢（4.36×），大 M 全输——于是做**双路径路由**（每层走自己赢的形态），
>   而不是硬碰 cuBLAS。
> - **证伪 2（分页瓶颈假设）**：假设 paged attention 的 gather 是瓶颈，实测 decode 的瓶颈
>   是权重带宽而非 KV 索引——这决定了 fp8 权重的方向（0.5× 字节）而不是继续优化索引。
> - **证伪 3（启动税假设）**：投机 verify 步慢，先怀疑内核，逐层 hook 计时发现是 CPU 启动
>   税 ~10ms/步 → 用 CUDA graph 固定容量重放，spec 从打平变赢（bs=8 repeat +3.87×）。
> - **诚实结论 1**：KV swap 在本机比 recompute 慢（27.8s vs 11.8s）——机制正确但不划算，
>   价值在 7B+（重算贵）+ 真实 Linux（D2H 快）。
> - **诚实结论 2**：2:4 稀疏、纯 int4 大 batch、EAGLE γ=4 都是"内核/机制正确但整体亏"——
>   知道边界在哪，比什么都做更能体现对推理系统的理解。

---

## 1. 主线叙事（按阶段，每阶段：一句话贡献 + 关键数字 + 为什么值得讲）

| # | 阶段 | 一句话贡献 | 关键数字 | 为什么值得讲 |
|---|---|---|---|---|
| 1 | vLLM 对比基准工程 | 先把"测量"做对：逐请求时间戳、同 workload 同 seed、两侧同 flash-attn | 吞吐 1.35-1.61× 领先；**decode 单步与 vLLM 持平（kernel 级可比）** | 口径诚实：离线 API 不暴露逐请求指标时用聚合直方图并声明近似 |
| 2 | FP8 KV cache + 自研 decode 内核 | FA3 是 Hopper-only，sm_120 只能自研 | 容量 1.9×；KL 0.0073；逐层 1.9→1.15ms | **"vLLM 在这张卡跑不了，我能跑"** 的差异化点 |
| 3 | W8A8（per-group + SmoothQuant） | int8 权重 + int8 激活 + 平滑折权重 | KL 0.0379（平滑后）；吞吐 -16% | 量化精度方法论：per-group 比 per-channel 细 8 倍 |
| 4 | 混合调度（V1 同款） | 消除"先全 prefill 后 decode"的死等 | 吞吐 +7~21%；抢占 85→68 / 141→71 | 调度器设计的核心权衡 |
| 5 | 投机解码框架（ngram→Medusa→EAGLE） | verify 步 = 带前缀复用的 varlen prefill + CUDA graph | verify 启动税 ~10ms/步被消除；EAGLE γ=2 repeat +3.26× | 成本模型 + α 上界 = 投机解码的完整理解 |
| 6 | fp8 varlen 内核 | verify 步直接读 fp8 缓存（免逐层反量化） | fp8+spec 从 0.15× 变 +3.92×（bs=8 repeat） | 内核与调度配合消除显存搬运 |
| 7 | INT4/AWQ 双路径 | 按形态路由：小 M 大 N 走 int4 内核，其余走 w_deq 稠密 | bs=8 +35%、bs=256 +3~6%；ppl 4.38→3.76（AWQ） | "带宽优化型内核在计算主导区间的天花板"的正面解法 |
| 8 | 2:4 稀疏（可行性评估） | 内核 bit-exact；一次性剪枝是精度灾难 | KL 8.5；cuSPARSELt sm_120 每调用 0.3-0.5ms | 技术评估类交付：知道"为什么不做" |
| 9 | 多模型：注册表 + 流式加载 + Qwen2.5/Llama | 16GB 卡跑 7B+ 的唯一路径 | Qwen2.5-7B 峰值 10.66GB；Llama-3.1-8B 11.62GB；parity top-1 100% | meta 物化三坑（to_empty/weight_loader/RoPE） |
| 10 | FP8 权重 | 唯一大 M 不输 cuBLAS 的量化 | ppl +0.1%；bs=256 5825 tok/s（1.22×）；8B prefill TTFT 210ms vs int4 813ms | 双路径（Triton 内核 + 硬件 _scaled_mm） |
| 11 | EAGLE-1 | 无 RoPE 草稿层 + 共享 LM head 自回归 | γ=2 repeat +3.26×（α 0.525）；γ=4 只有 +0.63× | γ 是成本模型的关键变量 |
| 12 | KV swap 抢占 | decode 序列 KV 换 CPU，恢复 bit-exact 免重算 | bitexact 0 误差；699 次换出；**本机比重算慢（27.8 vs 11.8s）** | 机制正确 + 条件诚实的样板 |
| 13 | Mistral（SWA）+ Gemma-2（soft-cap） | 两个新机制家族：滑动窗口 + logit soft-cap | parity top-1 100%（0.014/0.022）；Mistral 311 tok/s、Gemma2 1095 tok/s | Gemma-2 三个隐藏架构细节的定位方法论 |
| 14 | 工程素养：脚本 argv 化 + 验证工具箱 | 22+ 脚本模型路径全部参数化；pytest 41 例 + parity/内核对照/ppl 的回归矩阵 | 新模型复用全部基准零改动 | "测量与回归"是可信度的基础设施 |

---

## 2. 六大模块深水区（每个：机制 1 段 + 必问必答 + 数字 + 诚实结论 + 追问应对）

### 2.1 调度与批处理

**机制**：`Scheduler.schedule()` 返回 `(seqs, kind)`，kind ∈ prefill/decode/**mixed**/spec。
三队列 WAITING/RUNNING/FINISHED；混合批次 = prefill 行在前、decode 行在后，共享
`max_num_batched_tokens` 预算（vLLM V1 同款）；分块 prefill 只允许第一个被调度序列拆分
（其余拆分会破坏"每个 seq 每步至多一个 chunk"的不变量）；KV 块不足时抢占（decode 序列
优先 swap_out / 其余 recompute 回 waiting）。

**必问必答**：
- **Q：混合批次为什么赢？** A：早完成 prefill 的请求立即 decode，消除死等；decode 提前释放
  KV 块、降低抢占压力。实测吞吐 +7~21%、抢占减半（512 档 141→71）。但 TPOT 改善有限——
  它受总工作量下界约束，收益在流式延迟与资源利用率。
- **Q：分块 prefill 为什么只拆第一个序列？** A：多序列同时分块会让每步的 token 预算碎片化
  且块表管理复杂化；只拆第一个等价于"按到达顺序把预算给第一个长序列"，实现简单且覆盖
  了主要场景（少数长 prompt 拖尾）。
- **Q：抢占后怎么恢复？** A：recompute = 回 waiting + 释放块，恢复时按前缀缓存哈希命中
  部分免算；swap = KV 拷 CPU、释放块、进独立 swapped 队列，恢复时直接 decode（免 prefill）。
  草稿（投机）作废。

**数字**：混合调度吞吐全档 +7.2%~+21.3%；抢占 384 档 31→20、512 档 141→71；256 档峰值
5840 tok/s（fp16）。
**诚实结论**：我们的调度是"V1-style 简化版"——没有 SLO 感知优先级、没有多步 decode 的
调度语义、没有 PD 分离。vLLM V1 的完整调度器（token 预算 + 优先级 + 抢占策略）值得逐行深读。
**追问应对**：被问"生产调度还缺什么"→ 答 SLO（TTFT/TPOT 目标）、优先级队列、multi-step
decode 的调度、preemption 策略参数化（swap vs recompute 的成本模型）。

### 2.2 KV cache 与内存管理（paged / 前缀缓存 / COW / 分块 prefill / KV swap）

**机制**：BlockManager 管理固定块池 + `hash_to_block_id`；块哈希是链式
（xxhash(token_ids) + 前块哈希 8 字节 LE）；**部分块也缓存**（末块按实际 token 数记账）；
写共享块前 COW 复制（GPU 整块 K/V 拷贝，`cow_pairs` 在 run 前执行）；哈希条目删除带守卫
（两个相同内容块可共享同一哈希）；KV swap 用独立 swapped 队列 + CPU 非 pinned 缓冲 +
`kv_swap_space_gb` 预算。

**必问必答**：
- **Q：paged attention 与 vLLM 的差异？** A：块大小 256 vs vLLM 16（影响内部碎片与哈希
  粒度）；我们的 COW 在调度器记账、引擎 step 里执行；vLLM 的 block manager 更细（
  BlockAllocator 分 CPU/GPU、前缀缓存与 COW 的块级记账）。
- **Q：COW 的安全性怎么保证？** A：写起点落在共享块（ref_count>1）时复制一块并换表；
  复制对在 GPU 上执行；被复制块保留 refcount 直到写完成；哈希条目删除带守卫防止误删
  他人条目。
- **Q：KV swap 为什么 bit-exact？** A：KV 内容原样拷 CPU、换入时 `index_copy_` 原位写回
  新分配的私有块（不查前缀缓存），恢复后采样流确定。**坑**：高级索引
  `kv_cache[:, :, ids]` 返回副本，`.copy_` 只改副本——必须 `index_copy_`（故事 14）。
- **Q：swap vs recompute 怎么选？** A：成本模型：swap = D2H+H2D 拷贝带宽；recompute =
  重新 prefill 的计算量。7B+ 重算贵（几十 ms/步）→ swap 赢；0.6B 重算便宜 + WSL2 D2H 慢
  → recompute 赢。实测 27.8s vs 11.8s（本机）。

**数字**：前缀缓存跨批次 prefill 降为 0 token/0 步；FP8 KV 容量 1.9×（421→802 块）；
KV swap bitexact 0 误差、96×512 压力 699 次换出、27.8s vs recompute 11.8s。
**诚实结论**：滚动缓冲（SWA 真省显存）未做——flash 从块表索引推导 key 位置，滚动表需
flash fork 或自研内核（vLLM 也是只掩码不滚动）。
**追问应对**：被问"前缀缓存怎么失效"→ 内容哈希链式、被拒草稿永不进哈希（投机）、
COW 副本重新发布哈希。

### 2.3 CUDA 内核与性能工程（6 个 Triton 内核 + CUDA graph + roofline 归因）

**机制**：自研内核按形态分两类——**计算型 GEMM**（int8/int4/fp8/sparse24，M-adaptive
tile：小 M 用 16×128 权重带宽主导、大 M 用 16×128（int4，roofline 搜索后））与**访存型
注意力**（fp8 KV decode/varlen 内核，GQA 融合 + MMA + 寄存器内反量化，BLOCK_T=32/warps=1
实测最准最快）。CUDA graph：decode 按 batch 容量族捕获共享内存池；spec 步按"行容量 ×
双 stride"捕获，真实行 + 零长度填充行重放（bit-exact）。

**必问必答**：
- **Q：你的 int4 内核为什么小 M 赢、大 M 输？** A：小 M（decode）是权重带宽主导——int4
  权重字节 = bf16 的 1/4，带宽减半 → 实测 gate_up M=8 4.36×；大 M（prefill）是计算主导，
  MMA 计数与稠密相同 + 反量化开销 → 之前 0.2-0.6×。**结论：软件低比特 GEMM 是带宽优化，
  不是计算优化**——双路径路由的动机。
- **Q：roofline 归因怎么做？** A：先用实测标定上界（cuBLAS 大 GEMM 测 TC 峰值 48.5 TFLOPS、
  D2D copy 测带宽 370 GB/s），再按 arithmetic intensity（AI=2MNK/流量字节）分类：AI≥128
  算力受限、AI<128 带宽受限，还有第三种"启动/并行度受限"（小 M 时两者都远低于峰值）。
  归因结果：fp8 大 M 到 **81% TC**、int4 大 M 只有 44%（寄存器内反量化限制）、decode 小 M
  是启动受限。
- **Q：手写 MatMul 到什么水平？** A：SMEM-tiled fp16（Triton 控制 tile/流水/线程）达
  **101% cuBLAS**（4096³/16384³）。消融：GROUP_M swizzle 只 +4%（大矩阵 L2 收益有限）；
  stages=2 最优（三缓冲挤 SMEM，四缓冲 OOM）；**最优配置 occupancy 只有 17%**——大 GEMM
  是 TC 吞吐型，occupancy 不是瓶颈（反直觉，面试反例）。
- **Q：tile 搜索找到什么？** A：**int4 大 M 用 BM16/BN128（regs=128）反超 BM64/BN256
  （regs=255）19%**——大 tile 的 acc 累加器把寄存器打到 255 上限、占用掉到 1 block/SM；
  小 tile 2 blocks。低比特内核的"反量化在寄存器里做"让大 tile 的寄存器成本尤其高。
  落地后**纯 int4 大 batch +28%**（0.64×→0.82× fp16）。
- **Q：自研 fp8 注意力内核的关键决策？** A：①GQA 融合（一 program 处理 seq×kv_head 的整组
  q 头，KV 只读一次）；②直接 fp8 load + 硬件 cvt 反量化（无 LUT gather）；③MMA 计算
  （QPAD 填充到 16 满足 dot 的 N≥16，8× 计算浪费换内存效率——decode 是 memory-bound）；
  ④num_warps=1（跨 warp 归约顺序变化放大误差）。归因：有效 KV 读带宽 517-529 GB/s
  （超过 copy 370 的双向口径）——**带宽型内核，算力远未饱和**。
- **Q：CUDA graph 的坑？** A：形状必须静态（按容量族）；共享内存池避免碎片；spec 图用
  尾部重复 cu_seqlens 的空行填充（flash grid 按容量烘焙，重放只是数据）——bit-exact 用
  probe 验证过。

**数字**：硬件锚点（实测）TC 峰值 **48.5 TFLOPS**、带宽 **370 GB/s**、36 SM、SMEM 100KB/SM；
手写 MatMul **101% cuBLAS**；int4 gate_up M=8 **4.36×**、lm_head 3.42×、down_proj 0.40×；
fp8 M=8 4.17×、M=256 1.98×（scaled_mm）；fp8 大 M Triton **81% TC**；**int4 大 M tile 改进
+19% → 纯 int4 大 batch 3916.5 tok/s（0.82× fp16，原 0.64×）**；fp8 decode 内核逐层
1.9→1.15ms；verify 启动税 ~10ms/步被 graph 消除。
**诚实结论**：已做 roofline 归因 + 系统性 tile 搜索（`benchmarks/_kernel_roofline.md`）；
**CUDA C 补课已落地**（`benchmarks/_cuda_gemm_report.md`：nvcc 12.8 工具链 + 手写 fp16 GEMM
到 cuBLAS 53% + bank-conflict 消融 + split-K/persistent 边界，全部 SASS 实证）；fp8 大 M
Triton +8.6% 未落地（引擎走硬件 scaled_mm）；搜索方法论教训：**快速搜索的 "+144%" 异常值
被高迭代复测推翻**。
**追问应对**：被问"还能怎么快"→ cp.async/TMA 双缓冲流水（v2b 之后的主差距）、ldmatrix
canonical 布局、KV 块排序提 L2 命中、把 w_deq 降精度存储（fp8 引入 dequant 流量不划算）；
persistent/split-K 已实测（本机 persistent 亏、split-K 仅小 M 赢——被追问时给条件不给
背书）。

### 2.4 量化（五条路径的取舍 + 精度方法论）

**机制**：w8a8（per-group 128 int8 + per-token int8 + SmoothQuant 平滑）、int4（per-group
128 对称 + 2-dot 反量化内核 + 双路径）、AWQ（α 搜索缩放折叠进权重）、fp8（e4m3 全量化：
per-column 权重 + per-token 激活）、sparse24（2:4 幅值剪枝）。FP8 KV cache 单独一条线。

**必问必答**：
- **Q：AWQ 为什么有效？** A：大激活通道的权重误差贡献大——把 s 折进权重（W'=W·s）让大
  激活通道的量化相对误差变小，激活侧除 s（X'=X/s）压小误差贡献。**方向必须对**（权重乘、
  激活除）；方向错了 α 搜索会假装"不缩放最优"。实测 ppl 4.38→3.76（把 int4 相对 fp16 的
  差距砍半）。
- **Q：fp8 为什么近乎无损？** A：e4m3 的 3 位尾数 + per-column scale + per-token 动态激活
  scale；ppl 3.60 vs fp16 3.60（+0.1%）；KL 0.017、top-1 100%。
- **Q：2:4 稀疏为什么失败？** A：内核 bit-exact，但**一次性幅值剪枝丢 ~35% 权重质量**
  （KL 8.5）——SparseGPT 式误差补偿或剪枝感知训练是修复路线；且 sm_120 上 cuSPARSELt
  每调用开销 0.3-0.5ms、CUTLASS 仅 sm_8x。
- **Q：量化精度怎么测才可信？** A：8-prompt KL 被尾部单点主导（曾与 ppl 结论相反）→
  换真实文本困惑度（3000+ token）。**指标的样本量决定结论方向**。

**数字**：0.6B ppl：fp16 3.60 / fp8 3.60 / int4 4.81 / awq 4.22；KL：fp8 0.017、w8a8
0.0379、int4 1.08；Qwen2.5-0.5B ppl fp16 5.12 / int4 7.45（0.5B 量化鲁棒性弱）。
**诚实结论**：int4 双路径的 w_deq 副本让显存 1.73GB 比 fp16 还大——吞吐无损的定价。
**追问应对**：被问"为什么不用 GPTQ"→ GPTQ 用 Hessian 逆做逐列误差补偿，理论上优于 RTN；
我们没实现——这是已知差距（§6 路线 4）。

### 2.5 投机解码（n-gram / Medusa / EAGLE）

**机制**：verify 步 = 带前缀复用的 varlen prefill（query = [末 token, 草稿...]，位置从
len-1 起，num_cached=len-1）；接受规则（Leviathan et al.）：草稿是点质量分布，"接受 iff
目标采样==草稿"严格保持分布；被拒草稿永不进前缀缓存哈希；verify 步 CUDA graph 化（行容量
族 + 双 stride）。三个草稿源：n-gram（历史窗口搜索，零成本）、Medusa（γ+1 个 MLP 头）、
EAGLE（无 RoPE 草稿层 + 共享 LM head 自回归）。

**必问必答**：
- **Q：投机解码的成本模型？** A：期望加速 = (1+αγ)/(γ·T_draft+T_verify+1)，α = 草稿
  接受率；**上界 α ≤ 模型 top-1 可预测性**（实测自由文本 35%）——这解释了为什么所有方案
  在 free 文本上只有 ~1.5-2×、重复内容上 3-4×。
- **Q：为什么 EAGLE γ=2 赢、γ=4 输？** A：γ=4 每草稿一次 LM head 前向（0.6B 上 ~0.8ms）
  + 特征误差累积（草稿质量随深度下降）；γ=2 的 (1+αγ) 收益 > 成本。实测 γ=2 repeat
  +3.26×（α 0.525）、γ=4 只有 +0.63×。
- **Q：verify 步为什么用 varlen prefill 而不是 decode？** A：多草稿是"一行多个 query token"
  ——本质是变长的小 prefill；复用分块 prefill + 前缀复用路径（缓存形状 K/V + block tables），
  不需要新的注意力形态。
- **Q：fp8 KV + 投机怎么结合？** A：verify 步走自研 fp8 varlen 内核直接读 fp8 缓存（免
  逐层全缓存反量化，~18GB/步的搬运）——从 0.15× 变 +3.92×。

**数字**：ngram bs=8 repeat +3.87×、bs=256 +1.43×；Medusa bs=8 +1.57×；EAGLE γ=2 +3.26×；
free 文本全部 ~1.5-2×（α 0.19-0.23）；0.6B top-1 可预测性 35%。
**诚实结论**：α 被模型可预测性封顶——投机在"模型太笨"时赚不到；这是理解投机解码的关键。

### 2.6 系统与工程（流式加载 / 多模型 / TP / 验证方法论）

**机制**：按层流式加载（meta 构造 → 逐层 to_empty 物化 → 加载即量化 → 释放 fp16）；
注册表按 model_type 分发；TP 用 NCCL + 共享内存命令通道（weight_loader 分片 + all_reduce）；
验证方法论 = HF 参考 logits 对照（top-1 100%）+ 内核独立对照 + 引擎级 smoke。

**必问必答**：
- **Q：meta 物化踩了什么坑？** A：①torch 2.8 禁止 meta→真实设备 `.to()`，必须 `to_empty`；
  ②to_empty 替换 Parameter 丢 weight_loader → 按模块重挂；③RoPE 的 cos_sin_cache 是计算型
  buffer，物化后全零 → q/k 被零旋转逐层发散 → 必须 build_cache() 重建；④tie 词表重绑。
- **Q：16GB 卡怎么跑 7B+？** A：按层加载 + 即时量化——任一时刻显存 ≈ 累计量化权重 +
  单个 fp16 层 + embed；7B int4 峰值 10.7-11.6GB。自动触发阈值 = fp16 估重 > 空闲显存 45%。
- **Q：新架构端口（Gemma-2）怎么验证？** A：HF parity 逐层二分——embed → 层0 → 层1 →
  hidden，每个中间量对比；发现三个隐藏细节（embed ×√d、RMSNorm (1+w)、双残差四 norm），
  都是"checkpoint 里看不出来、读源码才能发现"的初始化语义。**教训：debug 脚本自身的
  形状/口径也要先钉对，否则拿到的 diff 全是假象**。
- **Q：TP 为什么没实测多卡？** A：单卡环境；实现完整（weight_loader 分片 + NCCL 命令通道
  + 序列跨进程），但 multi-GPU 验证是已知空白（§6 路线 5 的理论补强）。

**数字**：Qwen2.5-7B int4 峰值 10.66GB / Llama-3.1-8B 11.62GB / Mistral-7B 11.37GB；
parity：qwen2.5-0.5B top-1 100%（mean 0.096）、mistral 100%（0.014）、gemma2 100%（0.022）。
**诚实结论**：PP/DP/EP 未实现；TP 未实测；CacheBlend、PD 分离只在文档里设计过。

---

## 3. 数字速查（全部带条件：RTX 5060 Ti 16GB / WSL2 / bf16 / 单卡，除非另注）

| 项 | 数字 | 条件 |
|---|---|---|
| 引擎吞吐峰值 | **5825 tok/s**（fp8，1.22× fp16）；fp16 基线 **4792 tok/s** | Qwen3-0.6B，bs=256，干净 workload |
| vLLM 对比 | 吞吐 1.35-1.61× 领先；prefill ~2×；**decode 单步持平** | 同 workload 同 seed 同 flash-attn |
| 混合调度 | 吞吐 +7~21%；抢占减半 | 全 batch 档 |
| FP8 KV | 容量 1.9×；KL 0.0073；top-1 100% | fp8_e4m3 + 自研内核 |
| fp8 权重 | ppl +0.1%（3.60）；bs=256 5825 tok/s；8B prefill TTFT 210 vs int4 813ms | e4m3 全量化 |
| int4（双路径） | bs=8 +35%、bs=256 +3~6%；ppl 4.81；显存 1.73GB | w_deq 副本 |
| int4（纯 int4 模式） | 显存 0.85GB（0.57×）；**bs=256 3916.5 tok/s（0.82× fp16，tile 搜索后从 0.64× +28%）** | 大 batch 仍慢于 fp16，显存优先选项 |
| int4 微基准 | gate_up M=8 **4.36×**、lm_head 3.42×、qkv 1.6-2.0×；down_proj **0.40×**、M≥128 全输 0.2-0.6× | 权重带宽主导才赢 |
| AWQ | ppl 4.38→3.76；112 层 α 搜索全赢 | 真实文本校准 |
| W8A8 | KL 0.0379；吞吐 -16% | per-group 128 + SmoothQuant |
| 2:4 稀疏 | 内核 bit-exact；KL 8.5；cuSPARSELt 0.02-0.17× | 一次性幅值剪枝 |
| n-gram spec | bs=8 repeat +3.87×；bs=256 +1.43× | verify graph 后 |
| Medusa | bs=8 repeat +1.57×；head_0 = 模型 top-1 的 87% | 自蒸馏训练 ~7min |
| EAGLE-1 | γ=2 repeat +3.26×（α 0.525）；γ=4 +0.63× | 0.6B，重复内容 |
| 模型可预测性 | 自由文本 top-1 35% → 投机 α 天花板 | 0.6B |
| KV swap | bitexact 0 误差；699 次换出；**27.8s vs recompute 11.8s（亏）** | 0.6B+WSL2 |
| 流式加载 | Qwen2.5-7B 峰值 10.66GB / 8B 11.62GB / Mistral 11.37GB | int4，16GB 卡 |
| 端口 parity | qwen2.5 100%（0.096）/ mistral 100%（0.014）/ gemma2 100%（0.022） | prefill logits vs HF |
| 模型吞吐 | Llama-3.1-8B 303.5 tok/s（bs=16）；Mistral-7B 311.4（bs=32）；Gemma-2-2B 1095（bs=64） | int4 |
| SWA | window 约定 (W-1,0)；4876-token 跨窗口跑通（957 tok/s prefill） | Mistral-7B |
| soft-cap | attn cap=50 最大改 logits 1.7%；final cap=30 必须实现 | gemma-2-2b |
| 精度方法论 | 8-prompt KL 与 ppl 结论相反 → 用 3000+ token 困惑度 | 样本量决定结论 |

---

## 4. 踩坑故事精选（讲"方法论"而非"事故"）

1. **独立检查通过 ≠ 引擎正确**（fp8 varlen 差 1 掩码）：随机数据检查通过、真实数据 logits
   差 9-25——self-attention 的"自己分量"放大差 1。教训：独立验证必须覆盖真实分布。
2. **小形状通过 ≠ 内核正确**（int4 打包列缺偏移）：小 N 检查全过、大 N 全错——按 N 块打印
   误差分布一眼定位。教训：内核测试要覆盖多块路径。
3. **8-prompt KL 与 ppl 结论相反**：被尾部单点主导——换 3060-token 困惑度后 AWG 收益才显现。
   教训：指标的敏感度决定结论方向。
4. **meta 物化丢计算 buffer（RoPE 全零）**：逐层 hook 对比定位（embed 一致、layer0 首 token
   精确、RoPE 后 q/k 差 80-130 → 锁死 RoPE）；"buffer 对比通过"是 lru_cache 共享实例的假象。
   教训：检查脚本必须清缓存再重建。
5. **AWQ 方向三连错**：`W·s且X·s` → 反方向 → 论文方向才收敛；方向错时 α 搜索假装"不缩放
   最优"。教训：先独立验证恒等式再信搜索。
6. **高级索引写回是临时副本（KV swap 静默写垃圾）**：输出全错但长度/无崩溃全过——bitexact
   测试（读-写-读回对比）才抓到。教训："跑通"不等于"写对"，写回操作必须验证内容。
7. **Gemma-2 三个"读源码才能发现"的架构细节**：parity top-1 0% 逼出 embed ×√d（48×）、
   RMSNorm (1+w)（比例 9.57 与 1.1167/0.1167 精确吻合）、双残差四 norm。教训：新架构端口的
   唯一可靠验证是逐层对照 HF，且 debug 脚本自身口径先钉对。
8. **flash window_size 含两端（off-by-one）**：probe 同时演示 (W,0) 与 (W-1,0)——API 的
   "窗口大小"语义不唯一，先用最小对照实验钉死。
9. **fp8 内核窗口掩码的 m=-inf 全掩块 NaN**：掩码改变"首块有效"假设——给内核加掩码要检查
   m/l 累加器的空块行为。
10. **Triton JIT 编译落进计时区间（TTFT 435ms 假象）**：bench 预热必须用真实 prefill 形状。
    教训：任何带新 Triton 内核的功能都要验证基准口径。
11. **投机多 token 接受跳过 max_tokens（`_maybe_finish` 的 `==`）**：投机步一次接受 2+ token
    时精确相等永不命中 → 序列一路长到 max_model_len 溢出 spec graph 容量。ngram 的草稿预算
    恰好避免，EAGLE 没加就暴露。教训：**预存 bug 常由新路径触发**——结束条件用精确相等
    在多 token 步进下必然漏判，改 `>=`。

---

## 5. 方法论总结（如何证明你懂推理系统）

- **先讲成本模型与上界**：投机 γ·T_draft+T_verify vs (1+αγ)、α≤模型 top-1；swap 的
  D2H 带宽 vs 重算；int4 的带宽 vs 计算主导。
- **讲"我证伪过什么"**：大 tile 假设、分页瓶颈假设、启动税假设——证明有实证习惯而不是背书。
- **主动说"哪里是亏的"**：KV swap 本机亏、2:4 精度灾难、纯 int4 大 batch 慢、EAGLE γ=4 亏——
  比吹嘘可信。
- **数字带条件**：单卡、WSL2、0.6B/7B/2B、flash-attn 版本、dtype。
- **口径诚实**：vLLM 离线 API 不暴露逐请求指标 → 用聚合直方图并声明近似；decode 单步
  才是 kernel 级可比口径。
- **亮点句**："vLLM 在这张卡上跑不了 fp8 KV（FA3 是 Hopper-only），我的自研内核是唯一可跑
  的实现，且精度 KL 0.0073、top-1 100%。"

---

## 6. 深度精进路线图（往深走：顺序与理由）

> 原则：按 **面试追问频率 × 技能可迁移性 × 本机可验证性** 加权排序；每个阶段给出
> "具体动作 + 交付物 + 为什么"。穿插项：每个阶段读对应 vLLM/llama.cpp 源码做对照。

### 阶段 1：内核与性能工程（CUDA 记忆模型 + roofline 归因 + 手写 MatMul）——**✅ 已完成**
**交付（`benchmarks/_kernel_roofline.md` + `benchmarks/_cuda_gemm_report.md`）**：实测
标定硬件锚点（TC 48.5 TFLOPS / 带宽 370 GB/s / 36 SM）；GEMM 三种性能形态（算力/带宽/
启动受限）归因表（fp8 大 M 81% TC、int4 44% TC、decode 小 M 启动受限）；手写 SMEM-tiled
fp16 MatMul **101% cuBLAS** + GROUP_M/stages/regs/occupancy 消融（最优配置 occupancy
仅 17%——TC 吞吐型反直觉反例）；tile 网格搜索 → **int4 大 M 小 tile 反超 19%（regs
255→128）→ 落地后纯 int4 大 batch +28%（0.64×→0.82× fp16）**；方法论教训：快速搜索的
+144% 异常值被高迭代复测推翻。
**CUDA C 补课（阶段 1b，全部 SASS 实证）**：工具链四坑打通（pip nvcc wheel 拆包只剩
ptxas → conda nvcc 12.8.93；gcc15 崩 pybind11 → conda gcc14；CUDAHOSTCXX 失效 →
gcc symlink；cuobjdump 12.4 无法解码 SM120 → 12.8）。手写 fp16 GEMM 四步：FMA naive
1.6 → mma 单 tile 6.1 → **8 tile/warp + BsT 转置布局 20.8 TFLOPS（cuBLAS 53%）**，
每步 SASS 证据（NOP/mma 6.5→1.3 是 TC 延迟隐藏的可视化指标）；**bank-conflict 消融：
BsT 行距 32→34 消 16-way 写冲突，实测 +86%**（bank = 行距与 32 的 gcd 决定冲突度）；
split-K 只在 block 数 < SM 数时赢（M=64 S=4 +59%），并行度够时部分和流量纯亏；
persistent 本机全亏（0.89×，硬件 block 分发近零成本 + 动态均衡更优）。

### 阶段 2：MLA（DeepSeek 潜在注意力）+ SWA 滚动缓冲——**第二**
**为什么第二**：注意力是推理的核心，DeepSeek 是当前面试必考；MLA 有真模型可验证
（DeepSeek-V2-Lite int4≈8GB 本机可跑）；滚动缓冲把文档里的 TODO 变完成，且正好用上
阶段 1 的内核能力（滚动表的 key 位置偏移需要自研内核）。**具体动作**：①实现 MLA 层
（latent 压缩 c_KV、decoupled rope、权重绑定 W_UK=W_DKVᵀ）+ KV cache 布局泛化 + naive
对照单测（0 误差）；②DeepSeek-V2-Lite 端口 + HF parity；③KV 压缩率账本（V2-Lite 每
token 每层 512+64×16 vs 等效 GQA 4096 ≈ 2.7×；V3 官方口径 576 的推导）；④滚动缓冲：
per-seq 物理环 + refcount 守卫 + bf16 decode/varlen 内核的窗口位置偏移。

### 阶段 3：调度系统深读（vLLM V1 源码对照 + SLO + multi-step decode）——**第三**
**为什么第三**：调度是 vLLM 面试核心话题；我们的实现是"V1-style 简化版"，逐行读 vLLM
找差距 = 把概念钉死（不需要 GPU）。**具体动作**：①读 vLLM V1 scheduler/block_manager/
preemption 源码，产出"vLLM vs nano"逐项差距表；②实现 SLO-aware 优先级调度（TTFT/TPOT
目标约束 + 优先级队列）；③multi-step decode（一次调度多步 decode，减少 kernel 启动与
CPU 空转）。

### 阶段 4：量化/稀疏算法层（GPTQ 误差补偿 + 剪枝感知）——**第四**
**为什么第四**：现有量化是应用层（RTN/AWQ/fp8 的工程实现），补算法层才能答"为什么 AWQ
有效、2:4 怎么不丢精度、GPTQ 和 RTN 差在哪"。**具体动作**：①实现 GPTQ（Hessian 逆 +
逐列误差补偿）在 0.6B 上与 RTN/AWQ 对比 ppl；②SparseGPT 式误差补偿稀疏（把 2:4 的
KL 8.5 修到可用）；③精度方法论：校准集设计、离群通道分析、误差传播（逐层累积曲线）。

### 阶段 5：分布式推理理论（PP/DP/EP + NCCL 集体通信）——**最后**
**为什么最后**：单卡无法实测，性价比最低；作为理论补强。**具体动作**：PP 的 1F1B 内存
分析（bubble 比例 = (p-1)/(m+p-1)）与切分策略、EP 的路由 + 通信量、NCCL allreduce 的
环/树带宽模型；用 paper 推导 + 数值模拟验证（无实机）。

### 穿插项（每阶段做一块）
- vLLM：attention backends（阶段1 对照内核）、scheduler（阶段3）、quant（阶段4）；
- llama.cpp：GGUF 量化与 kernel 设计（阶段1/4）；
- 论文：FlashAttention（阶段1）、MLA 原论文（阶段2）、PD 分离/Mooncake（阶段3）、GPTQ/
  AWQ/SmoothQuant（阶段4）、Megatron 1F1B/DeepSeek MoE（阶段5）。

---

## 7. 附录：代码地图（功能 → 文件 → 函数 + 运行链）

> 用途：精读源码时的定位索引（行号为当前版本真实位置，可直接跳转）。先看运行链（§7.3），
> 再按 §6 阶段顺序逐模块读；`git log` 的 commit message 是"为什么这么写"的第一手材料
> （本项目每个 commit 都带动机）。

### 7.1 文件地图（谁是谁）

| 文件 | 职责 | 核心类/函数 |
|---|---|---|
| `nanovllm/llm.py` | 公共 API 入口 | `LLM(LLMEngine)` L4——纯别名，没逻辑 |
| `nanovllm/config.py` | 引擎配置解析 | `Config` L7、`__post_init__` L36（断言合法性） |
| `nanovllm/sampling_params.py` | 采样参数 | `SamplingParams` L6 |
| `nanovllm/engine/llm_engine.py` | **引擎主循环** | `LLMEngine` L17：`add_request` L53 / `step` L203 / `generate` L290 / `_verify` L61 / `_medusa_drafts` L99 / `_eagle_drafts` L144 / `collect_metrics` L353 |
| `nanovllm/engine/scheduler.py` | **调度器**（批组成、抢占、swap） | `Scheduler` L12：`schedule` L54 / `_schedule_mixed` L111 / `_schedule_prefill` L258 / `_schedule_decode` L309 / `_schedule_spec` L212 / `preempt` L337 / `swap_out` L361 / `swap_in` L394 / `postprocess` L432 / `postprocess_spec` L446 |
| `nanovllm/engine/sequence.py` | 序列状态（CPU 侧唯一真源） | `Sequence` L15、`SequenceStatus` L8 |
| `nanovllm/engine/block_manager.py` | **KV 块池 + 前缀缓存 + COW** | `BlockManager` L26：`compute_hash` L37 / `can_allocate` L62 / `allocate` L92 / `cow_block` L183 / `hash_blocks` L209 / `can_append` L146 / `can_append_spec` L158 |
| `nanovllm/engine/model_runner.py` | **批处理打包 + GPU 执行** | `ModelRunner` L16：`__init__` L18（启动链）/ `call` L163（TP）/ `warmup_model` L202 / 各 `quantize_*` L266-285 / `allocate_kv_cache` L477 / `prepare_prefill` L521 / `prepare_mixed` L572 / `prepare_spec` L642 / `prepare_decode` L747 / `run_model` L776 / `capture_cudagraph` L945 / `capture_spec_graph` L858 / `run` L916 |
| `nanovllm/engine/ngram.py` | n-gram 投机（纯函数，无状态） | `find_ngram_draft` L15、`verify_drafts` L54 |
| `nanovllm/models/registry.py` | 按 `model_type` 选模型类 | `get_model_class` L44 |
| `nanovllm/models/qwen3.py` 等 5 个 | 模型定义（结构模板完全一致） | `*ForCausalLM`（含 `packed_modules_mapping`）/ `*Model` / `*DecoderLayer` / `*Attention` / `*MLP` / `compute_logits` |
| `nanovllm/layers/attention.py` | **注意力：写 KV + flash/fp8 路由** | `store_kvcache` L33、`paged_decode_attention_fp8` L115、`paged_varlen_attention_fp8` L207、`Attention.forward` L268 |
| `nanovllm/layers/linear.py` | **全部 GEMM + 权重量化** | Triton 内核：`gemm_int8_kernel` L24 / `gemm_int4_kernel` L98 / `gemm_fp8_kernel` L298 / `gemm_sparse24_kernel` L183；封装：`int4_gemm` L153 / `fp8_gemm` L346 / `w8a8_gemm` L74 / `sparse24_gemm` L248；量化：`WeightQuantMixin` L372（`quantize_int4` L386 / `quantize_fp8` L448 / `quantize_sparse24` L489）；并行层：`ColumnParallelLinear` L619 / `MergedColumnParallelLinear` L649 / `QKVParallelLinear` L669 / `RowParallelLinear` L704 |
| `nanovllm/layers/layernorm.py` | RMSNorm（含 Gemma-2 变体） | `RMSNorm` L5：`rms_forward` L24 / `add_rms_forward` L37 |
| `nanovllm/layers/rotary_embedding.py` | RoPE（含 Llama-3 缩放） | `RotaryEmbedding` L48、`build_cache` L73、`get_rope` L111 |
| `nanovllm/layers/activation.py` | SwiGLU 融合激活 | `SiluAndMul` L6 |
| `nanovllm/layers/sampler.py` | Gumbel 采样 | `Sampler` L5 |
| `nanovllm/layers/embed_head.py` | Embedding + LM Head（TP 感知） | `VocabParallelEmbedding` L10、`ParallelLMHead` L46（继承 WeightQuantMixin） |
| `nanovllm/layers/medusa.py` / `eagle.py` | 投机草稿头 | `MedusaHeads` L39 / `EagleLayer` L46 |
| `nanovllm/utils/context.py` | **每步张量的全局契约** | `Context` L6、`set_context` L27、`get_context` L24、`reset_context` L37 |
| `nanovllm/utils/loader.py` | 权重加载（eager + 流式） | `load_model` L32、`_load_eager` L54、`_load_streaming` L85、`default_weight_loader` L8 |

### 7.2 功能 → 文件 → 函数（按学习主题）

**① 入口与配置**：用户入口 `llm.py:4`；全部引擎参数 `config.py:7`（quantization /
speculative / tensor_parallel_size / kv_swap_space_gb / max_num_batched_tokens）；
采样参数校验 `sampling_params.py:6`（禁 greedy——Sampler 用 Gumbel，必须 temperature>1e-10）。

**② 引擎主循环**（必读核心）：提交请求 `llm_engine.py:53 add_request`；**每步推进**
`llm_engine.py:203 step`（调度→COW/swap→跑模型→采样→后处理→草稿）；外层循环
`llm_engine.py:290 generate`（含逐步吞吐统计与输出 decode）；投机验收
`llm_engine.py:61 _verify`；Medusa/EAGLE 下轮草稿 `llm_engine.py:99 _medusa_drafts` /
`:144 _eagle_drafts`；基准指标导出 `llm_engine.py:353 collect_metrics`。

**③ 调度与抢占**：决定本步 kind（prefill/decode/mixed/spec）`scheduler.py:54 schedule`
（先 `_try_swap_in`；waiting+running 都非空 → mixed）；mixed 批组成（prefill 在前、
decode 在后共享 token 预算）`scheduler.py:111`（vLLM V1 同款）；prefill/decode 批
`scheduler.py:258 / :309`（只有首个序列可被 chunk 拆分）；KV 不足抢占 `scheduler.py:337
preempt`（decode/spec 优先 swap_out）；swap 换出/换入 `scheduler.py:361 / :394 / :404`
（换入优先，bit-exact 免重 prefill）；步后处理 `scheduler.py:432 postprocess` /
`:446 postprocess_spec`（spec 版只提交被接受 token）。

**④ KV Cache：块管理 + 前缀缓存 + COW + swap**：块哈希链（xxhash + 前块哈希）
`block_manager.py:37 compute_hash`；能否复用缓存块 `block_manager.py:62 can_allocate`
（检查部分块 ceiling end）；分配/共享块（refcount）`block_manager.py:92 allocate`；
**写共享块前的复制** `block_manager.py:183 cow_block`（返回 (old,new) 对 → GPU 侧拷贝在
`step` 里 `model_runner.call("cow_block")`）；新哈希发布 `block_manager.py:209 hash_blocks`
（带 guard 删除，防 COW 副本撞哈希）；decode/spec 追加 `block_manager.py:146 can_append` /
`:158 can_append_spec` / `:175 may_append_spec`（跨块写跨度）；释放 `block_manager.py:137
deallocate`；GPU 侧拷贝/swap `model_runner.py:171 cow_block` / `:180 swap_out` /
`:190 swap_in`（swap_in 必须 `index_copy_`——list 高级索引返回临时副本会静默写垃圾）。

**⑤ 批处理打包（prepare_* + Context 契约）**：KV cache 分配+绑层 `model_runner.py:477
allocate_kv_cache`（`[2, L, num_blocks, block_size, kv_heads, head_dim]`）；prefill 打包
`model_runner.py:521 prepare_prefill`（含 chunked seq 的缓存形状 K/V）；mixed 打包
`model_runner.py:572 prepare_mixed`（设 `Context.is_mixed` / `n_prefill_tokens`）；decode
打包 `model_runner.py:747 prepare_decode`（slot_mapping + context_lens）；spec 打包
（verify 行 = chunked prefill）`model_runner.py:642 prepare_spec` / `:690
_prepare_mixed_spec`；张量交接 `context.py:27 set_context` / `:37 reset_context`（每步 reset）。

**⑥ 模型定义**：注册表 `registry.py:44`；5 个同构模型文件 `models/qwen3.py:212` /
`qwen2.py:209` / `llama3.py:215` / `mistral.py:192` / `gemma2.py:219`；前向链
（embed → L 层 → norm）各 `*Model.forward`（如 `qwen3.py:199`）；层内链（attn + mlp +
残差）各 `*DecoderLayer.forward`（如 `qwen3.py:169`）；logits 各 `compute_logits`
（如 `qwen3.py:243`，LM Head 在图外执行）；HF 权重名→打包参数映射各 `packed_modules_mapping`
（如 `qwen3.py:214`）。

**⑦ 注意力（写 KV + 读路由）**：写 K/V 到分页缓存 `attention.py:33 store_kvcache`
（Triton 按 `slot_mapping` 散写；fp8 写路径先 clamp 448 再 cast 防 NaN 位模式）；fp8
decode 读内核 `attention.py:115 paged_decode_attention_fp8`（寄存器内反量化 + WINDOW 掩码）；
fp8 varlen 读内核 `attention.py:207 paged_varlen_attention_fp8`；**路由总入口**
`attention.py:268 Attention.forward`（先写 KV → 按 is_mixed/is_spec/use_fp8/分块与否选
flash varlen / flash kvcache / 自研 fp8 内核）。

**⑧ 量化 GEMM（`linear.py` 一条龙）**：int4 打包 `[N, K//2]` + per-group 128 scale +
2-dot 去量化 GEMM `linear.py:98` 内核 / `:153 int4_gemm` / `:386 quantize_int4`；int4 形状
路由（dual-path：小 M 走内核、大 M 走 `w_deq` cuBLAS）`linear.py:431 _int4_forward`
（阶段 1 的 BM16/BN128 tile 在这条链上）；fp8 权重-only Triton（小 M）+ 硬件
`torch._scaled_mm`（大 M）`linear.py:346 fp8_gemm` / `:448 quantize_fp8` / `:468
_fp8_forward`；w8a8 SmoothQuant 折叠 `linear.py:24` / `:547` / `:577`；sparse24 2:4 剪枝
`linear.py:183` / `:248` / `:489` / `:513`；引擎侧调用点 `model_runner.py:266
quantize_int4_weights`（streaming 钩子 `:377`）。

**⑨ 投机解码**：n-gram 草稿搜索 + 验收（纯函数）`ngram.py:15 find_ngram_draft` /
`:54 verify_drafts`；每步算草稿 `scheduler.py:90 _compute_draft`；verify 行打包
`model_runner.py:642 prepare_spec`；验收+提交 `llm_engine.py:61 _verify` +
`scheduler.py:446 postprocess_spec`（hash 范围只含接受 token）；spec CUDA graph
`model_runner.py:858 capture_spec_graph` / `:829 _spec_graph_hidden`；Medusa 头 / EAGLE 层
`medusa.py:39` / `eagle.py:46`。

**⑩ 张量并行 / 权重加载**：TP 命令分发（共享内存 + Event）`model_runner.py:163 call` /
`:138 read_shm` / `:130 loop`（worker 死循环）；权重切分各并行层 `weight_loader`
（`linear.py:630/660/687/715`、`embed_head.py:28`）；eager 加载 `loader.py:32 load_model` /
`:54 _load_eager`；**流式加载**（meta 构造 → 逐层物化 → 立即量化）`loader.py:85
_load_streaming` + `model_runner.py:343 _decide_streaming` / `:377 _streaming_quant_hook` /
`:433 _finalize_streaming`（重建 RoPE 缓存）。

### 7.3 运行链

**链 A：进程启动（只跑一次）**

```
LLM(...) → LLMEngine.__init__ [llm_engine.py:19]
 ├─ Config 解析 + tokenizer 加载
 └─ ModelRunner.__init__ [model_runner.py:18]
     ├─ dist.init_process_group("nccl", ...)          # 无条件，TP=1 也初始化
     ├─ get_model_class(model_type) → 选模型类         # registry.py:44
     ├─ _decide_streaming() → load_model(...)          # 大模型走流式（逐层物化+量化）
     ├─ 量化：quantize_int4/fp8/w8a8/awq/sparse24      # model_runner.py:266-285
     ├─ warmup_model()                                 # 真实形状跑一次：JIT编译+测峰值显存
     ├─ allocate_kv_cache()                            # 大块KV + 绑到各Attention层
     ├─ capture_cudagraph()                            # decode图族 [1,2,4,8,16..512]
     └─ capture_spec_graph()                           # 投机：stride家族 × 行容量家族
```

**链 B：每步推理循环（`generate` 内 `while not is_finished()`）**

```
step() [llm_engine.py:203]
 ├─ scheduler.schedule() → (seqs, kind)                # scheduler.py:54
 │   ├─ _try_swap_in()                                 # 先把换出的KV换回
 │   ├─ 投机：先给 running 全算草稿 (_compute_draft)
 │   └─ 分支：waiting+running→mixed | waiting→prefill | 其余→decode/spec
 ├─ COW 拷贝：cow_pairs → call("cow_block")            # run() 之前，prepare 需要新表
 ├─ swap 拷贝：swap_pairs → call("swap_out"/"swap_in")
 ├─ model_runner.call("run", seqs, kind)               # model_runner.py:916
 │   ├─ prepare_prefill/decode/mixed/spec              # 打包 → set_context()
 │   ├─ run_model(input_ids, positions, kind)          # model_runner.py:776
 │   │   ├─ kind=spec → spec CUDA graph 重放（填零长行）
 │   │   ├─ kind=decode 且 bs≤512 且非eager → decode CUDA graph 重放
 │   │   └─ 否则 eager：model(input_ids, positions)    # 模型前向（链C）
 │   ├─ model.compute_logits(hidden)                   # LM Head（图外）
 │   └─ Sampler(logits, temperatures) → token_ids      # Gumbel采样
 │       └─ reset_context()                            # 每步清空契约
 ├─ 投机：_verify() → postprocess_spec()               # 验收+只提交接受token
 │         → _medusa_drafts/_eagle_drafts()            # 用hidden生成下轮草稿
 ├─ 否则：postprocess()                                # 追加token/EOS/rehash
 └─ 收集 finished 序列 → outputs
```

**链 C：单层前向（以 Qwen3 为例，每步每条 token 都走）**

```
Qwen3ForCausalLM.forward [qwen3.py:235]
 └─ Qwen3Model.forward [qwen3.py:199]
     ├─ embed_tokens(input_ids) → hidden
     ├─ for layer in layers: Qwen3DecoderLayer.forward [qwen3.py:169]
     │   ├─ Qwen3Attention.forward [qwen3.py:80]
     │   │   ├─ qkv_proj(x) → q,k,v                      # QKVParallelLinear
     │   │   ├─ RotaryEmbedding(q,k)                     # 按 positions 旋转
     │   │   ├─ Attention.forward [attention.py:268]     # 链D
     │   │   └─ o_proj(o)
     │   ├─ Qwen3MLP.forward [qwen3.py:132]
     │   │   ├─ gate_up_proj(x) → SiluAndMul → down_proj
     │   │   └─ 残差相加（layer norm 走 add_rms_forward）
     ├─ norm(hidden, residual)                           # RMSNorm（残差融合）
     └─ compute_logits → ParallelLMHead                  # 词表映射（TP>1 时 gather）
```

**链 D：Attention 数据流（Context 契约——本项目最核心的接口设计）**

```
prepare_* 构建 GPU 张量 ──set_context()──> Context（全局单例）
   [cu_seqlens_q/k, max_seqlen_q/k, slot_mapping,
    context_lens, block_tables, n_prefill_tokens, is_mixed, is_spec]
                    │
Attention.forward [attention.py:268]  ← get_context()
 ├─ store_kvcache(k, v, k_cache, v_cache, slot_mapping)  # 本步K/V散写入分页缓存
 └─ 读路由（按批次形态）：
     ├─ is_spec        → 全批次 flash_attn_varlen_func（K/V=缓存形状）
     ├─ is_mixed       → prefill组 varlen（分块序列用缓存形状K/V）
     │                    + decode组 fp16→flash_attn_with_kvcache / fp8→自研内核
     ├─ 纯 prefill     → flash_attn_varlen_func（连续K/V）
     └─ 纯 decode      → fp16→flash_attn_with_kvcache / fp8→paged_decode_attention_fp8
```

**链 E：量化路由决策（以 int4 为例）**

```
模型forward里 LinearBase.forward [linear.py:590]
 └─ 已量化? → _int4_forward [linear.py:431]
     ├─ M≤128 且 N≥2048 → Triton int4_gemm（阶段1的 BM16/BN128 tile）
     └─ 否则            → F.linear(x, w_deq)（bf16反量化副本，cuBLAS）
权重来源：quantize_int4 [linear.py:386] 在 warmup 前一次性打包
        （dual-path 同时存 q/scale 和 w_deq；纯 int4 模式不存 w_deq）
```

**链 F：投机解码完整链路**

```
Scheduler._compute_draft [scheduler.py:90]  ─每步CPU─> 写 seq.draft_tokens
 → schedule() → kind="spec"（纯verify）或 "mixed"（prefill在前）
 → prepare_spec/_prepare_mixed_spec [model_runner.py:642/690]
     verify行 = query=[last_token, 草稿...] 的 chunked prefill，num_cached=len-1
 → run_model：spec CUDA graph（stride×容量家族）或 eager varlen
 → LLMEngine._verify [llm_engine.py:61]
     γ+1 行采样 s_i ↔ 草稿 d_i 逐个验收；末行 bonus
 → postprocess_spec [scheduler.py:446]
     只提交接受 token；hash 范围 [num_tokens-n_acc-1, num_tokens-1)（被拒草稿不进前缀缓存）
 → medusa/eagle：_medusa_drafts/_eagle_drafts 用 hidden 生成下轮草稿（写回 draft_tokens）
```

### 7.4 推荐阅读顺序（从浅到深，每步都有可验证出口）

| 步 | 读什么 | 验证出口 |
|---|---|---|
| 1 | `example.py` + `llm.py` + `config.py` | 跑通 `python example.py` |
| 2 | `llm_engine.py` 的 `generate`/`step` | 打断点看每步的 kind 变化 |
| 3 | `scheduler.py` 的 `schedule` + 四个 `_schedule_*` | 打印每步 (seqs, kind) |
| 4 | `model_runner.py` 的 `prepare_*` + `context.py` | 打印 `set_context` 的各张量 shape |
| 5 | `models/qwen3.py`（一个模型吃透，其他 4 个是变体） | 对照 HF 实现看逐层等价 |
| 6 | `layers/attention.py` 的 `forward` 路由 | 三种批次形态各跑一次 |
| 7 | `layers/linear.py`（量化全链） | `--quantization int4` 对比输出 |
| 8 | `block_manager.py`（前缀缓存 + COW） | `--shared-prefix-len 512` 看命中 |
| 9 | `model_runner.py` 的 CUDA graph 两段 | `enforce_eager` 开/关对比 |
| 10 | 投机（`ngram.py` → `_verify` → spec graph）→ TP → 流式加载 | `benchmarks/spec_bench.py` |

穿插阅读：`CLAUDE.md`/`AGENTS.md` 的 Architecture 一节是维护者视角的浓缩；§2 深水区是
这份地图的"人话版"。
