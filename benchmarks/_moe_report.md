# MoE 支持报告（阶段 1.5：router + 循环专家 + 量化 + 引擎验证）

> **阶段快照说明（2026-10-05）**：本报告中的“本机无 MoE 真实模型”和真实模型显存估算是阶段 1.5 当时的记录，不是当前状态。后续阶段已完成 DeepSeek-V2-Lite 的 4 层真权重 parity 和全量流式 int4 推理；见 [`_stage2b_ext_report.md`](_stage2b_ext_report.md)。本报告仍保留随机 Qwen3-MoE toy 的机制验证结果，不能把 toy 精度/吞吐外推到真实模型。
>
> 条件：RTX 5060 Ti 16GB (sm_120) / WSL2 / torch 2.8.0+cu128 / transformers 5.15 /
> 本机无 MoE 真实模型 → 验证用**随机 toy 模型**（Qwen3-MoE 结构、4 层 = 2 dense + 2 MoE
> 混合、E=8/k=2、tie 词表、bf16）——机制正确性金标准 = 与 transformers 同权重同 dtype
> 对照 + 数学同构参考位级对照；**精度/吞吐结论均标注 toy 边界**（随机权重顶层贴边、
> 无真实负载分布）。

## 1. 为什么做 / 放置

- 阶段 2 的验证模型 DeepSeek-V2-Lite 是 **MLA + MoE** 双机制——先单独做 MoE 避免
  两个新机制排错互相污染；MoE 也是 DeepSeek-V3 面试主线。
- 面试要点：MoE 只替换 FFN（router + 专家），**不碰调度/KV/注意力**——引擎侧改动≈0。

## 2. 实现与关键设计决策（含一次被事实推翻的决策）

| 决策 | 过程 |
|---|---|
| 权重布局 | 先按 transformers 5.15 **类属性**做 3D（`gate_up_proj [E,2I,H]`），随后发现其 **checkpoint 存盘是 2D per-expert**（`experts.{i}.gate_proj/up_proj/down_proj`，`use_experts_implementation` 装饰器内存 3D、存盘转 2D 兼容生态）→ 引擎改 **2D ModuleList**：参数名与权重文件直配，**loader 零改动**、量化路径自动继承 |
| loader 匹配 | dense packed key `"up_proj"` 是 `"gate_up_proj"` 的**子串**（旧实现任意子串 replace 会毁掉 MoE 权重名）→ 改"**点分段相等**"匹配（对既有 5 家族语义等价），+3 单测 |
| 前向组织 | 逐专家循环：top-k 路由 → mask → gather 选中行 → ExpertFFN → ×路由概率 → index_add（**串行、e 升序、x dtype 累加**，与 HF 同语义） |
| router | fp32 softmax + topk + 可选 norm_topk_prob；**永不量化**（quantize_exclude——gate 精度决定路由） |
| 验证对照 | `reference`：全行掩码概率向量（无 gather/index_add）——数学同构、同序 → 位级/尾差一致 |
| CUDA graph | 动态 gather 形状 → 不能入图（capture 烘焙形状重放会静默错）→ MoE 模型基准 `enforce_eager`；graph 化需"路由 padding"（未做，见 §7） |

## 3. 验证链（每层都过）

| 层 | 方法 | 结果 |
|---|---|---|
| MoE 层数学 | forward vs reference（GPU fp16，10 场景：E 2..64、k 1..4、T=1 边界、52 空专家、全零输入均匀路由、norm 开关） | 全 PASS，多数 **0 diff**（2 例仅 GEMM K-归约尾差 3e-5，rel 阈值） |
| 同款 CPU 单测 | tests/test_moe.py（5 例：位级/尾差、路由形状、norm 归一、quantize_exclude、k=1/k=E） | pytest 49 passed |
| 装载 | 随机 toy 目录（HF 命名 safetensors + tokenizer）→ loader 全名直配（tie 词表、混合层、3D→2D） | 无异常 |
| **端到端 parity** | 引擎 vs transformers 5.15（同权重同 bf16）prefill logits | **top-1 100%，mean diff 0.003** |
| 量化机制 | int4 引擎结构核验：48 专家线性全量化、0 gate；层误差 fp8 6.4% / int4 12.5%（RTN 预期） | PASS |
| 量化 top-1 假警报 | toy int4/fp8 top-1=0% 初看吓人——证据闭环：toy logits 顶层间距仅 **0.4-0.7σ**，量化 diff 0.5 ≫ gap → 全翻**必然**。随机 toy 不适合 top-1 判据；真实模型精度需真权重 | 判据教训 |

**transformers 兼容性坑**（面试可讲）：transformers 5.15 默认专家 grouped_mm 路径
`torch._grouped_mm` **仅 sm_90+**，sm_120 直接崩 → 需 `_experts_implementation="eager"`
回退逐专家循环（对照与引擎同构）。

## 4. 负载分布与组织税（循环专家实现的性格，实测）

| 观察 | 数字 | 解读 |
|---|---|---|
| 对负载不均 | imbalanced/balanced = 0.20-0.28（k=1/2）；k=8 全激活 sanity ≈1.0 | **串行循环偏好集中路由**（大 gather GEMM 效率高、启动少）；无"最慢专家"瓶颈（那是并行/EP 形态的问题） |
| 固定组织税 | 每层 ~3.5ms/forward，**T 64→4096 时间平坦**（sync 移除后 6.9→3.6ms） | Python 循环 + 每专家 3 次小 GEMM 启动 + 中间张量；**小 T 完全被 CPU 税主导** |
| 同 FLOPs dense 对照 | MoE/dense ≈ 8×（T=4096）~150×（T=256） | 循环专家是"正确性实现"，吞吐上限受组织税约束 |
| sync 移除优化 | `bool(m.any())` host 同步 → 无条件空 gather：**6.9 → 3.6ms（-48%）**，对照不变 | 教训：host 每专家同步是隐藏的 ms 级税 |

## 4b. grouped GEMM 后端（阶段 1.5b：fused/grouped 批量路径，提交 072a1cc）

**实现**（`moe.py`，forward 自动选择后端）：
- **可组条件**：所有专家线性的 float 权重（未量化，或 int4 dual-path 的 `w_deq`
  bf16 副本）→ 3D 堆叠（`[E,2I,H]` gate_up 融合 + `[E,H,I]` down，转置缓存一次）；
  纯 int4 默认回退逐专家循环，但可显式启用下文 4d 的 INT4 分段后端；fp8/w8a8/
  sparse24 仍走逐专家循环（各自量化内核）。
- `_forward_grouped`：top-k → **按专家稳定排序**（段连续）→ 排序行 gather →
  padded `[E, max_n, H]`（每段放回自己的行带，其余 0）→ **单次批量 bmm × 3D
  gate_up（gate/up 融合）→ silu(g)·u → 单次批量 bmm × 3D down** → 反排 →
  加权 index_add（token 的 k 个 slot 合并）。Python 专家循环与每专家启动全消。
- 正确性：10 GPU 场景对照 reference 全 PASS（多数 0 diff，最大 6e-5 = slot 累加
  序尾差）。修过一个错位 bug：段在排序拼接里的 cumsum 起始 vs padded 的 `e*max_n`
  行带不一致（前缀直铺错位；单段场景（全零输入）巧合正确掩盖过它——教训：多段
  场景必须显式 scatter，别用"前缀拼接"捷径）。

**基准**（settled、中位×5；层 H512/I768/E8/k2）：

| T | grouped | loop | loop/grouped |
|---|---|---|---|
| 64 | 0.95 ms | 4.17 ms | **4.4×** |
| 256 | 0.79 ms | 4.13 ms | **5.2×** |
| 1024 | 0.73 ms | 3.48 ms | **4.8×** |
| 4096 | 2.04 ms | 3.79 ms | 1.9× |

固定组织税 **3.8 → ~0.7ms（-80%）**，小 T 批量路径由 bmm 主导（随 T 增长开始出现）。
**不均衡（100%→e0）**：理论 padding 放大 E·max_n/R = 8×，实测时间只 +30%
（vs 均衡 grouped）——小 GEMM 效率收益抵消 padding 计算（该尺度下；大 E/大 T 时
padding 浪费会占优——真段式 grouped（无 padding）是下一步候选）。
**引擎**：int4 dual-path 因 w_deq 自动走 grouped，toy decode 单次 2588 tok/s
（此前 loop 档 927-1110，~2.4-2.8×）；fp16/fp8 档仍在 WSL 时钟噪声带内（引擎
绝对值不可靠，以层级基准为准）。

## 4c. Triton 真段式 grouped 内核（阶段 1.5c，提交 89c210d）

**内核**（`nanovllm/layers/moe_segment.py`）：offsets/counts 驱动的段式 GEMM——
grid (e, m-tile, n-tile)，段行 = offs[e]+t·BM（掩码到 n_e），空 M-tile 提前返回，
K 分块循环 + masked store；**无 padding**；直接复用 grouped 的 3D 堆叠转置权重
（`_gup_t`/`_dn_t`），零额外布局拷贝；一次内核实例化服务 gate_up 与 down 两阶段。
正确性：4 独立场景（E8/E8-big/decode-like/E128）vs 逐段 torch 参考 **rel err = 0.0
（位级）**；`MoE(segment_backend=True)` 接入后 forward vs reference 全 PASS
（≤3e-5）；**引擎 parity（config 字段 `moe_segment_backend`）：top-1 100%、
mean diff 0.0031**（与 bmm 后端一致）。

**实测边界**（K=512/O=1536、E∈{8,128}、sm_120；triton/bmm 时间比）：

| 场景 | 比 | 解读 |
|---|---|---|
| 小段 decode 类（R=24-512，段 <BM） | 3.4-9.6× 慢 | BM 地板 + 空 program；bmm 的 E×max_n 更小 |
| E=128 prefill（R=4K-32K） | 1.05-1.14× 慢 | cuBLAS 128-batch 已高效，padding 仅 1.15-1.4× |
| **E=8 + 极长段（R=32K、段 ~4K 行）** | **0.70× 快** | batched bmm 对"少 batch 每 batch 超大"切分差；单一大 GEMM 赢 |

**结论（诚实）**：真段式内核的正确性完整（位级 + 引擎 parity），但**赢的窗口很窄**
（小 E + 极长段）；现代 MoE（E=128+）cuBLAS padded-bmm 已最优或接近。因此默认后端
保持 padded-bmm，`segment_backend` 是显式开关（窗口内用户可用；也为无 cuBLAS
平台/后续内核优化保留）。

## 4d. 纯 INT4 MoE 专家的分段 GEMM 与 gate/up 融合（2026-10-09）

新增 `moe_int4_backend="segment"`：router 结果按 expert 排序，offset/count 描述每个
expert 的真实段长，packed 权重仍留在原专家 buffer，由小型 device pointer table 寻址；
不填充到 `max_tokens_per_expert`，也不复制整层权重。`moe_int4_fuse_gate_up=True` 时 gate
和 up 共用一个 Triton launch，False 时保留两个独立投影以便消融。默认后端仍是 loop。
支持范围为 TP=1、BF16/FP16 激活、所有 expert projection 使用相同且内核支持的 INT4
group size；AWQ、带 `w_deq` 的 INT4 dual-path、NVFP4、TP>1 走原路径。CUDA Graph 尚未验证。

**实测**：RTX 5060 Ti 16 GB / SM120，WSL2、PyTorch 2.8.0+cu128、Triton 3.4.0；真实
DeepSeek-V2-Lite 第一 MoE 层 H=2048/I=1408/E=64/top-k=6，group64，模型以流式纯 INT4
加载。基准每项先 warmup 3 次，再将 10 次 forward 作为一次 CUDA-event 样本，取 7 个样本中位数：

| 输入 token | loop | segment split | segment fused | loop / split | loop / fused |
|---:|---:|---:|---:|---:|---:|
| 1 | 63.158 ms | 1.205 ms | 1.120 ms | 52.4× | 56.4× |
| 8 | 29.873 ms | 2.490 ms | 2.133 ms | 12.0× | 14.0× |
| 32 | 78.592 ms | 2.951 ms | 2.933 ms | 26.6× | 26.8× |

三种结果对 loop 均为逐位一致（rel error / max abs error = 0）。分阶段计时单独重复各
阶段，彼此不是可相加的 partition；在 8-token 样本中，loop 的 dispatch/sort/gather、
gate/up、SiLU、down、scatter 分别为 11.646/11.040/0.612/2.047/3.588 ms，segment split
为 1.576/0.889/0.093/0.348/0.181 ms，fused 为 0.847/0.892/0.058/0.347/0.170 ms。
主要差异来自移除每专家循环与小 GEMM 启动；gate/up 融合的 GEMM 阶段耗时接近 split，
尚未显示稳定的矩阵计算缩短。

**连续到达消融（首轮历史观察）**：两次独立运行使用同一 fingerprint（seed 7、8 请求、64 输入/16 输出、
constant 4 req/s、TTFT 10 s、TPOT 600 ms、all_on、相同设备与引擎配置）。

| run | backend | 输出 tok/s | 联合 SLO goodput tok/s | 联合 SLO 请求 |
|---:|---|---:|---:|---:|
| 1 | loop | 4.83 | 0.00 | 0/8 |
| 1 | segment split | 27.86 | 27.86 | 8/8 |
| 1 | segment fused | 33.23 | 33.23 | 8/8 |
| 2 | loop | 5.28 | 0.00 | 0/8 |
| 2 | segment split | 36.75 | 36.75 | 8/8 |
| 2 | segment fused | 37.96 | 37.96 | 8/8 |

两轮中 segmentation 都将该 workload 的 SLO goodput 从 0 提高到正值，输出吞吐约为 loop
的 5.8–7.2×。fused 相比 split 两轮分别高 19.3% 和 3.3%；TPOT p50 变化方向也不一致。
这是单一 trace 的早期观察，融合收益以如下扩展复测为准。产物：`results/moe_int4_ablation_final.json`、
`results/moe_int4_arrival_loop.json`、`results/moe_int4_arrival_segment_split.json`、
`results/moe_int4_arrival_segment_fused.json` 及 `results/moe_int4_arrival_r2_*.json`。

#### 扩展固定输入正确性与服务负载复测

**固定输入正确性**：第一 MoE 层使用 seed 7/17/42/91/123 和 1/8/32/128 个输入 token，
20 组检查中的 loop、segment split、segment fused 均逐位一致。随后覆盖真实模型全部 26 个
MoE 层，使用 seed 7/17/42 与 batch 1/8/32，共 234 个 layer/seed/batch 组合；split 与 fused
输出相对 loop 的最大绝对误差和相对误差均为 0。该结果验证这些固定输入下的专家层计算正确，
不代表在线调度下生成 token 必然逐次相同。

**4/8 req/s 配对 serving sweep**：DeepSeek-V2-Lite，TP=1、eager、streaming 纯 INT4 group64；
每个请求输入 64 token，输出长度 16/64，seed 7/17/42，每条件重复两次，共 24 个 split/fused
配对测量。所有运行都满足 10 s TTFT / 600 ms TPOT 联合 SLO，因此该矩阵的 token goodput
变化与总输出吞吐变化一致。

| 到达率 | 输出 token | 配对数 | fused 胜/负 | fused 吞吐 delta 中位数 | delta 范围 |
|---:|---:|---:|---:|---:|---:|
| 4 req/s | 16 | 6 | 1 / 5 | -5.1% | -10.0% 至 +14.4% |
| 4 req/s | 64 | 6 | 3 / 3 | +2.2% | -3.4% 至 +19.9% |
| 8 req/s | 16 | 6 | 1 / 5 | -4.0% | -18.5% 至 +0.6% |
| 8 req/s | 64 | 6 | 2 / 4 | -2.7% | -8.4% 至 +27.1% |
| **总体** | — | **24** | **7 / 17** | **-2.4%** | **-18.5% 至 +27.1%** |

该矩阵连续到达生成序列在 22/24 对中相同。两对不一致的 workload 中，同一后端重复运行也
出现 token fingerprint 变化；结合固定输入的逐位层级 parity，不能把这类在线序列差异单独
归因于融合内核。短输出负载尤其倾向 split，整体数据没有显示稳定正收益。

**16 req/s 压力点**：16 个请求、64 输入/64 输出 token、seed 7/17/42、每个 seed 重复两次，
TTFT 目标 10 s、TPOT 目标 150 ms，共 6 个配对测量。split 和 fused 都在每次运行中达到
16/16 联合 SLO；fused 吞吐中位数为 +5.4%，5 胜/1 负，范围 -20.3% 至 +31.9%。生成 token
在 5/6 对中一致；同后端的六组重复比较中有三组输出 fingerprint 变化。正向中位数伴随较大
波动，且所有请求都通过 SLO，因此尚未测出 goodput 拐点，也不足以证明融合有稳定服务收益。

**SLO 拐点复测（2026-10-11）**：24 请求/条件，constant arrival，输入 64 token，seed
7/17/42，每个 seed 重复 3 次，配对顺序交替；TTFT 10 s，greedy argmax。150/125 ms
TPOT 测 16/24/32 req/s 和 16/64/128 输出，100 ms 测 64/128 输出。表内为每个后端在 9 条
trace 中达成 joint SLO 的请求总数（split/fused，满分各 216）：

| TPOT | 输出 token | 16 req/s | 24 req/s | 32 req/s |
|---:|---:|---:|---:|---:|
| 150 ms | 16 | 216/216 | 216/197 | 211/216 |
| 150 ms | 64 | 216/216 | 216/216 | 216/216 |
| 150 ms | 128 | 216/216 | 216/216 | 216/216 |
| 125 ms | 16 | 125/168 | 166/181 | 145/190 |
| 125 ms | 64 | 216/195 | 213/216 | 192/216 |
| 125 ms | 128 | 216/216 | 216/216 | 216/216 |
| 100 ms | 64 | 133/136 | 93/92 | 88/89 |
| 100 ms | 128 | 111/115 | 63/143 | 72/141 |

100 ms 下，64-token joint-SLO 达标率从约 62% 降到 41–43%；128-token split 在 24/32 req/s
为 63/216 和 72/216，fused 为 143/216 和 141/216，呈现长输出、严格 TPOT、高到达率下的
条件性收益。但 100 ms 的部分条件仍有 0/24 与 24/24 的整轮波动；三档 216 个配对的吞吐 delta 中位数
分别为 +2.51%、+1.28%、+3.53%，范围均跨越正负，端到端 token parity 总计 177/216。固定
输入 MoE layer parity 仍是 234/234 逐位一致。详细 goodput 和运行分布见 `INTERVIEW.md`
§10.3.16；结果文件为 `results/moe_int4_fuse_load150_20261011.json`、
`results/moe_int4_fuse_load125_20261011.json`、`results/moe_int4_fuse_load100_long_20261011.json`。

结论：这轮确认了联合 SLO goodput 的下降区间：128-token 输出在 125 ms/32 req/s 尚为全达标，
100 ms 下已下降；64-token 输出在 125 ms/32 req/s 的 split 路径已下降。保留 split/fused
消融选项，segment 仍为显式后端；100 ms 长输出 fused 的优势值得追加重复，但当前数据不
支持普遍融合收益。此后完成了固定 decode batch、边界条件重复与在线分歧捕获；当前未解决项是
逐位置定位目标请求的前序 KV 差异。结果产物：`results/moe_int4_ablation_seed_matrix_20261009.json`、
`results/moe_int4_all_layer_correctness_20261009.json`、
`results/moe_int4_fuse_sweep_greedy_rep2_20261009.json`、
`results/moe_int4_fuse_stress_r16_out64_tpot150_20261009.json`。

### 100–125 ms TPOT 拐点细化复测（2026-10-11）

**设置**：DeepSeek-V2-Lite / RTX 5060 Ti 16 GB / WSL2，TP=1、eager、INT4 group64、segment 后端、KV swap 关闭；24 个请求/trace，固定 64-token prompt，greedy argmax，TTFT SLO 10 s。测 105/110/115/120 ms TPOT、24/32 req/s、64/128 输出；seed 7/17/42，各格每 seed 重复 5 次，共 240 个 split/fused 配对、480 次 measured trace。每对复用相同 prompt、arrival schedule 和 workload fingerprint，运行顺序交替；每次 2-seq warmup 后清空 prefix KV。CUDA kernel 预热覆盖两后端及 batch 1–24。GPU 时钟每秒采样一次。

表中联合 SLO 请求数是每端 15 次 trace 的总数（上限 360）；吞吐 delta 为每对 fused 相对 split 的输出吞吐百分比；goodput delta 是每对 joint-SLO output goodput 差，单位 tok/s。范围均为 15 对重复中的最小值至最大值。

| TPOT | req/s | 输出 | 联合 SLO 请求 split/fused（/360） | 吞吐 delta 中位数 [范围]；胜/负 | goodput delta 中位数 [范围] tok/s | token parity |
|---:|---:|---:|---:|---:|---:|---:|
| 105 ms | 24 | 64 | 233/244 | +0.68% [-20.69,+21.22]；8/7 | +0.0 [-215.9,+222.3] | 12/15 |
| 105 ms | 24 | 128 | 245/315 | +4.18% [-9.32,+13.11]；9/6 | +10.5 [-75.0,+222.7] | 12/15 |
| 105 ms | 32 | 64 | 281/278 | +1.95% [-28.65,+14.42]；8/7 | +3.4 [-245.1,+222.7] | 15/15 |
| 105 ms | 32 | 128 | 304/299 | +1.22% [-14.15,+16.54]；10/5 | +10.7 [-236.7,+245.3] | 14/15 |
| 110 ms | 24 | 64 | 292/273 | +0.02% [-19.94,+29.58]；8/7 | +0.0 [-225.2,+238.5] | 14/15 |
| 110 ms | 24 | 128 | 357/341 | +0.97% [-13.67,+10.21]；8/7 | +2.1 [-144.5,+47.7] | 13/15 |
| 110 ms | 32 | 64 | 263/325 | +6.37% [-14.51,+26.36]；11/4 | +19.2 [-213.5,+225.2] | 13/15 |
| 110 ms | 32 | 128 | 322/352 | +3.31% [-8.56,+18.16]；11/4 | +7.5 [-89.1,+232.7] | 12/15 |
| 115 ms | 24 | 64 | 318/351 | +2.70% [-15.72,+28.81]；8/7 | +6.0 [-80.3,+209.9] | 14/15 |
| 115 ms | 24 | 128 | 360/360 | -3.40% [-17.93,+4.73]；5/10 | -8.1 [-44.7,+10.8] | 15/15 |
| 115 ms | 32 | 64 | 336/302 | -4.04% [-18.92,+19.20]；6/9 | -9.2 [-220.4,+197.8] | 11/15 |
| 115 ms | 32 | 128 | 360/359 | +4.00% [-16.69,+12.77]；9/6 | +9.2 [-48.8,+27.0] | 13/15 |
| 120 ms | 24 | 64 | 312/357 | -4.44% [-18.39,+29.87]；7/8 | -9.3 [-65.3,+226.6] | 14/15 |
| 120 ms | 24 | 128 | 360/360 | +3.72% [-13.05,+13.54]；11/4 | +8.2 [-31.4,+28.2] | 13/15 |
| 120 ms | 32 | 64 | 336/352 | +0.92% [-23.45,+24.19]；8/7 | +2.1 [-118.2,+205.0] | 13/15 |
| 120 ms | 32 | 128 | 360/360 | -0.57% [-13.62,+15.13]；7/8 | -1.2 [-32.3,+30.5] | 14/15 |

**判断**：128-token 输出的 fused 吞吐 delta 中位数为 +2.24%（120 对中 70 胜、50 负，范围 -17.93% 至 +18.16%）；64-token 为 +0.81%（64 胜、56 负，范围 -28.65% 至 +29.87%）。对应的 joint-SLO output goodput delta 中位数为 +5.04 与 +1.48 tok/s，但两者的重复范围都跨越较大的正负值。128-token 结果按目标从 105 到 120 ms 分别有正、正、负、正的中位吞吐 delta；24 与 32 req/s 也不总是同向。因此这轮没有确认稳定的长输出融合收益。累计联合 SLO 请求显示 128-token 在 115/120 ms 几乎或完全全达标（按两个到达率合计，split/fused 分别 720/719、720/720，共 720），但这描述的是输出长度和负载下的达标率，不构成 fused 一定提升 goodput 的证据。

SM 时钟每条 trace 的中位数，两后端均为 2775 MHz，范围 2760–2775 MHz；memory clock 固定 13801 MHz，运行中温度中位数 58 °C。decode batch p50 在全部 480 次 trace 中均为 24 行，单次模型调用观察范围为 1–24 行；纯 decode 调用中，24 行 batch 占 split 21,411/22,800、fused 21,414/22,800。完整 decode/prefill shape histogram 在 240 个配对中只有 53 对完全一致，说明相同 arrival trace 并未固定出相同的 model-call shape 分布。

Greedy 输出 token 全量一致的配对为 212/240。28 个不一致配对都来自 seed=42，且每对都只有 request index 23 一个请求不同；首个差异均在 0-based 输出 token index 48（第 49 个 token），生成长度仍符合 64/128 上限。五轮同后端复跑中，split 有 8/48 个条件组、fused 有 11/48 个条件组出现两个不同的输出 fingerprint。当前 JSON 记录首个差异位置和长度，没有保存分歧处 token ID 或 logits；因此还不能断定是融合数值差、动态 batch 形状，还是该请求的近似 argmax 边界导致。

保留 fused 为实验选项，不据此宣称稳定性能提升。后续固定 batch、在线层间追踪和 MLA 行置换隔离结果见下文；在线 fused goodput 收益仍按负载条件单独报告。原始结果：`results/moe_int4_fuse_load105_120_rep5_20261011.json`。

### 固定 batch 形状分歧诊断（2026-10-12）

**目的**：复核 2026-10-11 连续到达实验中 seed=42 / request 23 / 输出 token 48 的 split/fused
首次分歧。通过固定同一批 prompt、采样参数和输出 token trace，先把算子数值与调度形状影响分开。

**配置**：DeepSeek-V2-Lite，RTX 5060 Ti / SM120，PyTorch 2.8.0+cu128、CUDA 12.8、Triton 3.4；
TP=1、INT4 group64、segment backend、eager，关闭 CUDA Graph、多步 decode 与 KV swap。每组 24 个
请求同时到达，每个 prompt 固定 64 tokens。使用 `benchmarks/bench.py build_workload(seed)` 生成
seed 7/17/42 的 prompt 集；随机采样温度为 0.6/0.8/1.0，`ignore_eos=True`，输出长度为 64/128。
每个 variant 先做两次完整 24 路、128-token warmup；性能阶段每个 seed × temperature × 输出长度
做 3 次配对，共 54 对，split-first / fused-first 各 27 对。SLO 设为 60 s TTFT / 10 s TPOT，
仅作为报告指标，不构成有效的 goodput 门槛。

**正确性**：18 个随机采样条件（3 seeds × 3 temperatures × 2 lengths）中，432/432 个请求的
token 输出一致；另有 6 个 greedy 条件，144/144 个请求一致。性能阶段固定重放 split 生成的 token
trace，54/54 对的模型输入 fingerprint 与 model-call shape histogram 相同。每次 prefill 均为 24
条 / 1536 tokens；64-token 输出有 63 个 24-row decode call，128-token 输出有 127 个。

| 输出长度 | 温度 | 配对数 | fused 胜/负 | 输出吞吐中位 delta | delta 范围 |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.6 | 9 | 7/2 | +5.13% | -10.05%～+21.60% |
| 64 | 0.8 | 9 | 7/2 | +2.06% | -4.05%～+11.85% |
| 64 | 1.0 | 9 | 6/3 | +4.22% | -9.27%～+19.77% |
| 128 | 0.6 | 9 | 6/3 | +3.57% | -8.31%～+13.60% |
| 128 | 0.8 | 9 | 5/4 | +1.72% | -9.61%～+10.76% |
| 128 | 1.0 | 9 | 7/2 | +6.74% | -4.70%～+14.20% |

合并 54 对后，fused 输出吞吐中位 delta 为 **+3.52%**（38 胜、16 负，范围 **-10.05%～+21.60%**）；
64-token 与 128-token 的分组中位数分别为 **+3.49%**（20/7，范围 -10.05%～+21.60%）和
**+3.56%**（18/9，范围 -9.61%～+14.20%）。TPOT p50 中位 delta 为 -3.69%，但范围为
-19.73%～+12.41%；SM clock 两端中位数均为 2760 MHz，配对中位差为 0 MHz。正向中位数值得继续
验证，但区间跨零，不能据此断言稳定收益。由于所有请求同一时刻到达且 SLO 目标很宽，本实验不代表
连续到达 serving 的联合 SLO goodput。
按 workload seed 汇总，seed 7/17/42 的吞吐中位 delta 分别为 +3.07%（14 胜/4 负）、+3.53%（12/6）、
+2.64%（12/6）；三个 seed 的配对范围仍都跨零。

**历史 sentinel**：seed=42、request 23、0-based 输出位置 48，以 greedy 和共同前缀 teacher-forced
重放完整 logits；102400 个 logit bitwise 相同。26 个 MoE 层的输入/输出、路由专家与权重以及
gate/up/down 投影均 bitwise 相同。上一轮 token 48 分歧在本次固定 batch 测试中未复现；本次 greedy
teacher-forced probe 也没有重建旧的连续到达调度历史，因此根因仍待查。GPU clock 以 250 ms 采样，共 4653
个样本；详见 [原始结果](../results/moe_int4_fixed_shape_diag_20261012.json)。

复现实验入口：`python benchmarks/moe_int4_fixed_shape_diag.py --model ~/huggingface/DeepSeek-V2-Lite`；
完整条件和默认值见脚本参数。后续连续到达 trace 与 tight-SLO 复测结果见下一节。
固定 batch 的脚本不会替代服务负载基准。

### 连续到达分歧追踪与 tight-SLO 扩展（2026-10-12）

**分歧追踪设置**：复放历史高复现条件：seed 42、constant 32 req/s、24 个请求、64-token 输入 /
64-token 输出、10 s TTFT / 115 ms TPOT。workload fingerprint 为
`9f9947466d6d1844a4ac1b6ebedf565eec33d6d170e37c3c0ccc496ec9a84782`，与 10/11 原始记录一致。
INT4 group64、TP=1、eager、segment、greedy argmax；每种后端预热 batch 1–24，每条 trace 做相同的
2 请求 warmup 并清空 idle prefix KV。5 个配对中，request 23 的输出 token 48 在 1 对中再次分歧。
诊断脚本记录所有模型调用的 request 行顺序、输出 token/模型位置、shape 与目标 token 的 CPU/CUDA
RNG state hash，并在目标处抓完整 logits、top-k 和 27 个 decoder block / 26 个 MoE 层的值。
这些 hook 会扰动运行时，只用于正确性诊断，不能用其墙钟时间评价性能。

5 对中有 4 对的目标 logits 与所有 decoder / MoE 中间值逐位相同；失败配对仍是相同 token ID 和
position（输入 token 17、position 111），两端目标调用都是 24 行纯 decode，但 request 行顺序不同。
CPU/CUDA RNG state 前后 hash 均未变化，排除随机采样消耗顺序。失败配对的 top-1 从 split 的 15
变为 fused 的 7002，margin 分别为 9.5625 / 0.6875；完整 logits max abs 差 14.78125。
27 个 decoder block 中第一个差异出现在 decoder layer 1 的 `self_attn` 输出：2048 个元素中 958
个不同，max abs `9.77e-4`、relative L2 `2.72e-3`；随后 MoE layer 0 输入、路由和投影才不同。
这将首个可观测差异定位在专家 GEMM 融合之前的 attention 路径。当时仍不能区分 batch 行置换、
前序 KV 数值或 MLA block/context 映射；下一节的固定逻辑 KV 重放已经排除了目标调用的行置换影响。

**100 ms TPOT 配对扩展**：将 10/11 的三次重复与本次追加的两次重复合并，每个 workload seed 在
24/32 req/s × 64/128 输出的每格共 5 对（15 对 / 格，360 请求 / 后端）。旧、新文件的每个
seed/负载 fingerprint 均一致。GPU SM clock 中位数 2775 MHz（2760–2775），decode batch
p50/max 均为 24 行；shape histogram 在每格 10–11/15 对完全一致。

| 到达率 | 输出 | joint-SLO 请求 split/fused（/360） | 吞吐 delta 中位数 [范围]；胜/负 | 输出 goodput delta 中位数 [范围] tok/s | token parity |
|---:|---:|---:|---:|---:|---:|
| 24 req/s | 64 | 174 / 147 | -1.87% [-16.74,+28.38]；6/9 | +0.00 [-223.5,+230.0] | 15/15 |
| 24 req/s | 128 | 134 / 209 | +4.19% [-11.96,+15.85]；10/5 | +129.81 [-242.8,+239.9] | 14/15 |
| 32 req/s | 64 | 157 / 146 | +5.50% [-16.46,+14.65]；8/7 | +9.03 [-221.6,+223.4] | 11/15 |
| 32 req/s | 128 | 172 / 230 | +7.92% [-10.16,+21.54]；10/5 | +26.63 [-239.5,+250.5] | 14/15 |

长输出下 fused 的累计 joint-SLO 请求数更高，但 seed 方向并不一致，配对 goodput 范围也跨越很大的
正负值；64-token 结果的累计达标数则 fused 较低。当前数据不支持对一般 serving 负载宣称融合
收益。保留 fused 为实验选项。相同逻辑 KV 下的 MLA 行置换检查已在下一小节完成；该结果没有显示
当前 batch 行序会改变 attention 输出，后续应先定位前序 KV 首次不同的位置，再决定是否需要改动 MoE 内核。

原始文件：[在线诊断](../results/moe_int4_arrival_trace_diag_layers_rep5_20261012.json)、
[新增 2 次 tight-SLO 配对](../results/moe_int4_fuse_load100_extend_rep2_20261012.json)、
[合并前的 3 次重复](../results/moe_int4_fuse_load100_long_20261011.json)。

### 固定逻辑 KV 的 MLA decode 行置换诊断（2026-10-12）

**配置**：复用同一 seed-42 / constant 32 req/s / 24 请求 / 64-token 输入与输出 / 115 ms TPOT
trace，独立追加五对 split/fused 行序诊断运行。每次目标调用均为 24 行纯 decode；五对的 request 行顺序都不同。
在 layer 1 attention hook 中保存每个 decode request 的输入 hidden、position 和当前调用写入后的完整逻辑
MLA KV；配对完成后，把每侧缓存内容装入新的紧凑 block 表，分别按该侧原行序和对端实际行序重放。
slot_mapping 置空，重放不写 KV。此诊断只检查数值正确性，hook / cache capture / replay 的耗时不能
用于性能比较。

**结果**：五对中有四对 token 48 的 logits 与输出一致；第五对再次出现 request 23 / output token 48
分歧，输入仍是 token 17、position 111。目标 layer-1 hidden 在两端 2,048/2,048 元素 bitwise
相同；逻辑 KV 有 19,575/64,512 元素不同，max abs `0.125`、relative L2 `1.88e-3`；该行 attention
输出 958/2,048 元素不同，max abs `9.77e-4`、relative L2 `2.72e-3`。这次配对 split top-1 为
7002（top1-top2 margin `0.6875`），fused top-1 为 15（margin `9.5625`）。CPU/CUDA RNG state hash 均未变化。


| 固定状态回放检查 | 精确一致 |
|---|---:|
| 原始行序重放与 live attention 输出 | 10/10 后端状态；每次 24/24 行 |
| 固定每侧 KV，换成对端观察到的行序 | 10/10 后端状态；每次 24/24 行 |
| 两端相同 position 的请求行：layer-1 hidden | 96/96 行 |
| 两端相同 position 的请求行：逻辑 KV / attention 输出 | 各 95/96 行 |

因此，在发生分歧的状态下，仅重排同一组 query、position 和每请求 KV，不会改变 MLA attention 输出；
目标 decode 调用的 batch 行置换不是这次 token mismatch 的原因。当前 hidden 相同而缓存不同，说明差异
已存在于进入本次 layer-1 attention 时的 KV 状态。重放原行序可以精确复现 live 输出，验证了本实验对
该层逻辑 cache 与请求行映射的重建；但尚未定位最早不同的 KV token，也不能区分差异来自更早的模型数值、
prefill 写入还是 cache 生命周期。下一步应跟踪 request 23 从 prompt prefill 到 position 111 的 layer-1
hidden/KV 写入，并在第一个不同位置核对 slot mapping 与 block table。现有结果不支持据此修改 MoE fused GEMM；
fusion 仍是实验选项。

复现实验命令：

```bash
wsl -e bash /mnt/d/Project/nano-vllm/benchmarks/run_in_wsl.sh \
  benchmarks/moe_int4_arrival_diag.py \
  --model /home/zjq/huggingface/DeepSeek-V2-Lite \
  --seeds 42 --arrival-rates 32 --output-lengths 64 --tpot-targets 115 \
  --num-seqs 24 --input-length 64 --warmup-seqs 2 --repeats 5 \
  --focus-request-index 23 --focus-token-index 48 \
  --output results/moe_int4_mla_row_permutation_rep5_20261012.json
```

原始结果：[固定 MLA KV / 行序回放 JSON](../results/moe_int4_mla_row_permutation_rep5_20261012.json)。

## 5. 引擎吞吐（toy 12-seq × 16-token decode，多次运行取区间）

| 模式 | 多运行区间（tok/s） | vs 同运行 fp16 |
|---|---|---|
| fp16 | 185-475（中位 ~350） | 1× |
| fp8 | 480-903 | ~1.9-2.6× |
| int4 | 927-1110 | ~2-6×（fp16 档被拖慢时比率虚高） |

解读：方向性结论稳健——decode 专家 GEMM 是**权重带宽受限**形态（专家权重全量读取、
每 token 激活行少）→ 量化字节减半直接 ≈2×（接阶段 1 的形态论）。**绝对值噪声大**
（WSL 时钟波动 + CPU 组织税混合，fp16 档跨运行差 2.6×）——精确数字需真实模型 +
更长基准复测；此处只做方向断言。
**更新（阶段 1.5b，grouped 后端后）**：int4 dual-path 自动走 grouped（w_deq），
toy decode 单次 2588 tok/s（此前 loop 档 927-1110）；fp16/fp8 仍在时钟噪声带——见
§4b 层级基准（可靠证据）。

## 6. 真实模型边界（阶段 1.5 的历史记录；后续状态见阶段 2b-ext）

下表里的体积均为当时基于 config 的下载前估算，不能代替后续实测。DeepSeek-V2-Lite 后续已下载并跑过真实权重，详见阶段 2b-ext 报告。

| 模型 | 结构线索（config） | int4 权重估算 | 16GB 卡 | 下载（bf16） |
|---|---|---|---|---|
| Qwen3-30B-A3B | 128 experts/top-8?（以真实 config 为准） | ~15.5GB | **✗ 超可用显存**（~14GB） | ~60GB，需核对 |
| Qwen1.5-MoE-A2.7B | 老架构（per-expert 文件） | ~7.5GB | ✓ 可跑 | ~28GB+，需核对权重格式 |
| DeepSeek-V2-Lite | MLA+MoE（当时的阶段 2 目标） | ~8GB（旧估算） | 旧估算为可跑 | 后续实际验证：4 层 parity；全量 27 层流式 int4 常驻权重约 5GB |

所有"需核对"项 = 下载前用 config.json/safetensors 清单确认（本机 hub 连接不稳定，
联网后跑 `benchmarks/_moe_model_probe.py` 一次拿准）。

## 7. 未做（诚实边界）+ 下一步

- **浮点 grouped 的 padding**：padded-bmm 仍可能按 E·max_n 放大计算；浮点权重
  `segment_backend` 已在 4c 实现无 padding 内核，实测收益窗口限于小 E + 极长段。
- **量化专家的 grouped**：纯 packed INT4 已在 4d 实现 offsets/counts 分段 GEMM 与
  gate/up launch 融合；限制是 TP=1、BF16/FP16、非 AWQ、无 `w_deq`。FP8、W8A8、NVFP4
  保留原有 backend 选择；TP>1 不使用此分段内核，CUDA Graph/TP>1 组合尚未验证。
- **CUDA graph**：路由 padding（decode 图内固定路由模式）未做——MoE 模型现在
  enforce_eager。
- **TP/EP**：TP 分片语义可用（Column/Row 继承）但未实测；EP（专家并行 + all-to-all）
  是 MoE 分布式正题，未做（单卡）。
- **shared expert**：transformers 5.15 的 qwen3_moe 已无 shared；如需（Qwen3-235B 类）
  在 MoE 上加回（设计已留空）。
- 真实模型：DeepSeek-V2-Lite 已覆盖 26 个真实 MoE 层的 loop/segment/fused parity、阶段计时、
  多 seed / 到达率 / 输出长度的纯 INT4 serving sweep；这不代表其他架构、量化格式、TP/EP
  配置或 GPU 的结果，且当前数据没有证明 fused 有稳定的服务级收益。

## 8. 面试要点

- 能讲完整验证链：数学同构参考位级对照 → CPU 单测 → 端到端 parity（top-1 100%）→
  量化机制核验；能讲"checkpoint 3D vs 2D"的存储格式事实与 loader 子串碰撞修复。
- 能讲两个反直觉实测：①循环专家对负载不均**免疫且偏好集中**（0.20-0.28×）；②toy
  量化 top-1 全翻是**顶层贴边**（0.4-0.7σ gap）而非机制错——"判据先于结论"。
- 能讲量化收益形态：decode 专家 GEMM 带宽受限 → int4/fp8 ≈2×（与 roofline 形态论一致）。
- **能讲 grouped 后端演进**：组织税实测 3.8ms/层 → sort+padded bmm（fused gate_up）
  后 0.7ms（-80%），小 T 快 4-5×；padded 批量在均衡路由下近 1× 计算、不均衡时放大
  到 E·max_n（浮点 segment_backend 的边界见 4c）；int4 dual-path 因 w_deq 自动获益
  （引擎 ~2.4-2.8× 观察），纯 INT4 segment 在真实 V2-Lite 层级和 serving trace 上的测量见 4d。
- 诚实边界：本报告的随机 Qwen3-MoE toy 没有真实模型精度数字；DeepSeek-V2-Lite 的
  INT4 MoE 结果仅覆盖指定层形状与两轮短 trace，绝对吞吐受 WSL 时钟与运行噪声影响，
  不外推到其他模型或硬件。
