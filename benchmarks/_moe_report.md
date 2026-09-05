# MoE 支持报告（阶段 1.5：router + 循环专家 + 量化 + 引擎验证）

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
  纯 int4/fp8/w8a8/sparse24（无 float 视图）→ 回退逐专家循环（各自量化内核）。
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

## 6. 真实模型边界（诚实标注：本机无 MoE 真权重；hub 网络本轮不可达 → 离线估算待核对）

| 模型 | 结构线索（config） | int4 权重估算 | 16GB 卡 | 下载（bf16） |
|---|---|---|---|---|
| Qwen3-30B-A3B | 128 experts/top-8?（以真实 config 为准） | ~15.5GB | **✗ 超可用显存**（~14GB） | ~60GB，需核对 |
| Qwen1.5-MoE-A2.7B | 老架构（per-expert 文件） | ~7.5GB | ✓ 可跑 | ~28GB+，需核对权重格式 |
| DeepSeek-V2-Lite | MLA+MoE（阶段 2 目标） | ~8GB | ✓ 可跑 | ~30GB，需核对格式 |

所有"需核对"项 = 下载前用 config.json/safetensors 清单确认（本机 hub 连接不稳定，
联网后跑 `benchmarks/_moe_model_probe.py` 一次拿准）。

## 7. 未做（诚实边界）+ 下一步

- **真段式 grouped（无 padding）**：当前 padded-bmm 在不均衡/大 E 时计算放大到
  E·max_n（本尺度实测仅 +30%，效率摊销）；Triton/CUDA 段式 grouped kernel（offsets
  驱动、逐专家实长段）可彻底消除 padding——大 E（128+）真实模型的下一步。
- **量化专家的 grouped**：纯 int4/fp8 无 float 视图 → 回退循环；grouped 版需
  打包权重按 3D 布局 + 专用内核（Triton 反量化 grouped），未做。
- **CUDA graph**：路由 padding（decode 图内固定路由模式）未做——MoE 模型现在
  enforce_eager。
- **TP/EP**：TP 分片语义可用（Column/Row 继承）但未实测；EP（专家并行 + all-to-all）
  是 MoE 分布式正题，未做（单卡）。
- **shared expert**：transformers 5.15 的 qwen3_moe 已无 shared；如需（Qwen3-235B 类）
  在 MoE 上加回（设计已留空）。
- 真实模型：下载/显存边界见 §6；DeepSeek-V2-Lite 的 MLA+MoE 端到端是阶段 2。

## 8. 面试要点

- 能讲完整验证链：数学同构参考位级对照 → CPU 单测 → 端到端 parity（top-1 100%）→
  量化机制核验；能讲"checkpoint 3D vs 2D"的存储格式事实与 loader 子串碰撞修复。
- 能讲两个反直觉实测：①循环专家对负载不均**免疫且偏好集中**（0.20-0.28×）；②toy
  量化 top-1 全翻是**顶层贴边**（0.4-0.7σ gap）而非机制错——"判据先于结论"。
- 能讲量化收益形态：decode 专家 GEMM 带宽受限 → int4/fp8 ≈2×（与 roofline 形态论一致）。
- **能讲 grouped 后端演进**：组织税实测 3.8ms/层 → sort+padded bmm（fused gate_up）
  后 0.7ms（-80%），小 T 快 4-5×；padded 批量在均衡路由下近 1× 计算、不均衡时放大
  到 E·max_n（真段式是下一步）；int4 dual-path 因 w_deq 自动获益（引擎 ~2.4-2.8× 观察）。
- 诚实边界：无真实模型精度数字；引擎绝对吞吐受 WSL 时钟噪声（以层级/同刻对比为准）。
