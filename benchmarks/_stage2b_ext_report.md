# 阶段 2b 扩展报告：fp8 KV + 环、投机+环、非统一窗口环、量化 MLA decode、真实模型验证

日期/硬件：RTX 5060 Ti 16GB（sm_120）/ WSL2（RAM 11GB+4GB swap，前 7GB 反复 OOM 后调大）

> **阶段快照说明（2026-10-05）**：本报告记录当时 MLA/rolling-cache 的组合验证，不含后续 HTTP 服务、会话管理或在线调度策略的性能数据。本文的 Gemma-2 split 是单个 ModelRunner/GPU 内的 local/global 两个 KV 池，**不是** Prefill/Decode 双 GPU 分离。2026-10-08 代码状态更新：Gemma-2 soft-cap + FP8 KV、rolling/split KV swap 与 decode/mixed Graph 已接通，但本报告没有覆盖这些组合，尚待 RTX 5060 Ti/WSL2 实测。当前系统状态和 PD 限制见仓库根目录 `README.md` / `INTERVIEW.md`。

## 1. 新增能力（相对 2b 的断言关）

| 能力 | 落点 | 说明 |
|---|---|---|
| fp8 KV + 环 | `attention.py` fp8 decode 传 `chunk_starts` | fp8 内核按行首块号 j0 还原位置（store 带 scale 不变） |
| fp8 KV + MLA | `attention_mla.py` fused 行两段独立 scale | e4m3 存储 1B/元素；写时 [c\|k̃] 各自量化、读时（内核/稠密装配/吸收式）各自反量化；`_mla_check` 的 gather+dequant 内核 |
| 投机（ngram）+ 环 | BlockManager 环 spec 账本修正 + `_ring_rows` 装配 | verify 行 key 集 = 环内现存行 [j0·B, end) 稠密装配喂 flash（段内相对下标使窗口掩码精确）；ring slack = γ+2 |
| 非统一窗口（gemma2 split） | 双池：环池（local 层）+ full 池（global 层） | 单卡上两个独立 KV cache / BlockManager（full=no_share 普通分页）、Context full_* 侧、内核 softcap（cap·tanh） |
| 纯 int4/fp8 MLA decode | kv_b 反量化副本保留（`is_mla_kv_b` → w_deq） | 流式纯 int4/fp8 不再落稠密兜底/不再强制 eager（占参 ~1%，V2-Lite 实测 113MB） |
| int4 组大小可配 + 尾 K 掩码 | `int4_group_size` Config + 内核 GROUP 展开 | DeepSeek-V2-Lite dense 中间维 10944=64×171 不能被 128 整除（官方社区 int4 同约束） |

## 2. 验证链（跑通≠写对）

### 2.1 环 fp8（toy mistral W=512，引擎对跑）
- ring(fp8) vs masked(fp8)：**1099 decode 步逐位相等**（同内核、同行值序列，环只换页地址）；
- 采样步 vs bf16 手工窗口参考：42/42 top-1（fp8 量化噪声带内）。

### 2.2 MLA fp8 KV（toy DeepSeek，引擎对跑 + 内核单测）
- fused 行 gather+dequant 内核 vs 手工同序：**逐位一致**；
- prefill 末行 vs 稠密参考 top-1 100%；decode fp8-vs-bf16：44 可比行 1 翻转（fp8 噪声带），mean 0.014。

### 2.3 投机 + 环（toy mistral W=512，ngram γ=4）
- ring(ngram) vs masked(ngram)：bf16 538 可比行、fp8 230 可比行，**翻转均 0**、轨迹同构；
- 诚实边界：随机权重下 ngram 草稿接受率 α=0（草稿正确需训练权重），多 token 接受路径未实际触发；
  验收控制流（拒绝/重写/stale 槽位）经全程同轨迹验证。

### 2.4 非统一窗口（gemma2 split）
- toy（6 层交替 W=512，softcap=50）：split vs masked 368 可比行 top-1 99.18%（3 翻转 = 内核 vs flash softcap/tanh 数值差），mean 0.005；vs 手工稠密参考（窗口+softcap）40/40 步 top-1 100%（含 W、2W 边界）；
- **真实 gemma-2-2b-it**（26 层 13+13，W=4096）：4800-token 解码（ignore_eos 跨窗），运行中环表采样：local 环池到 4500 步封顶 **18 块 = ring_cap** 而 full 池/masked 表继续线性增长（4750 步 19）；30 个稀疏 decode 步（每 300 步，含窗口越界后）vs 手工 fp16 稠密参考 **top-1 失配 0**，mean logits 差 0.031（与 flash-vs-手工同量级）；
- 环 vs 掩码逐行对照在真实模型上因内核数值差导致的近并列采样翻转（首分歧 ~36-409 步）不能长对齐——真实一致性改由"稀疏步手工参考 + 分叉前全一致"论证。

### 2.5 真实 Mistral-7B-v0.1 长解码（int4 流式权重，decode 5050 token > W=4096）
- fp8 KV：ring vs masked **全程 5050 步轨迹一致、可比行 top-1 100%、logits 差 0.0**（逐位）；
- bf16 KV：对齐段 103 行 top-1 100%（早期采样分叉退出比对；分叉来自 ring 内核 vs flash 数值差 + 温度 0.8 采样）；
- 环表稳态 ≤ cap=18（末段不再增长）vs 掩码线性 20+（5050 时两者仅差 2 块——7B 内存账本收益需 len ≫ W 才显著，公式/表长证据见上）。

### 2.6 真实 DeepSeek-V2-Lite（31.4GB bf16，hub 镜像下载，流式 int4 group64）
- **4 层真权重切片（含 embed/head）**：引擎 fp16 decode vs fp16 稠密手工参考 24 行失配 **0**；
  int4(group64) 引擎 vs 同量化稠密参考 23/24（1 翻转，噪声带内）——真实权重的 MLA/MoE/路由分布数值锚；
- **全量 27 层流式 int4**：启动 55s（逐层加载即量化）；权重常驻：int4 打包 + kv_b w_deq 113MB + 其余浮点 0.84GB；
  decode ~2.8 tok/s（3 条并发生成、MoE eager 循环、纯 int4 无 w_deq 双路径、无图——诚实口径）；中/英 3 prompt 生成文本连贯可用；
- 顺带修复：MLA decode 内核在真实尺寸（kv_lora=512,H=16）共享内存超限（129KB>101KB）→ BLOCK_T 32→16、warps 4→8；
  int4 内核 K 尾块（10944 非 128 倍数漏算 64 列 → 引擎 logits 灾难性漂移 +NaN）→ 尾 K 掩码 + 组粒度静态展开（128 组路径逐位不变，64 组路径 vs 反量化逐位一致）；
  transformers 5.15 的 MoE `torch._grouped_mm` 仅 sm_90 → 本机无法用 HF-GPU 直连做真实权重对照（记录，不用引擎侧代码绕开）。

## 3. 组合矩阵（本报告对应的 2026-10-05 checkout）

| 组合 | 状态 |
|---|---|
| ring + bf16/fp8 KV + decode + graph | ✅（fp8 真机逐位证据） |
| ring + ngram spec（bf16/fp8 KV） | ✅（eager verify，α=0 路径验证） |
| ring + medusa/eagle spec | ❌ 断言关 |
| gemma2 交替窗口 ring（split, auto KV；本实测为 bf16；no-spec, eager） | ✅ |
| split + fp8 KV / spec / graph / swap | ❌ 当时断言关或未实现（FP8 与 softcap 冲突等） |
| MLA fp8 KV（decode 内核/稠密装配/吸收式） | ✅ toy；真实模型未跑（fp8 KV 与真实 V2-Lite 时间成本未覆盖） |
| MLA 纯 int4/fp8 权重 decode | ✅（kv_b 反量化副本；稠密兜底仅剩 w8a8/sparse24 情形） |
| 滚动模型前缀缓存 | ❌ 停用（重复 prompt 代价，refcount 守卫） |
| KV swap + ring / split | 当时未验证（测试基准均显式关闭） |

## 4. 复现

```bash
python benchmarks/_ring_fp8_e2e.py          # 2.1
python benchmarks/_mla_fp8_check.py         # 2.2
python benchmarks/_ring_spec_e2e.py         # 2.3
python benchmarks/_gemma2_ring_e2e.py       # 2.4 toy
python benchmarks/_gemma2_real_ring.py      # 2.4 真实 2B（缓存 /tmp/_g2ring_{r,m}.pt）
python benchmarks/_ring_real_mistral.py     # 2.5 真实 7B（每引擎 ~15-20 分钟）
python benchmarks/_v2lite_build_4l.py       # 2.6 切片（需 ~/huggingface/DeepSeek-V2-Lite 全量）
python benchmarks/_v2lite_4l_parity.py      # 2.6 4L parity
python benchmarks/_v2lite_full_probe.py     # 2.6 全量流式 int4
python benchmarks/_int4_group_check.py      # int4 group64/尾块内核单测
```
