# 阶段 2b：SWA 滚动缓冲（滚动缓存）报告

> **阶段快照说明（2026-10-05）**：这是 rolling-cache 初版记录。后续 [`_stage2b_ext_report.md`](_stage2b_ext_report.md) 已扩展 Mistral rolling 到 FP8 KV 和 n-gram verify，并加入 Gemma-2 split 双池。当前仍不支持 rolling+Medusa/EAGLE；Mistral rolling decode 可用 CUDA Graph，rolling verify 使用 eager；Gemma-2 split decode 仍 eager，且要求 `kv_cache_dtype="auto"`、关闭 speculative 与 KV swap。

## 1. 动机与机制

flash-attn / vLLM 传统 SWA 只做"窗口掩码"：KV 缓存仍按全上下文线性增长，窗口
只负责把窗口外的分数 mask 掉（旧块永远占着显存）。滚动缓冲把 **缓存本身** 变成环：
decode 期每序列只保留窗口内容，旧块越过窗口立即释放、新块复用其槽位 → 长生成
序列的 KV 内存有界（≈ 窗口/块大小 + 2 块），不再随生成长度增长。

## 2. 实现（对照路线图：per-seq 物理环 + refcount 守卫 + 内核窗口位置偏移）

| 件 | 文件/机制 |
|---|---|
| per-seq 逻辑环 | `Sequence.kv_j0`：块表 = 窗口内容清单（第 i 项 = 第 kv_j0+i 个逻辑块）；`BlockManager` 驱逐：`(front+1)·B ≤ N−W−slack` 时表头整块释放（先释放再分配，净零 free 消耗）|
| refcount 守卫 | 滚动块的 `_evict_front` 显式断言 ref_count==1 && hash==-1；环模型**不发布/不消费前缀缓存**（窗口内容过期 + decode 哈希链起点会被驱逐）→ 块恒私有，守卫恒真——若将来放开共享，断言就是守卫落点 |
| 回读余量 | 驱逐阈值留 `slack = max_draft_len + 2`：本初版未启用 verify；后续 n-gram verify+ring 已实现并使用 γ+2 余量（见 2b-ext）|
| 内核位置偏移 | flash-attn 从表下标推 key 位置 → 环表会错位；自研 bf16 paged decode 内核（fp8 内核源码泛化：`chunk_starts` 每行首块序号 j0；key_pos = (j0+b)·B+t；读块数 = ceil(seqlen/B) − j0）。bf16 走同一内核 scale=1（bf16→fp16 无精度损失）。CUDA-graph decode 同步支持（chunk_starts 静态缓冲）|
| 初版前置校验 | mistral（全层统一窗口）/ bf16 KV / 无投机；FP8 KV 与投机当时被断言拒绝，已由后续 2b-ext 扩展 |

## 3. 验证链

| 层级 | 方法 | 结果 |
|---|---|---|
| ① BlockManager CPU 属性 | 3000-token 解码模拟：驻留覆盖 [max(0,N−W−slack),N)；表长 ≤ ring_cap；refcount 守卫；极小池（8 块）长解码不失败；双序列共享小池 | pytest 4 项全过 |
| ② 引擎 e2e | mistral toy（W=512, B=256, 2 序列 decode 1100 token，跨 W / W+B / 2W 边界）rolling_cache=True vs 稠密掩码参考（独立手工注意力）| **84 采样步 top-1 全一致**、mean diff 0.001–0.0016 |

## 4. 内存账本与代价（诚实）

- 稳态：decode 期每序列块数 ≤ ring_cap = (W+slack−1)//B + 2（B=256、W=4096 → 18 块 ≈ W+B，比 vLLM 掩码式 SWA 的全上下文线性缓存省掉"生成越长越占"的部分）；
- prompt 本身超过窗口时，首次越过窗口前仍线性占块（同掩码式）；解码开始后逐步驱逐收敛到窗口（比 vLLM 掩码式"prompt 全量常驻"更省）；
- 代价：滚动模型前缀缓存停用（重复 prompt 的 TTFT 收益丢失）；环需要自研内核（flash 不能表达）；未对齐窗口边界最多多留 B 个 token/块（+2 块公式的来源）；
- 未做：真实 Mistral-7B 长解码验证（本机有 checkpoint，解码 4K+ token 的对照时间成本未跑）；fp8 KV + 环；投机 + 环；非统一窗口模型（Gemma-2 交替 local/global 因块表跨层共享无法逐层环）。
