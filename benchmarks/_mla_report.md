# 阶段 2a：MLA（Multi-head Latent Attention）报告

## 1. 交付物

| 文件 | 内容 |
|---|---|
| `nanovllm/layers/attention_mla.py` | MLAAttention 层 + fused cache `[c_kv \| k̃_pe]` 存储内核 + **吸收式 decode Triton 内核** + cache-shaped 行稠密化组装 + 稠密参考路径 |
| `nanovllm/models/deepseek_v2.py` | DeepSeek-V2 全模型端口（MLA + dense/MoE 混合层 + shared experts）|
| `nanovllm/layers/rotary_embedding.py` | interleaved-pairs RoPE（DeepSeek 约定，与两半式不等价）|
| `nanovllm/layers/layernorm.py` | **修复**：RMSNorm fp32 原位修改输入（残差流正确性 bug）|
| `nanovllm/layers/moe.py` | routed_scaling_factor（DeepSeek 路由概率缩放）|
| engine/runner/context | KV cache 布局泛化（MHA 双张量 vs MLA fused）、COW/swap 泛化、MLA 行信息、eager 强制 |
| `tests/test_mla.py`、`benchmarks/_mla_check.py`、`benchmarks/_ds_e2e.py` | 验证链 |

## 2. 语义锚点（对齐 transformers 5.15 modeling_deepseek_v2）

- q/kv 潜在层带 **RMSNorm**（q_a_layernorm、kv_a_layernorm）；k_pe（rope 部分）不 norm；
- kv_b 逐头输出 [k_nope(128) | v(128)]，行内切分；scaling = qk_head_dim^-0.5 = 192^-0.5；
- RoPE 是 **interleaved-pairs 复数旋转**（相邻维配对），与 qwen/llama 两半式配对不等价；
- 路由：softmax → topk → ×routed_scaling_factor（无 topk 概率归一化）；
- MoE：routed 输出 + shared_experts(输入残差)；experts 存盘 2D per-expert。

## 3. 验证链（跑通≠写对）

| 层级 | 方法 | 结果 |
|---|---|---|
| ① decode 内核 | 吸收式内核 vs 稠密展开参考（同权重同缓存，S=1/130/600 跨 0/1/3 块）| **abs_max = 0.0（位级一致）** |
| ② CPU 全模型 | 本实现 vs transformers 5.15 同款类，同 state dict（3D→2D 专家键映射），fp32，逐层显式因果掩码规范链 | max 1e-6、**top-1 100%**（T=17/20/23）|
| ③ 引擎 prefill | 引擎（bf16 GPU flash）vs 稠密手工参考末行 | mean 0.003、**top-1 100%** |
| ④ 引擎 decode | 吸收式内核路径 vs 稠密参考（逐 token 整段重算）| mean 0.003–0.03、**top-1 全 Y** |

额外回归：Qwen3-MoE e2e（HF 对照 top-1 100%）与 pytest 49+4 全过——RMSNorm 修复不改变 bf16 引擎行为。

## 4. KV 压缩率账本（V2-Lite：h=16、kv_lora=512、rope=64、qk_nope=128、v=128）

每 token 每层存储（元素）：

| 方案 | 元素 | vs GQA |
|---|---|---|
| 等效 GQA（16 kv 头 × (K 128 + V 128)）| 4096 | 1× |
| 简化 naive MLA（潜在 512 + **每头** rope key 64×16）| 1536 | 2.67× |
| **本实现/论文 MLA（潜在 512 + 共享 rope key 64）** | **576** | **7.11×** |

- 576 = V3 官方口径（官方推理只缓存潜在 + 共享 rope key）；
- bf16 下 576 元素 = 1152 B/token/层；decode 带宽账本：GQA 读逐头 K/V 8192 B vs MLA 1152 B（7.1×），前提是读一次跨头共享——吸收式内核按行 program、c 块逐头复用实现；
- 若 naive 地"每头一份 rope key"（1536 元素，2.67×）会丢一半收益——路线图旧口径 2.7× 即此假设，实测按论文结构修正为 7.1×。

## 5. 上游怪癖（transformers 5.15，作为对照经验记录）

1. DeepseekV2 无缓存全模型前向不产生因果掩码（create_causal_mask → None），中间行语义非规范——parity 只能取各行末行或逐层显式掩码；
2. experts 内存是**直挂 3D Parameter**（state_dict 键无 `.weight` 后缀），存盘才转 2D per-expert；
3. flash-attn varlen 要求 v 的 head dim == k 的 head dim（MLA 128 vs 192 不符）→ 本实现 v 零填充到 qk 维再截断（数学恒等）；
4. HF 增量 cache API（DynamicCache）在该模型上也出现索引越界（未深入），E2E 参考改用逐行整段重算。

## 6. 设计决策与诚实边界

- **吸收式解码**：W_UK 吸收进 q（每步每头一次 128×512）、W_UV 吸收进输出（kernel 内只累加 [h, kv_lora] 潜在加权和，出内核一次 v 投影）→ 每 token 每层只读 576 元素；
- decode 快路径要求 kv_b 有 float 视图（未量化 / int4 dual-path w_deq）；纯 int4/fp8（streaming 强制）→ **稠密兜底解码**（整段缓存逐层稠密化 + varlen），正确性等价、带宽优势消失（自动强制 eager）——真模型 int4 8GB 可跑但 decode 走兜底，性能预期差，诚实标注；
- cache-shaped 行（spec verify / 分块续写 / 前缀复用）→ 缓存前缀 gather + kv_b 稠密化 + 行主序组装（每层一次性）；spec 步 MLA 保持 eager（组装动态形状不可入图）；
- fp8 KV + MLA fused cache 未实现（断言报错）；MLA cache 每层 576 元素/token，容量账本见上；
- 未做：真实 DeepSeek-V2-Lite checkpoint 验证（bf16 ≈30GB 下载；hub 从 WSL 当前超时，`_ds_probe.py` 就绪）；decode 带宽实测（需真实模型或长序列 toy 基准，见阶段 2b 报告）；量化段式等。
