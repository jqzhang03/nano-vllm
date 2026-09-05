# CUDA C 手写 GEMM 内核报告（阶段 1b：SMEM 显式编程 / SASS / split-K / persistent）

> 条件：RTX 5060 Ti 16GB (sm_120, Blackwell) / WSL2 / CUDA 12.8 (nvcc 12.8.93, conda) /
> torch 2.8.0+cu128 / fp16，row-major `C[M,N] = A[M,K]@B[K,N]`。
> 对照基线：cuBLAS fp16（torch.matmul，实测 39-41 TFLOPS @ 4096³——远低于纸面 48.5，
> 本机（WSL2/小卡）的 TC 实测峰值；硬件锚点见 `_kernel_roofline.md`）。
> 所有正确性对照：fp64 参考，rel 误差 ~2-4e-4（fp16 输入的表示噪声级）。

## 1. 为什么做（目标）

阶段 1 的空白：Triton 自动管理 SMEM/流水/布局，无法回答"显式 SMEM tile 怎么写、
bank conflict 长什么样、SASS 里发生了什么"。本报告用 **CUDA C 从零手写 GEMM 内核链**
补上，全部带 SASS 证据。

## 2. 工具链（4 个坑，已固化为 benchmarks/_cuda_common.py）

| # | 坑 | 解法 |
|---|---|---|
| 1 | pip 的 `nvidia-cuda-nvcc-cu12` wheel（12.6/12.8/12.9）**只有 ptxas，没有 nvcc 主程序**（新拆包） | conda `-c nvidia cuda-nvcc=12.8.93` |
| 2 | 系统 gcc 15.2 > nvcc 12.8 上限（host_config.h `#error`）且 pybind11 模板推导崩（99 错） | conda gcc 14.3 作 host 编译器 |
| 3 | `CUDAHOSTCXX` 环境变量不生效 | 建 `env/bin/gcc`、`g++` symlink + PATH 前置 |
| 4 | 系统 cuobjdump/nvdisasm 12.4 **无法解码 SM120 SASS** | conda `cuda-cuobjdump/cuda-nvdisasm=12.8.90` |

配套：`ninja`（torch cpp_extension 必需）、`TORCH_CUDA_ARCH_LIST=12.0`（sm_120 cubin）。

## 3. 内核演进（每步正确性全过 + SASS 实证）

| 版本 | 结构 | 4096³ | % cuBLAS | SASS 关键证据 |
|---|---|---|---|---|
| v1a `gemm_fma_naive` | 16×16 tile，每线程 1 acc，FMA，SMEM 单缓冲 | 1.6 TFLOPS | 4% | 无 MMA；FFMA×16；regs=37 |
| v2a `gemm_mma16x64` | 16×64 tile，每 warp 1 个 m16n8 tile，手工 fragment | 6.0 TFLOPS | 15% | HMMA×2，**NOP×13/2 MMA**（TC 延迟暴露）；regs=32 |
| v2b `gemm_v2b` | **64×128 tile，8 warps = 2wm×4wn，每 warp 8 个 m16n8 tile**（32 acc regs）；BsT[n][k] 转置布局 → fragment uint32 直读 | **20.7 TFLOPS** | **53%** | HMMA×16、NOP/mma 6.5→1.3；regs=60、smem 13824、无 spill |

**每步的教训**：
- v2a：fragment 布局**先查 PTX ISA 公式再写**（A: `row=lane>>2+8*((l>>1)&1)`、col 对称），一次写对，省掉试错循环。坑：C fragment 的 l=2,3 在 **+8 行**（曾写 +1 行）；fp32→fp16 输出必须 half2（float2 会越 2 列写）。
- v2b 首版全错 + 诡异模式（看似 XOR 实为 tid 区间 [64,192)）：写了两个隔离探针（`store_probe` 常量写验证几何 / `probe_tile00` 单 tile），扫描写分布**一眼定位 wm = tid>>6 产生 4 值**（应为 `warp>>2 = tid>>7`，8 warps 是 2wm×4wn 不是 4wm×2wn）→ 行 64+ 越界写。**教训：warp 布局位运算必须对照 warp 数验证；诡异错误先画 block 布局图 + 探针隔离，别冥想。**

## 4. bank-conflict 消融（理论 → 实测闭环，提交 f76f5e7）

BsT 转置布局的**写路径**（B 加载散写 `BsT[n*BS+k]`）：BS=32（行距 32 half）时
bank = (n·16 + k/2) mod 32 → n 每 2 行一循环 → **16-way 写冲突**；BS=34（+2 pad）时
bank = (17n + k/2) mod 32，gcd(17,32)=1 → 16 个不同 bank。实测（50 iter）：

| BsT 行距 | 4096³ | 16384³ | M=64 |
|---|---|---|---|
| 32（无 pad） | 11.0 TFLOPS | 11.1 | 7.1 |
| 34（pad） | **20.7（+88%）** | **20.8（+87%）** | 8.8（+24%，启动受限区不明显） |

教训：**写路径的 bank 分析以"行距与 32 的 gcd"为核心**；优化收益取决于瓶颈形态
（大矩阵是吞吐型 → bank 冲突直接进关键路径；小 M 启动受限 → 不明显）。

## 5. split-K（赢在"block 数 < SM 数"的形态）

实现：K 切 S 段，`grid.z = S`，每 block 算 fp32 部分和 → `Cpart[S][M][N]` + 归约 kernel。
（不用 fp32 原子加，避免归约顺序不确定性。）

| 形状 | v2b | S=2 | S=4 | S=8 |
|---|---|---|---|---|
| M=64 N=4096 K=4096（32 C-block < 36 SM） | 8.8 | 12.5（1.42×） | **14.0（1.59×，复测区间 1.59-1.66×）** | 12.1（1.38×） |
| M=256（128 blocks，不缺并行度） | 18.4 | 17.3（0.94×） | 16.4（0.89×） | 14.1（0.77×） |

**诚实结论**：split-K 的收益场景很窄——**只有 C-block 数不足时**（M=64 只有 32 个 block）
才赢；并行度够时部分和的写读流量（S×M×N×4B×2）是纯开销，S 越大亏越多。S=8 相对 S=4
的衰减 = 流量 ×2（M=64: 8MB vs 4MB 往返，~21µs vs 143µs 计算 @14 TFLOPS）。

## 6. persistent kernel（本机全场景不赢——如实记录）

固定 grid（36/72/108/144 block）循环领取 C tile。结果：**0.45-0.98×**，4096³ 用 nb=108
为 18.5 vs 20.9（-11%）。

**为什么亏**：GPU 硬件 block 分发（GigaThread）近零成本且**动态均衡**（先完成 SM 领下一
个 block）；persistent 把 tile 串行钉在 block 上，失去跨 SM 迁移，慢 tile（L2 命中差）
拖累整体；且每 tile 间的 SMEM 重载 + 同步无法与别的 block 重叠。persistent 的真正价值
在**每 block 有昂贵 prologue**（如分配/预热/多 kernel 融合）的场景——本实验不存在。
nb=36（=SM 数）时最差（0.45×）：每 block 串太多 tile，负载不均被放大。

## 7. 最终数字（高迭代复测，iters=200 中位数×5 组）

见 `_cuda_final_verify.py` 输出（v1a 用 iters=30：naive 在 16384³ 太慢）。

| 项 | 数字（复测） |
|---|---|
| cuBLAS 4096³ | 39-41 TFLOPS |
| v1a → v2a → v2b（4096³） | 1.6 → 6.1 → 20.8 TFLOPS（4% → 15% → 53%） |
| v2b 16384×4096 | 20.8 TFLOPS（53%） |
| v2b M=64 | 8.8 TFLOPS（~29%，cuBLAS 30.5 档）——启动受限，接 split-K |
| pad vs nopad（4096³） | 1.86× |
| split-K S=4（M=64） | 1.59× → 14.0 TFLOPS（cuBLAS 的 ~46%） |
| persistent 108（4096³） | 0.89×（亏损，时间比 1.12） |

## 8. 未做（诚实边界）

- **cp.async / TMA 流水**：当前每 BK 步同步加载→计算（v2b 的 NOP×21 仍含 SMEM 读等待
  与 barrier）；cp.async 双缓冲是到 cuBLAS 80%+ 的主要下一步（未实现）。
- **ldmatrix**：fragment 用手工 uint32 读（正确、无布局魔法），LDS 指令数偏多（LDS×32
  vs ldmatrix 理论 ~1/4）；ldmatrix 需要 canonical 8×8 布局（CUTLASS 式 swizzle），未做。
- **只做了 fp16**：int4/fp8 的 CUDA C 变体未做（阶段 1 的 Triton int4 已覆盖业务）。
- 未做 warp-specialization、双精度、split-K 的 wave 量化模型。
- cuBLAS 剩余 47% 差距来源（按 SASS/结构推断，未逐一验证）：cp.async 流水、更优
  C tile 形状（128×256）、L2 swizzle（GROUP_M 类）、TMA + tcgen05（Blackwell 原生路径，
  mma.sync 已是兼容模式）。

## 9. 面试要点

- **能讲"从零到 53% cuBLAS 的四步 + 每步的 SASS 证据"**：FMA → MMA → 多 tile ILP → 转置
  布局；NOP/mma 比是"TC 延迟是否被隐藏"的可视化指标。
- **bank-conflict 不靠背**：行距与 32 的 gcd 决定冲突度，pad 一个 half 消冲突（+88% 实测）。
- **split-K/persistent 的诚实边界**：split-K 只在并行度不足时赢（M=64 +66%），persistent
  在本机全亏（硬件分发已近零成本）——"知道机制在哪些形态下不成立"。
- **探针方法论**：诡异错误模式 → 常量写探针验证几何 + 单 tile 探针隔离计算，比冥想快。
- 工具链故事（nvcc 12.8 拆包/gcc15/pybind11/SM120 SASS）是"环境工程能力"的实证。
