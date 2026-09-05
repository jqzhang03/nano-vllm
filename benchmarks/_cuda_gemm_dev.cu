// 阶段 1b（内核补课）：CUDA C 显式 SMEM fp16 GEMM 开发文件。
// 路线：naive（FMA + 单缓冲，每线程 1 输出）→ 大 tile/多输出 → cp.async 流水 →
// ldmatrix + mma.sync →（对比 cuBLAS / Triton）。
// 本文件当前版本：v1a_naive——教学起点：正确的显式 SMEM tile + 同步，但
// ①每线程仅 1 个累加器（无寄存器 tile 复用）②无流水（每 k 步 2 次 __syncthreads）
// ③A/B 行主序直存 → 读 As 每 k 同 bank（32-way conflict 隐患由编译器乱序缓解）。
// 目标：跑通正确性 + 拿到性能曲线的"地板"。
#include <torch/extension.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// v1a：naive SMEM FMA GEMM。BM=BN=BK=16，256 线程，每线程 1 输出。
// C[m,n] = A[m,k] @ B[k,n]，三者均 row-major。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK>
__global__ void __launch_bounds__(256) gemm_fma_naive_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* __restrict__ C, int M, int N, int K) {
    __shared__ __half As[BM * BK];
    __shared__ __half Bs[BK * BN];
    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int tid = threadIdx.x;              // 0..255
    const int my_m = m0 + tid / BN;           // 行内 m
    const int my_n = n0 + tid % BN;           // 列内 n
    float acc = 0.f;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 合作搬运 A tile 到 SMEM（As[m_local][kk] 行主序，行距 BK）
        for (int i = tid; i < BM * BK; i += 256) {
            int ml = i / BK, kl = i % BK;
            As[ml * BK + kl] = A[(m0 + ml) * K + (k0 + kl)];
        }
        // 合作搬运 B tile（Bs[kl][n_local] 行主序，行距 BN）
        for (int i = tid; i < BK * BN; i += 256) {
            int kl = i / BN, nl = i % BN;
            Bs[kl * BN + nl] = B[(k0 + kl) * N + (n0 + nl)];
        }
        __syncthreads();
        // 每线程：自己的那行 As × 自己那列 Bs
        const __half* arow = &As[(tid / BN) * BK];
        for (int kk = 0; kk < BK; ++kk) {
            acc += __half2float(arow[kk]) * __half2float(Bs[kk * BN + (tid % BN)]);
        }
        __syncthreads();  // 下一轮覆写 As/Bs 前保证所有线程读完
    }
    C[my_m * N + my_n] = __float2half(acc);
}

torch::Tensor gemm_fma_naive(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "cuda tensors required");
    TORCH_CHECK(a.scalar_type() == torch::kHalf && b.scalar_type() == torch::kHalf,
                "fp16 tensors required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
                "shape mismatch");
    const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(1);
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                "M/N/K must be multiples of 16 for the naive kernel");
    auto c = torch::empty({M, N}, a.options());
    constexpr int BM = 16, BN = 16, BK = 16;
    dim3 grid(N / BN, M / BM);
    gemm_fma_naive_kernel<BM, BN, BK><<<grid, 256>>>(
        reinterpret_cast<const __half*>(a.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<torch::Half>()), M, N, K);
    return c;
}

// ---------------------------------------------------------------------------
// v2a：mma.sync m16n8k16 手工 fragment（无 ldmatrix，正确性优先）。
// 布局公式来自 PTX ISA 文档（m16n8k16 fragment 映射，博客实证）：
//   A 16x16 row-major, 8 half/线程 → 4x uint32（每 uint32 = 行内连续 2 half）
//   B 16x8  col-major, 4 half/线程 → 2x uint32（k 对: 2*lane%4 + 8*(l>>1)）
//   C 16x8, 4 float/线程
// block tile = C[16 x 64]（8 warp × 各 16×8）；BK=32（每步 2 次 k16 MMA）。
// 与 v1a 的差距点：每线程 1 acc → 4 acc；FMA → MMA（TC）；K 全程寄存器累加。
// ---------------------------------------------------------------------------
struct FragA16x16 {
    static __device__ int row(int lane, int l) {
        return (lane >> 2) + 8 * ((l >> 1) & 1);
    }
    static __device__ int col(int lane, int l) {
        return 2 * (lane & 3) + (l & 1) + 8 * (l >> 2);
    }
};
struct FragB16x8 {  // B 按 (k, n) 坐标；col-major 语义
    static __device__ int k(int lane, int l) {
        return 2 * (lane & 3) + (l & 1) + 8 * (l >> 1);
    }
    static __device__ int n(int lane, int l) { return lane >> 2; }
};
struct FragC16x8 {
    static __device__ int row(int lane, int l) { return (lane >> 2) + 8 * (l >> 1); }
    static __device__ int col(int lane, int l) { return 2 * (lane & 3) + (l & 1); }
};

template <int BK>
__global__ void __launch_bounds__(256) gemm_mma16x64_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* __restrict__ C, int M, int N, int K) {
    __shared__ __align__(16) __half As[16 * BK];  // 行主序: As[m_local*BK + k_local]
    __shared__ __align__(16) __half Bs[BK * 64];  // 行主序镜像: Bs[k_local*64 + n_local]
    const int m0 = blockIdx.y * 16;
    const int n0 = blockIdx.x * 64;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;          // 0..7 → n 片
    const int lane = tid & 31;

    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < 16 * BK; i += 256)
            As[i] = A[(m0 + i / BK) * K + (k0 + i % BK)];
        for (int i = tid; i < BK * 64; i += 256)
            Bs[i] = B[(k0 + i / 64) * N + (n0 + i % 64)];
        __syncthreads();
#pragma unroll
        for (int kh = 0; kh < BK; kh += 16) {
            // A fragment: p∈0..3 → uint32 = As[row][2ti + 8*(p>>1) + kh] 起 2 half
            // （kh 是 A 在本 BK 片内的 16 列偏移）
            uint32_t a_reg[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int r = (lane >> 2) + 8 * (p & 1);
                int cc = 2 * (lane & 3) + 8 * (p >> 1);
                a_reg[p] = *reinterpret_cast<const uint32_t*>(&As[r * BK + kh + cc]);
            }
            // B fragment: 2x uint32；Bs 行主序 k×n → k 对不连续 → 标量读 + 打包
            uint32_t b_reg[2];
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                int nl = warp * 8 + (lane >> 2);
                uint16_t lo = __half_as_ushort(Bs[(kh + 2 * (lane & 3) + 8 * q) * 64 + nl]);
                uint16_t hi = __half_as_ushort(Bs[(kh + 2 * (lane & 3) + 8 * q + 1) * 64 + nl]);
                b_reg[q] = (uint32_t)lo | ((uint32_t)hi << 16);
            }
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                  "r"(b_reg[0]), "r"(b_reg[1]));
        }
        __syncthreads();
    }
    // C 写回：warp 负责 C[m0:m0+16][n0+warp*8 : +8]。
    // C fragment l=0,1 在 (g, 2ti)/(g, 2ti+1) → half2（4B 对齐，t4=2ti 为偶）
    const int g = lane >> 2, t4 = 2 * (lane & 3);
    __half* cout = &C[(m0 + g) * N + (n0 + warp * 8 + t4)];
    *reinterpret_cast<half2*>(cout) =
        make_half2(__float2half(c[0]), __float2half(c[1]));
    cout += 8 * N;
    *reinterpret_cast<half2*>(cout) =
        make_half2(__float2half(c[2]), __float2half(c[3]));
}

torch::Tensor gemm_mma16x64(torch::Tensor a, torch::Tensor b) {
    const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(1);
    TORCH_CHECK(M % 16 == 0 && N % 64 == 0 && K % 32 == 0,
                "M%16/N%64/K%32 must be 0 for gemm_mma16x64");
    auto c = torch::empty({M, N}, a.options());
    dim3 grid(N / 64, M / 16);
    gemm_mma16x64_kernel<32><<<grid, 256>>>(
        reinterpret_cast<const __half*>(a.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<torch::Half>()), M, N, K);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fma_naive", &gemm_fma_naive);
    m.def("gemm_mma16x64", &gemm_mma16x64);
}
