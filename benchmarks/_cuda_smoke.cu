// CUDA C 开发 smoke：验证 nvcc 12.8 + sm_120 编译链 + torch 扩展加载 + 内核执行。
// 只验证工具链（不含 MMA fragment 布局——那是 GEMM 开发脚本里用 torch 对照细调的事）。
// 用法：python benchmarks/_cuda_smoke.py
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

__global__ void saxpy_kernel(const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ z, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = a * x[i] + y[i];
}

__global__ void half_add_kernel(const __half* __restrict__ x,
                                const __half* __restrict__ y,
                                __half* __restrict__ z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = __hadd(x[i], y[i]);
}

int cuda_arch() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    return prop.major * 100 + prop.minor;
}

torch::Tensor saxpy(torch::Tensor x, torch::Tensor y, double a) {
    auto z = torch::empty_like(x);
    int n = (int)x.numel();
    saxpy_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                           z.data_ptr<float>(), (float)a, n);
    return z;
}

torch::Tensor half_add(torch::Tensor x, torch::Tensor y) {
    auto z = torch::empty_like(x);
    int n = (int)x.numel();
    half_add_kernel<<<(n + 255) / 256, 256>>>(
        reinterpret_cast<const __half*>(x.data_ptr<torch::Half>()),
        reinterpret_cast<const __half*>(y.data_ptr<torch::Half>()),
        reinterpret_cast<__half*>(z.data_ptr<torch::Half>()), n);
    return z;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("saxpy", &saxpy);
    m.def("half_add", &half_add);
    m.def("cuda_arch", &cuda_arch);
}
