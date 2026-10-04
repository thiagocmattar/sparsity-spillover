// Exact gate pass over already-rounded K050 norms and K019 RoPE operands.
// The pass is identical code for scalar and corner gates, selected at runtime.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>
using bf16 = __nv_bfloat16;

__device__ __forceinline__ bf16 gated(const bf16* x, int index, int width, int mode, float threshold) {
    float value = __bfloat162float(x[index]);
    bool live;
    if (mode == 1) live = value >= threshold;
    else if (mode == 3) live = fabsf(value) >= threshold;
    else {
        int row = index / width, col = index % width;
        float corner = __bfloat162float(x[(row / 8 * 8) * width + col / 16 * 16]);
        live = mode == 4 ? fabsf(corner) > threshold : corner > threshold;
    }
    return live ? x[index] : __float2bfloat16_rn(0.f);
}

__global__ void gate_many(const bf16* x0, const bf16* x1, const bf16* x2,
                         bf16* y0, bf16* y1, bf16* y2, int count, int width,
                         int m0, int m1, int m2, float t0, float t1, float t2) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    y0[index] = gated(x0, index, width, m0, t0);
    y1[index] = gated(x1, index, width, m1, t1);
    if (x2) y2[index] = gated(x2, index, width, m2, t2);
}

void forward(std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs,
             std::vector<int> modes, std::vector<double> thresholds) {
    auto x = inputs.at(0);
    int groups = inputs.size();
    TORCH_CHECK((groups == 2 || groups == 3) && outputs.size() == groups &&
                modes.size() == groups && thresholds.size() == groups, "Two or three gate operands required");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                (x.dim() == 3 || x.dim() == 4) && x.numel() > 0 &&
                x.size(-2) % 8 == 0 && x.size(-1) % 16 == 0, "Aligned CUDA BF16 [B,T,D] or [B,H,T,d] required");
    for (int i = 0; i < groups; ++i) {
        TORCH_CHECK(modes[i] >= 1 && modes[i] <= 4 && std::isfinite(thresholds[i]) && thresholds[i] >= 0,
                    "Invalid gate mode/threshold");
        for (auto tensor : {inputs[i], outputs[i]})
            TORCH_CHECK(tensor.device() == x.device() && tensor.scalar_type() == x.scalar_type() &&
                        tensor.sizes() == x.sizes() && tensor.is_contiguous(), "Operand type/layout/shape mismatch");
        for (int j = 0; j < groups; ++j)
            TORCH_CHECK(outputs[i].data_ptr() != inputs[j].data_ptr(), "Out-of-place gates required");
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ptr = [](torch::Tensor t) { return reinterpret_cast<bf16*>(t.data_ptr()); };
    int count = x.numel(), width = x.size(-1);
    gate_many<<<(count + 255) / 256, 256, 0, stream>>>(
        ptr(inputs[0]), ptr(inputs[1]), groups == 3 ? ptr(inputs[2]) : nullptr,
        ptr(outputs[0]), ptr(outputs[1]), groups == 3 ? ptr(outputs[2]) : nullptr, count, width,
        modes[0], modes[1], groups == 3 ? modes[2] : 0,
        thresholds[0], thresholds[1], groups == 3 ? thresholds[2] : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("forward", &forward); }
