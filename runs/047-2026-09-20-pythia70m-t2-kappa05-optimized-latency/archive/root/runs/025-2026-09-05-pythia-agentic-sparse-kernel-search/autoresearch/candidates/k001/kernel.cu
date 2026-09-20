// K001: fused exact warp compaction / Sakana-derived T2D.
// Lineage: ../../../kernels/twell_pythia.cu and ../../../upstream/LICENSE.
// All signed nonzeros survive; ascending-K FP32 FMA and fused bias match P0.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <climits>

namespace {
constexpr unsigned FULL = 0xffffffffu;
constexpr int ROW_WARPS = 4, OUTPUT_TILE = 256;

// Each independent warp compacts one row's next 32 values into the same
// ascending coordinate order used by P0. The compact representation lives in
// registers and the ballot; no capacity, scratch buffer, or overflow exists.
template<bool VECTOR_WEIGHT>
__global__ void fused_exact(const __nv_bfloat16* x,
                            const __nv_bfloat16* weight,
                            const __nv_bfloat16* bias,
                            __nv_bfloat16* out, int m, int k, int n) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5);
    if (row >= m) return; // Entire independent warp exits together.
    const int out_col = blockIdx.y * OUTPUT_TILE + lane * 8;
    float acc[8] = {0.f};
    for (int begin = 0; begin < k; begin += 32) {
        const int input_col = begin + lane;
        const float lane_value = input_col < k
            ? __bfloat162float(x[int64_t(row) * k + input_col]) : 0.f;
        unsigned remaining = __ballot_sync(FULL, lane_value != 0.f);
        while (remaining) {
            const int source_lane = __ffs(remaining) - 1;
            const int input_index = begin + source_lane;
            const float value = __shfl_sync(FULL, lane_value, source_lane);
            // All 32 lanes participate above, even N=128's inactive tail.
            if (VECTOR_WEIGHT) {
                if (out_col < n) {
                    // n and out_col multiples of 8 => 16-byte alignment.
                    const uint4 words = *reinterpret_cast<const uint4*>(
                        weight + int64_t(input_index) * n + out_col);
                    const uint32_t pairs[4] = {words.x, words.y, words.z, words.w};
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const unsigned short bits =
                            (pairs[j / 2] >> (16 * (j & 1))) & 0xffffu;
                        acc[j] = fmaf(value,
                            __bfloat162float(__ushort_as_bfloat16(bits)), acc[j]);
                    }
                }
            } else {
                #pragma unroll
                for (int j = 0; j < 8; ++j) if (out_col + j < n)
                    acc[j] = fmaf(value, __bfloat162float(
                        weight[int64_t(input_index) * n + out_col + j]), acc[j]);
            }
            remaining &= remaining - 1;
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) if (out_col + j < n) {
        const float result = acc[j] +
            (bias ? __bfloat162float(bias[out_col + j]) : 0.f);
        out[int64_t(row) * n + out_col + j] = __float2bfloat16_rn(result);
    }
}
}

void linear(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
            torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 &&
                x.dim() == 2 && x.is_contiguous(),
                "x must be contiguous CUDA BF16 [M,K]");
    TORCH_CHECK(weight.device() == x.device() &&
                weight.scalar_type() == at::kBFloat16 &&
                weight.dim() == 2 && weight.is_contiguous(),
                "weight must be same-device contiguous CUDA BF16 [K,N]");
    const auto m = x.size(0), k = x.size(1), n = weight.size(1);
    TORCH_CHECK(m > 0 && m <= INT_MAX && k > 0 && k <= 65535 &&
                n > 0 && n <= 65535 && weight.size(0) == k,
                "unsupported dimensions");
    TORCH_CHECK(out.device() == x.device() &&
                out.scalar_type() == at::kBFloat16 && out.dim() == 2 &&
                out.size(0) == m && out.size(1) == n && out.is_contiguous(),
                "invalid output buffer");
    TORCH_CHECK(bias.device() == x.device() &&
                bias.scalar_type() == at::kBFloat16 && bias.dim() == 1 &&
                bias.is_contiguous() && (bias.numel() == 0 || bias.numel() == n),
                "invalid bias");
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((m + ROW_WARPS - 1) / ROW_WARPS,
                    (n + OUTPUT_TILE - 1) / OUTPUT_TILE);
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    const auto* bp = bias.numel()
        ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr;
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    // A contiguous tensor can still be an offset view, so inspect its base too.
    if (n % 8 == 0 && reinterpret_cast<uintptr_t>(wp) % 16 == 0)
        fused_exact<true><<<grid, ROW_WARPS * 32, 0, stream>>>(xp, wp, bp, op, m, k, n);
    else
        fused_exact<false><<<grid, ROW_WARPS * 32, 0, stream>>>(xp, wp, bp, op, m, k, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
