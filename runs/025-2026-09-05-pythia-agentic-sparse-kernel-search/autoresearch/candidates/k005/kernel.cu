// K005: output-width-specialized descendant of K001 / Sakana T2D.
// Exact signed compaction and ascending-K FP32 FMA order are unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <climits>

namespace {
constexpr unsigned FULL = 0xffffffffu;

template<int VECTOR_WIDTH, int ROW_WARPS, bool VECTOR_WEIGHT>
__global__ void fused_exact(const __nv_bfloat16* x,
                            const __nv_bfloat16* weight,
                            const __nv_bfloat16* bias,
                            __nv_bfloat16* out, int m, int k, int n) {
    constexpr int OUTPUT_TILE = 32 * VECTOR_WIDTH;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5);
    if (row >= m) return;
    const int out_col = blockIdx.y * OUTPUT_TILE + lane * VECTOR_WIDTH;
    float acc[VECTOR_WIDTH] = {0.f};
    for (int begin = 0; begin < k; begin += 32) {
        const int input_col = begin + lane;
        const float lane_value = input_col < k
            ? __bfloat162float(x[int64_t(row) * k + input_col]) : 0.f;
        unsigned remaining = __ballot_sync(FULL, lane_value != 0.f);
        while (remaining) {
            const int source_lane = __ffs(remaining) - 1;
            const int input_index = begin + source_lane;
            const float value = __shfl_sync(FULL, lane_value, source_lane);
            if (out_col < n) {
                if (VECTOR_WEIGHT) {
                    if constexpr (VECTOR_WIDTH == 4) {
                        const uint2 words = *reinterpret_cast<const uint2*>(
                            weight + int64_t(input_index) * n + out_col);
                        const uint32_t pairs[2] = {words.x, words.y};
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const unsigned short bits =
                                (pairs[j / 2] >> (16 * (j & 1))) & 0xffffu;
                            acc[j] = fmaf(value,
                                __bfloat162float(__ushort_as_bfloat16(bits)),
                                acc[j]);
                        }
                    } else {
                        #pragma unroll
                        for (int base = 0; base < VECTOR_WIDTH; base += 8) {
                            const uint4 words = *reinterpret_cast<const uint4*>(
                                weight + int64_t(input_index) * n + out_col + base);
                            const uint32_t pairs[4] = {words.x, words.y, words.z, words.w};
                            #pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                const unsigned short bits =
                                    (pairs[j / 2] >> (16 * (j & 1))) & 0xffffu;
                                acc[base + j] = fmaf(value,
                                    __bfloat162float(__ushort_as_bfloat16(bits)),
                                    acc[base + j]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < VECTOR_WIDTH; ++j) if (out_col + j < n)
                        acc[j] = fmaf(value, __bfloat162float(
                            weight[int64_t(input_index) * n + out_col + j]), acc[j]);
                }
            }
            remaining &= remaining - 1;
        }
    }
    #pragma unroll
    for (int j = 0; j < VECTOR_WIDTH; ++j) if (out_col + j < n) {
        const float result = acc[j] +
            (bias ? __bfloat162float(bias[out_col + j]) : 0.f);
        out[int64_t(row) * n + out_col + j] = __float2bfloat16_rn(result);
    }
}

template<int VECTOR_WIDTH, int ROW_WARPS>
void launch(const __nv_bfloat16* x, const __nv_bfloat16* weight,
            const __nv_bfloat16* bias, __nv_bfloat16* out,
            int m, int k, int n, cudaStream_t stream, bool vector_weight) {
    constexpr int OUTPUT_TILE = 32 * VECTOR_WIDTH;
    const dim3 grid((m + ROW_WARPS - 1) / ROW_WARPS,
                    (n + OUTPUT_TILE - 1) / OUTPUT_TILE);
    if (vector_weight)
        fused_exact<VECTOR_WIDTH, ROW_WARPS, true>
            <<<grid, ROW_WARPS * 32, 0, stream>>>(x, weight, bias, out, m, k, n);
    else
        fused_exact<VECTOR_WIDTH, ROW_WARPS, false>
            <<<grid, ROW_WARPS * 32, 0, stream>>>(x, weight, bias, out, m, k, n);
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
    const auto m64 = x.size(0), k64 = x.size(1), n64 = weight.size(1);
    TORCH_CHECK(m64 > 0 && m64 <= INT_MAX && k64 > 0 && k64 <= 65535 &&
                n64 > 0 && n64 <= 65535 && weight.size(0) == k64,
                "unsupported dimensions");
    TORCH_CHECK(out.device() == x.device() &&
                out.scalar_type() == at::kBFloat16 && out.dim() == 2 &&
                out.size(0) == m64 && out.size(1) == n64 && out.is_contiguous(),
                "invalid output buffer");
    TORCH_CHECK(bias.device() == x.device() &&
                bias.scalar_type() == at::kBFloat16 && bias.dim() == 1 &&
                bias.is_contiguous() && (bias.numel() == 0 || bias.numel() == n64),
                "invalid bias");
    const int m = static_cast<int>(m64), k = static_cast<int>(k64), n = static_cast<int>(n64);
    const c10::cuda::CUDAGuard guard(x.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
    const auto* bp = bias.numel()
        ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr;
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    const bool vector_weight = n % 8 == 0 &&
        reinterpret_cast<uintptr_t>(wp) % 16 == 0;
    if (n == 128)
        launch<4, 8>(xp, wp, bp, op, m, k, n, stream, vector_weight);
    else
        launch<16, 4>(xp, wp, bp, op, m, k, n, stream, vector_weight);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &linear); }
