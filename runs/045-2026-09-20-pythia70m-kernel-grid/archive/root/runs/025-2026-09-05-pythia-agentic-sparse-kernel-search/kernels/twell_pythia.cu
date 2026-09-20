// Sakana TwELL T2D descendant; see ../upstream/LICENSE and ../SOURCE_AUDIT.md.
// Retains tile-wise BF16/index packing, warp broadcast, and 8 outputs/lane.
// P0 correctness adaptations: signed exact packing (no cap), dynamic shapes,
// FP32 products/accumulation, fused bias, and PyTorch's current CUDA stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <climits>

namespace {
constexpr int TILE = 256, WORDS = 288;
constexpr unsigned MASK = 0xffffffffu;

__global__ void pack_exact(const __nv_bfloat16* x, uint32_t* packed, int k, int tiles) {
    const int lane = threadIdx.x, row = blockIdx.x, tile = blockIdx.y;
    auto* dst = packed + (int64_t(row) * tiles + tile) * WORDS;
    int prefix = 0;
    #pragma unroll
    for (int group = 0; group < 8; ++group) {
        int col = tile * TILE + group * 32 + lane;
        auto value = col < k ? x[int64_t(row) * k + col] : __float2bfloat16(0.f);
        bool present = __bfloat162float(value) != 0.f;
        unsigned mask = __ballot_sync(MASK, present);
        int rank = __popc(mask & ((1u << lane) - 1u));
        if (present) dst[1 + prefix + rank] = (uint32_t(__bfloat16_as_ushort(value)) << 16) | uint32_t(col);
        prefix += __popc(mask);
    }
    if (lane == 0) dst[0] = prefix;
    // Initialize unused words too: T2D's warp-coalesced reads must never read
    // undefined allocator contents, even when the fetched lane is not used.
    for (int pos = prefix + 1 + lane; pos < WORDS; pos += 32) dst[pos] = 0;
}

__global__ void t2d_exact(const uint32_t* packed, const __nv_bfloat16* weight,
                          const __nv_bfloat16* bias, __nv_bfloat16* out,
                          int n, int tiles) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int out_col = blockIdx.y * TILE + lane * 8;
    float acc[8] = {0.f};
    for (int tile = 0; tile < tiles; ++tile) {
        const auto* src = packed + (int64_t(row) * tiles + tile) * WORDS;
        int count = __shfl_sync(MASK, int(src[lane]), 0);
        for (int chunk = 0; chunk <= count / 32; ++chunk) {
            uint32_t reg = src[chunk * 32 + lane];
            int first = chunk == 0 ? 1 : 0;
            int last = min(31, count - chunk * 32);
            for (int index = first; index <= last; ++index) {
                uint32_t word = __shfl_sync(MASK, reg, index);
                int col = word & 0xffffu;
                float value = __bfloat162float(__ushort_as_bfloat16(word >> 16));
                // All 32 lanes reach every shuffle, including N=128 tails.
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (out_col + j < n)
                        acc[j] = fmaf(value, __bfloat162float(weight[int64_t(col) * n + out_col + j]), acc[j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) if (out_col + j < n) {
        float value = acc[j] + (bias ? __bfloat162float(bias[out_col + j]) : 0.f);
        out[int64_t(row) * n + out_col + j] = __float2bfloat16_rn(value);
    }
}
}

void twell_linear(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                  torch::Tensor packed, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be contiguous CUDA BF16 [M,K]");
    TORCH_CHECK(weight.is_cuda() && weight.device() == x.device() && weight.scalar_type() == at::kBFloat16 && weight.dim() == 2 && weight.is_contiguous(), "weight must be same-device contiguous BF16 [K,N]");
    auto m = x.size(0), k = x.size(1), n = weight.size(1), tiles = (k + TILE - 1) / TILE;
    TORCH_CHECK(m > 0 && m <= INT_MAX && k > 0 && k <= 65535 && n > 0 && n <= 65535 && weight.size(0) == k, "unsupported dimensions");
    TORCH_CHECK(out.device() == x.device() && out.scalar_type() == at::kBFloat16 && out.dim() == 2 && out.size(0) == m && out.size(1) == n && out.is_contiguous(), "invalid output buffer");
    TORCH_CHECK(packed.device() == x.device() && packed.scalar_type() == at::kInt && packed.is_contiguous() && packed.numel() == m * tiles * WORDS, "invalid packing buffer");
    TORCH_CHECK(bias.device() == x.device() && bias.scalar_type() == at::kBFloat16 && bias.dim() == 1 && bias.is_contiguous() && (bias.numel() == 0 || bias.numel() == n), "invalid bias");
    const c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    pack_exact<<<dim3(m, tiles), 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<uint32_t*>(packed.data_ptr()), k, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    t2d_exact<<<dim3(m, (n + TILE - 1) / TILE), 32, 0, stream>>>(reinterpret_cast<const uint32_t*>(packed.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), bias.numel() ? reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr()) : nullptr, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), n, tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("linear", &twell_linear); }
