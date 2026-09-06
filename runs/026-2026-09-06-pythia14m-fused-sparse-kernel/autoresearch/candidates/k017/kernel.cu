// K017: exact threshold + warp-compacted projection. Derived from Run025 K001
// (Sakana T2D lineage; see Run025/upstream/LICENSE). No input-dependent cache.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

template<int E, int W>
__global__ void gated_linear(const __nv_bfloat16* x, const __nv_bfloat16* wt,
 const __nv_bfloat16* bias, __nv_bfloat16* out, int m, int k, int n, float threshold) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * W + (threadIdx.x >> 5);
    if (row >= m) return;
    int col = blockIdx.y * (32 * E) + lane * E;
    float acc[E] = {};
    for (int base = 0; base < k; base += 32) {
        float value = base + lane < k ? __bfloat162float(x[int64_t(row)*k+base+lane]) : 0.f;
        // This is exactly masked_fill(x < threshold, 0), including equality.
        if (value < threshold) value = 0.f;
        unsigned active = __ballot_sync(0xffffffffu, value != 0.f);
        while (active) {
            int source = __ffs(active) - 1;
            float v = __shfl_sync(0xffffffffu, value, source);
            #pragma unroll
            for (int j=0; j<E; ++j) if (col+j<n)
                acc[j] = fmaf(v, __bfloat162float(wt[int64_t(base+source)*n+col+j]), acc[j]);
            active &= active-1;
        }
    }
    #pragma unroll
    for (int j=0; j<E; ++j) if(col+j<n)
        out[int64_t(row)*n+col+j] = __float2bfloat16_rn(acc[j] + (bias ? __bfloat162float(bias[col+j]) : 0.f));
}

template<int E> void launch(torch::Tensor x, torch::Tensor w, torch::Tensor b,
 torch::Tensor y, float threshold, int warps) {
    auto stream=at::cuda::getCurrentCUDAStream();
    auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto wp=reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    auto bp=b.numel()?reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()):nullptr;
    auto yp=reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
    int m=x.size(0), k=x.size(1), n=w.size(1);
    dim3 grid((m+warps-1)/warps, (n+32*E-1)/(32*E));
    if (warps==4) gated_linear<E,4><<<grid,128,0,stream>>>(xp,wp,bp,yp,m,k,n,threshold);
    else gated_linear<E,8><<<grid,256,0,stream>>>(xp,wp,bp,yp,m,k,n,threshold);
}

void linear(torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor y,
 double threshold, int elements, int warps) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.is_contiguous(), "CUDA BF16 contiguous [M,K] required");
    TORCH_CHECK(w.device()==x.device() && w.scalar_type()==x.scalar_type() && w.dim()==2 && w.is_contiguous() && w.size(0)==x.size(1), "invalid weight");
    TORCH_CHECK(b.device()==x.device() && b.scalar_type()==x.scalar_type() && b.dim()==1 && b.is_contiguous() && (b.numel()==0 || b.numel()==w.size(1)), "invalid bias");
    TORCH_CHECK(y.device()==x.device() && y.scalar_type()==x.scalar_type() && y.dim()==2 && y.is_contiguous() && y.size(0)==x.size(0) && y.size(1)==w.size(1), "invalid output");
    TORCH_CHECK(x.size(0)>0 && x.size(0)<=2147483647 && x.size(1)>0 && x.size(1)<=65535 && w.size(1)>0 && w.size(1)<=65535, "unsupported shape");
    TORCH_CHECK(std::isfinite(threshold) && threshold>=0 && (warps==4 || warps==8), "invalid configuration");
    const c10::cuda::CUDAGuard guard(x.device());
    if(elements==2) launch<2>(x,w,b,y,threshold,warps);
    else if(elements==4) launch<4>(x,w,b,y,threshold,warps);
    else if(elements==8) launch<8>(x,w,b,y,threshold,warps);
    else TORCH_CHECK(false, "elements must be 2,4,8");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("linear",&linear);}
