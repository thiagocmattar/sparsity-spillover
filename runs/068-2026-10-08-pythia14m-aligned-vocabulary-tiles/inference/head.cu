// Gate decisions remain FP32. Tensor cores consume FP16 operands and emit all V logits.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cmath>

__global__ void prepare_kernel(const float* x, half* packed, int* support, int m, float threshold){
    int block=blockIdx.x, tid=threadIdx.x;
    // Each warp writes the same eight-bit certificate; only warp zero stores it.
    unsigned bits=__ballot_sync(0xffffffffu, (tid&31)<8 && x[block*16*128+(tid&31)*16]>threshold);
    if(tid==0) support[block]=bits;
    for(int offset=tid;offset<16*128;offset+=blockDim.x)
        packed[block*16*128+offset]=__float2half_rn(x[block*16*128+offset]);
}

template<int M,int N,bool Skip,bool Count>
__global__ void project_kernel(const half* x,const int* support,const half* weight,
                              half* out,long long* stats,int v){
    constexpr int Atoms=N/(4*8);
    int lane=threadIdx.x&31, warp=threadIdx.x/32;
    int first=blockIdx.x*M, col=blockIdx.y*N+warp*Atoms*8;
    unsigned live=support[first/16];
    unsigned pending=Skip?live:0xffu;
    __shared__ __align__(16) half stores[M*(N+8)];
    float acc[Atoms][4]={};
    int issued=0, bypassed=8-__popc(pending);
    while(pending){
        int group=__ffs(pending)-1; pending&=pending-1; ++issued;
        int k=group*16+(lane%4)*2;
        unsigned a0=*reinterpret_cast<const unsigned*>(x+(first+lane/4)*128+k);
        unsigned a2=*reinterpret_cast<const unsigned*>(x+(first+lane/4)*128+k+8);
        unsigned a1=0,a3=0;
        if constexpr(M==16){
            a1=*reinterpret_cast<const unsigned*>(x+(first+8+lane/4)*128+k);
            a3=*reinterpret_cast<const unsigned*>(x+(first+8+lane/4)*128+k+8);
        }
        #pragma unroll
        for(int atom=0;atom<Atoms;++atom){
            int n=col+atom*8+lane/4;
            unsigned b0=0,b1=0;
            if(n<v){
                int base=(group*((v+7)/8)+(col+atom*8)/8)*128+lane*2;
                b0=*reinterpret_cast<const unsigned*>(weight+base);
                b1=*reinterpret_cast<const unsigned*>(weight+base+64);
            }
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(acc[atom][0]),"+f"(acc[atom][1]),"+f"(acc[atom][2]),"+f"(acc[atom][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    #pragma unroll
    for(int atom=0;atom<Atoms;++atom){
        int n=warp*Atoms*8+atom*8+(lane%4)*2;
        *reinterpret_cast<half2*>(stores+(lane/4)*(N+8)+n)=__floats2half2_rn(acc[atom][0],acc[atom][1]);
        if constexpr(M==16) *reinterpret_cast<half2*>(stores+(8+lane/4)*(N+8)+n)=__floats2half2_rn(acc[atom][2],acc[atom][3]);
    }
    __syncthreads();
    for(int index=threadIdx.x*8;index<M*N;index+=blockDim.x*8){
        int row=index/N, n=blockIdx.y*N+index%N;
        int destination=(first+row)*v+n;
        half* source=stores+row*(N+8)+index%N;
        if(n+8<=v && destination%8==0){
            *reinterpret_cast<uint4*>(out+destination)=*reinterpret_cast<const uint4*>(source);
        }else{
            #pragma unroll
            for(int j=0;j<8;++j)if(n+j<v)out[destination+j]=source[j];
        }
    }
    if constexpr(Count){if(threadIdx.x==0){
        int i=(blockIdx.x*gridDim.y+blockIdx.y)*2;
        stats[i]=static_cast<long long>(issued)*4*Atoms;
        stats[i+1]=static_cast<long long>(bypassed)*4*Atoms;
    }}
}

void prepare(torch::Tensor x,torch::Tensor packed,torch::Tensor support,double threshold){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.dim()==2 && x.size(1)==128
                && x.size(0)>0 && x.size(0)%16==0 && x.is_contiguous(),"FP32 M16/D128 input required");
    TORCH_CHECK(packed.device()==x.device() && packed.scalar_type()==at::kHalf && packed.sizes()==x.sizes()
                && packed.is_contiguous(),"FP16 preparation shape/layout mismatch");
    TORCH_CHECK(support.device()==x.device() && support.scalar_type()==at::kInt && support.is_contiguous()
                && support.dim()==1 && support.numel()==x.size(0)/16,"Support shape/type mismatch");
    TORCH_CHECK(std::isfinite(threshold) && threshold>=0,"Finite nonnegative threshold required");
    c10::cuda::CUDAGuard guard(x.device());
    prepare_kernel<<<x.size(0)/16,128,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),
        reinterpret_cast<half*>(packed.data_ptr()),support.data_ptr<int>(),x.size(0),static_cast<float>(threshold));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int M,int N>
void dispatch(torch::Tensor x,torch::Tensor support,torch::Tensor w,torch::Tensor out,
              torch::Tensor stats,bool skip,bool count){
    dim3 grid(x.size(0)/M,(out.size(1)+N-1)/N);
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(S,C) project_kernel<M,N,S,C><<<grid,128,0,stream>>>(reinterpret_cast<half*>(x.data_ptr()),support.data_ptr<int>(),reinterpret_cast<half*>(w.data_ptr()),reinterpret_cast<half*>(out.data_ptr()),reinterpret_cast<long long*>(stats.data_ptr()),out.size(1))
    if(skip){if(count){CALL(true,true);}else{CALL(true,false);}}
    else{if(count){CALL(false,true);}else{CALL(false,false);}}
    #undef CALL
}

void project(torch::Tensor x,torch::Tensor support,torch::Tensor w,torch::Tensor out,
             torch::Tensor stats,bool skip,bool count,int m,int n){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf && x.dim()==2 && x.size(1)==128,"CUDA FP16 D128 required");
    TORCH_CHECK(m==16 && (n==128 || n==256 || n==512) && x.size(0)>0 && x.size(0)%m==0,"Unsupported tile/alignment");
    for(const auto& t:{x,w,out}) TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kHalf && t.is_contiguous(),"Projection type/layout mismatch");
    TORCH_CHECK(out.dim()==2 && out.size(0)==x.size(0) && out.size(1)>0 && w.sizes()==at::IntArrayRef({8,(out.size(1)+7)/8,128}),"Packed weight/output shape mismatch");
    TORCH_CHECK(support.device()==x.device() && support.scalar_type()==at::kInt && support.is_contiguous() && support.dim()==1 && support.numel()==x.size(0)/16,"Support type/layout mismatch");
    TORCH_CHECK(stats.device()==x.device() && stats.scalar_type()==at::kLong && stats.is_contiguous()
                && stats.sizes()==at::IntArrayRef({x.size(0)/m,(out.size(1)+n-1)/n,2}),"Counter shape mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    #define DISPATCH(M,N) dispatch<M,N>(x,support,w,out,stats,skip,count)
    if(n==128){DISPATCH(16,128);}else if(n==256){DISPATCH(16,256);}else{DISPATCH(16,512);}
    #undef DISPATCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,module){module.def("prepare",&prepare);module.def("project",&project);}
