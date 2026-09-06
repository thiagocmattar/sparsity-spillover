// K023: sparse attention with unnormalized probability reduction.
// BF16 probability mode rounds exp(score-max), then divides after PV.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>
using bf16=__nv_bfloat16;

__global__ void prepare(const bf16* k,const bf16* v,bf16* kt,bf16* values,
                       int* indices,int* counts,float* prefix,int t){
    int hd=blockIdx.x,head=hd/32,dim=hd%32,lane=threadIdx.x&31,warp=threadIdx.x/32;
    using Scan=cub::BlockScan<float,256>;
    __shared__ typename Scan::TempStorage scratch;
    __shared__ int warp_counts[8],offset;
    __shared__ float carry;
    if(threadIdx.x==0){offset=0;carry=0.f;}
    __syncthreads();
    for(int base=0;base<t;base+=256){
        int row=base+threadIdx.x;
        float value=row<t?__bfloat162float(v[(head*t+row)*32+dim]):0.f;
        if(row<t)kt[hd*t+row]=k[(head*t+row)*32+dim];
        unsigned active=__ballot_sync(0xffffffffu,row<t && value!=0.f);
        if(lane==0)warp_counts[warp]=__popc(active);
        __syncthreads();
        int before=offset;
        for(int w=0;w<warp;++w)before+=warp_counts[w];
        int rank=__popc(active & ((1u<<lane)-1u));
        if(row<t && value!=0.f){indices[hd*t+before+rank]=row;values[hd*t+before+rank]=__float2bfloat16_rn(value);}
        float inclusive;Scan(scratch).InclusiveSum(value,inclusive);
        float old=carry;
        if(row<t)prefix[hd*t+row]=old+inclusive;
        __syncthreads();
        if(threadIdx.x==255)carry=old+inclusive;
        if(threadIdx.x==0)for(int w=0;w<8;++w)offset+=warp_counts[w];
        __syncthreads();
    }
    if(threadIdx.x==0)counts[hd]=offset;
}

__device__ float sum_warp(float x){for(int s=16;s;s/=2)x+=__shfl_xor_sync(0xffffffffu,x,s);return x;}
__device__ float max_warp(float x){for(int s=16;s;s/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffffu,x,s));return x;}

template<bool Prefix,bool RoundP,bool Count>
__global__ void attention(const bf16* q,const bf16* kt,const bf16* values,const int* indices,
                         const int* counts,const float* prefix,bf16* out,int64_t* stats,int t,float scale){
    int lane=threadIdx.x&31,warp=threadIdx.x/32,row=blockIdx.x,head=blockIdx.y;
    float qv=__bfloat162float(q[(head*t+row)*32+lane]);
    unsigned qm=__ballot_sync(0xffffffffu,qv!=0.f);
    if(Prefix && qm==0){
        if(warp==0){
            float factor=1.f/float(row+1);
            out[(head*t+row)*32+lane]=__float2bfloat16_rn(prefix[(head*32+lane)*t+row]*factor);
            if(Count && lane==0){stats[(head*t+row)*3]=0;stats[(head*t+row)*3+1]=0;stats[(head*t+row)*3+2]=1;}
        }
        return;
    }
    float scores[8]={};int qwork=0,pwork=0;
    unsigned selected=qm;
    while(selected){
        int dim=__ffs(selected)-1;float x=__shfl_sync(0xffffffffu,qv,dim);
        #pragma unroll
        for(int j=0;j<8;++j){int key=j*256+threadIdx.x;
            if(key<=row){float y=__bfloat162float(kt[(head*32+dim)*t+key]);
                if(y!=0.f){scores[j]=fmaf(x,y,scores[j]);if(Count)++qwork;}}
        }
        selected&=selected-1;
    }
    __shared__ float reduction[256],probability[2048],denominator,maximum;
    float localmax=-INFINITY;
    #pragma unroll
    for(int j=0;j<8;++j)if(j*256+threadIdx.x<=row)localmax=fmaxf(localmax,scores[j]*scale);
    localmax=max_warp(localmax);
    if(lane==0)reduction[warp]=localmax;
    __syncthreads();
    if(warp==0){float m=max_warp(lane<8?reduction[lane]:-INFINITY);if(lane==0)maximum=m;}
    __syncthreads();
    float localsum=0.f;
    #pragma unroll
    for(int j=0;j<8;++j){scores[j]=j*256+threadIdx.x<=row?__expf(scores[j]*scale-maximum):0.f;localsum+=scores[j];}
    localsum=sum_warp(localsum);
    if(lane==0)reduction[warp]=localsum;
    __syncthreads();
    if(warp==0){float s=sum_warp(lane<8?reduction[lane]:0.f);if(lane==0)denominator=s;}
    __syncthreads();
    #pragma unroll
    for(int j=0;j<8;++j)if(j*256+threadIdx.x<=row){
        float p=scores[j];
        if(RoundP)p=__bfloat162float(__float2bfloat16_rn(p));
        probability[j*256+threadIdx.x]=p;
    }
    __syncthreads();
    int hd=head*32+lane;float result=0.f;
    for(int j=warp;j<counts[hd];j+=8){
        int key=indices[hd*t+j];if(key>row)break;
        float p=probability[key];
        if(p!=0.f){result=fmaf(p,__bfloat162float(values[hd*t+j]),result);if(Count)++pwork;}
    }
    reduction[threadIdx.x]=result;
    __syncthreads();
    if(warp==0){
        float a=(reduction[lane]+reduction[32+lane])+(reduction[64+lane]+reduction[96+lane]);
        float b=(reduction[128+lane]+reduction[160+lane])+(reduction[192+lane]+reduction[224+lane]);
        out[(head*t+row)*32+lane]=__float2bfloat16_rn((a+b)/denominator);
    }
    if(Count){
        __syncthreads();float qs=sum_warp(float(qwork)),ps=sum_warp(float(pwork));
        if(lane==0){reduction[warp]=qs;reduction[8+warp]=ps;}
        __syncthreads();
        if(warp==0){float qs2=sum_warp(lane<8?reduction[lane]:0.f),ps2=sum_warp(lane<8?reduction[8+lane]:0.f);
            if(lane==0){stats[(head*t+row)*3]=int64_t(qs2);stats[(head*t+row)*3+1]=int64_t(ps2);stats[(head*t+row)*3+2]=0;}}
    }
}

void forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor kt,
             torch::Tensor values,torch::Tensor indices,torch::Tensor counts,
             torch::Tensor prefix,torch::Tensor out,torch::Tensor stats,
             double scale,bool shortcut,bool round_p,bool count){
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4,"CUDA BF16 BH TD required");
    TORCH_CHECK(q.size(0)==1 && q.size(3)==32 && q.size(2)>0 && q.size(2)<=2048,"B1,D32,T<=2048 only");
    TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.sizes()==out.sizes(),"QKV/output mismatch");
    for(const auto& x:{q,k,v,kt,values,out})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"BF16 tensor layout/type mismatch");
    for(const auto& x:{indices,counts})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==at::kInt && x.is_contiguous(),"integer workspace mismatch");
    TORCH_CHECK(prefix.device()==q.device() && prefix.scalar_type()==at::kFloat && prefix.is_contiguous(),"prefix mismatch");
    TORCH_CHECK(stats.device()==q.device() && stats.scalar_type()==at::kLong && stats.is_contiguous(),"stats mismatch");
    int t=q.size(2),h=q.size(1);
    TORCH_CHECK(kt.numel()==q.numel() && values.numel()==q.numel() && indices.numel()==q.numel() && prefix.numel()==q.numel() && counts.numel()==h*32 && stats.numel()==h*t*3,"workspace capacity mismatch");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"invalid scaling");
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define B(tensor) reinterpret_cast<bf16*>(tensor.data_ptr())
    prepare<<<h*32,256,0,stream>>>(B(k),B(v),B(kt),B(values),indices.data_ptr<int>(),counts.data_ptr<int>(),prefix.data_ptr<float>(),t);
    #define LAUNCH(P,R,C) attention<P,R,C><<<dim3(t,h),256,0,stream>>>(B(q),B(kt),B(values),indices.data_ptr<int>(),counts.data_ptr<int>(),prefix.data_ptr<float>(),B(out),stats.data_ptr<int64_t>(),t,float(scale))
    if(count){if(shortcut){if(round_p){LAUNCH(true,true,true);}else{LAUNCH(true,false,true);}}else{if(round_p){LAUNCH(false,true,true);}else{LAUNCH(false,false,true);}}}
    else{if(shortcut){if(round_p){LAUNCH(true,true,false);}else{LAUNCH(true,false,false);}}else{if(round_p){LAUNCH(false,true,false);}else{LAUNCH(false,false,false);}}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}


