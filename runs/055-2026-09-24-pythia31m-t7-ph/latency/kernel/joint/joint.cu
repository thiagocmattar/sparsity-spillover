// Run042 opt001: M8 N256.
// Run040 opt002: N256, eight compute warps; same M8 arithmetic and gate policy.
// Exact short-row SIMT plus native-order MMA for the remaining h/z rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

struct TinyRow { int count=9,indices[8]={}; float values[8]={}; unsigned tiles[4]={}; };
__device__ __forceinline__ float gated(const bf16* x,int index,float threshold,bool gate){
    float value=__bfloat162float(x[index]);return gate && value<threshold?0.f:value;
}
template<int K>
__device__ __forceinline__ TinyRow inspect(const bf16* x,int row,int lane,float threshold,bool gate){
    TinyRow out;out.count=0;
    #pragma unroll
    for(int base=0;base<K;base+=32){
        float value=gated(x,row*K+base+lane,threshold,gate);
        unsigned mask=__ballot_sync(0xffffffffu,value!=0.f);
        if(mask&0xffffu)out.tiles[base/512]|=1u<<((base/16)%32);
        if(mask&0xffff0000u)out.tiles[base/512]|=1u<<((base/16+1)%32);
        if(out.count+__popc(mask)>8)out.count=9;
        else while(mask){
            int selected=__ffs(mask)-1;float next=__shfl_sync(0xffffffffu,value,selected);
            if(!(fabsf(next)>=0x1p-50f && fabsf(next)<=0x1p50f)){out.count=9;break;}
            out.indices[out.count]=base+selected;out.values[out.count]=next;
            ++out.count;mask&=mask-1;
        }
    }
    return out;
}
__device__ __forceinline__ bf16 short_linear(TinyRow a,const bf16* wt,const bf16* bias,int col){
    float value=0.f;
    if(a.count>0)value=__fmul_rn(a.values[0],__bfloat162float(wt[a.indices[0]*256+col]));
    #pragma unroll
    for(int i=1;i<8;++i)if(i<a.count)value=__fmaf_rn(a.values[i],__bfloat162float(wt[a.indices[i]*256+col]),value);
    return __float2bfloat16_rn(__fadd_rn(value,__bfloat162float(bias[col])));
}
__device__ __forceinline__ unsigned gate_pair(unsigned value,float threshold,bool gate){
    if(!gate)return value;
    bf16 a=__ushort_as_bfloat16(value&65535),b=__ushort_as_bfloat16(value>>16);
    if(__bfloat162float(a)<threshold)a=__float2bfloat16_rn(0.f);
    if(__bfloat162float(b)<threshold)b=__float2bfloat16_rn(0.f);
    return unsigned(__bfloat16_as_ushort(a))|(unsigned(__bfloat16_as_ushort(b))<<16);
}
template<int K,bool Skip,bool Count>
__device__ __forceinline__ void accumulate(const bf16* x,const bf16* w,int row,int col,int lane,
    float threshold,bool gate,bool hybrid,const bool* complex,int local_row,const unsigned* active_tiles,float (&acc)[4][4],int& issued,int& bypassed){
    for(int word=0;word<(K+511)/512;++word){
    const int tiles=min(32,(K-word*512)/16);
    const unsigned valid=tiles==32?0xffffffffu:((1u<<tiles)-1u);
    unsigned pending=(Skip?active_tiles[word]:valid)&valid;
    if constexpr(Count)bypassed+=(tiles-__popc(pending))*4;
    while(pending){
        int base=word*512+(__ffs(pending)-1)*16;pending&=pending-1;
        int k=base+(lane%4)*2;
        bool low=!hybrid || complex[local_row],high=false;
        unsigned a0=low?gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k),threshold,gate):0;
        unsigned a1=high?gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k),threshold,gate):0;
        unsigned a2=low?gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k+8),threshold,gate):0;
        unsigned a3=high?gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k+8),threshold,gate):0;
        if constexpr(Skip){
            // Unsafe-weight fast-path disable retains the original zero-A test.
            if(!hybrid && __all_sync(0xffffffffu,((a0|a1|a2|a3)&0x7fff7fffu)==0)){
                if constexpr(Count)bypassed+=4;
                continue;
            }
        }
        {
            #pragma unroll
            for(int atom=0;atom<4;++atom){
                unsigned b0=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k);
                unsigned b1=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k+8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[atom][0]),"+f"(acc[atom][1]),"+f"(acc[atom][2]),"+f"(acc[atom][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            if constexpr(Count)issued+=4;
        }
    }
    }
}
template<bool Count>
__device__ __forceinline__ void counters(long long* stats,int warp,int lane,int hi,int hb,int zi,int zb,int hs,int zs){
    if constexpr(Count){if(lane==0){int slot=((blockIdx.x+blockIdx.y)*8+warp)*6;
        stats[slot]=hi;stats[slot+1]=hb;stats[slot+2]=zi;stats[slot+3]=zb;stats[slot+4]=hs;stats[slot+5]=zs;}}
}

template<int K,int Threads>
__device__ __forceinline__ TinyRow parallel_inspect(const bf16* x,int row,float threshold,bool gate,
    int* counts,int* indices,int* cursor,TinyRow* summary){
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,count=0;
    #pragma unroll
    for(int base=0;base<K;base+=Threads){
        float value=gated(x,row*K+base+tid,threshold,gate);
        if(value!=0.f)count+=(!(fabsf(value)>=0x1p-50f && fabsf(value)<=0x1p50f))?9:1;
    }
    #pragma unroll
    for(int offset=16;offset>0;offset/=2)count+=__shfl_down_sync(0xffffffffu,count,offset);
    if(lane==0)counts[warp]=count;
    __syncthreads();
    if(tid==0){
        int total=0;
        #pragma unroll
        for(int w=0;w<Threads/32;++w)total+=counts[w];
        summary->count=min(total,9);*cursor=0;
    }
    __syncthreads();
    if(summary->count<=8){
        #pragma unroll
        for(int base=0;base<K;base+=Threads){
            int index=base+tid;
            float value=gated(x,row*K+index,threshold,gate);
            if(value!=0.f)indices[atomicAdd(cursor,1)]=index;
        }
        __syncthreads();
        if(tid==0){
            for(int i=1;i<summary->count;++i){
                int value=indices[i],j=i-1;
                while(j>=0 && indices[j]>value){indices[j+1]=indices[j];--j;}
                indices[j+1]=value;
            }
            for(int i=0;i<summary->count;++i){
                summary->indices[i]=indices[i];
                summary->values[i]=gated(x,row*K+indices[i],threshold,gate);
            }
        }
    }
    __syncthreads();
    TinyRow out=*summary;
    __syncthreads();
    return out;
}

template<int Threads>
__global__ void parallel_rows(const bf16* h,const bf16* z,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,
    float th,float tz,bool gh,bool gz,int* row_counts){
    __shared__ int counts[Threads/32],indices[8],cursor;
    __shared__ TinyRow summary;
    int row=blockIdx.x;
    TinyRow ha=parallel_inspect<1024,Threads>(h,row,th,gh,counts,indices,&cursor,&summary);
    TinyRow za=parallel_inspect<256,Threads>(z,row,tz,gz,counts,indices,&cursor,&summary);
    if(threadIdx.x==0){row_counts[row*2]=ha.count;row_counts[row*2+1]=za.count;}
    if(ha.count>8 || za.count>8)return;
    for(int col=threadIdx.x;col<256;col+=Threads){
        bf16 hv=short_linear(ha,wht,bh,col),zv=short_linear(za,wzt,bz,col);
        float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
        out[row*256+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*256+col]));
    }
}

template<bool Skip,bool Count>
__global__ void joint(const bf16* h,const bf16* z,const bf16* wh,const bf16* wz,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,long long* stats,float th,float tz,bool gh,bool gz,bool fast_weights,const int* row_counts){
    __shared__ bool hc[8],zc[8];
    __shared__ bf16 hs[8*256],zs[8*256];
    __shared__ unsigned h_masks[8][4],z_masks[8][4];
    int lane=threadIdx.x&31,warp=threadIdx.x/32,first_row=blockIdx.x*8;
    if(row_counts){
        bool done=true;
        #pragma unroll
        for(int i=0;i<8;++i)done=done && row_counts[(first_row+i)*2]<=8 && row_counts[(first_row+i)*2+1]<=8;
        if(done){
            counters<Count>(stats,warp,lane,0,256,0,64,
                row_counts[(first_row+warp)*2]*256,row_counts[(first_row+warp)*2+1]*256);
            return;
        }
    }

    int h_scalar=0,z_scalar=0;bool hybrid=Skip && fast_weights;
    unsigned h_tiles[4]={~0u,~0u,0,0},z_tiles[4]={0xffffu,0,0,0};
    if(hybrid){
        for(int local=warp;local<8;local+=8){
            int row=first_row+local;auto ha=inspect<1024>(h,row,lane,th,gh);auto za=inspect<256>(z,row,lane,tz,gz);
            bool h_complex=ha.count>8,z_complex=za.count>8;
            if(lane==0){hc[local]=h_complex;zc[local]=z_complex;}
            if(lane==0)for(int w=0;w<4;++w){h_masks[local][w]=h_complex?ha.tiles[w]:0;z_masks[local][w]=z_complex?za.tiles[w]:0;}
            if constexpr(Count){if(!h_complex)h_scalar+=ha.count*256;if(!z_complex)z_scalar+=za.count*256;}
            #pragma unroll
            for(int e=0;e<8;++e){
                int col=blockIdx.y*256+e*32+lane,slot=local*256+e*32+lane;bf16 hv,zv;
                if(!h_complex){hv=short_linear(ha,wht,bh,col);hs[slot]=hv;}
                if(!z_complex){zv=short_linear(za,wzt,bz,col);zs[slot]=zv;}
                if(!h_complex && !z_complex){
                    float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
                    out[row*256+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*256+col]));
                }
            }
        }
        __syncthreads();
        unsigned any=0;for(int w=0;w<4;++w){h_tiles[w]=z_tiles[w]=0;for(int i=0;i<8;++i){h_tiles[w]|=h_masks[i][w];z_tiles[w]|=z_masks[i][w];}any|=h_tiles[w]|z_tiles[w];}
        if(any==0){counters<Count>(stats,warp,lane,0,warp<8?256:0,0,warp<8?64:0,h_scalar,z_scalar);return;}
    }
    if(warp>=8){counters<Count>(stats,warp,lane,0,0,0,0,h_scalar,z_scalar);return;}
    int local_row=lane/4,row=first_row+local_row,col=blockIdx.y*256+warp*32;
    float ah[4][4]={},az[4][4]={};int hi=0,hb=0,zi=0,zb=0;
    accumulate<1024,Skip,Count>(h,wh,row,col,lane,th,gh,hybrid,hc,local_row,h_tiles,ah,hi,hb);
    accumulate<256,Skip,Count>(z,wz,row,col,lane,tz,gz,hybrid,zc,local_row,z_tiles,az,zi,zb);
    #pragma unroll
    for(int atom=0;atom<4;++atom){
        #pragma unroll
        for(int e=0;e<2;++e){
            int r=row+(e/2)*8,lr=local_row+(e/2)*8,c=col+atom*8+(lane%4)*2+e%2,slot=lr*256+c%256;
            if(!hybrid || hc[lr] || zc[lr]){
                float hv=hybrid && !hc[lr]?__bfloat162float(hs[slot]):__bfloat162float(__float2bfloat16_rn(ah[atom][e]+__bfloat162float(bh[c])));
                float zv=hybrid && !zc[lr]?__bfloat162float(zs[slot]):__bfloat162float(__float2bfloat16_rn(az[atom][e]+__bfloat162float(bz[c])));
                float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
                out[r*256+c]=__float2bfloat16_rn(sum+__bfloat162float(residual[r*256+c]));
            }
        }
    }
    // The parallel prepass also computes short rows in a mixed fallback group.
    // Those outputs are recomputed below the fallback path; count both executions.
    if constexpr(Count){
        if(row_counts && row_counts[(first_row+warp)*2]<=8 && row_counts[(first_row+warp)*2+1]<=8){
            h_scalar+=row_counts[(first_row+warp)*2]*256;
            z_scalar+=row_counts[(first_row+warp)*2+1]*256;
        }
    }
    counters<Count>(stats,warp,lane,hi,hb,zi,zb,h_scalar,z_scalar);
}
void forward(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,torch::Tensor wht,torch::Tensor wzt,
    torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,torch::Tensor stats,
    double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count,torch::Tensor row_counts){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 required");
    for(const auto& tensor:{h,z,wh,wz,wht,wzt,bh,bz,residual,out})TORCH_CHECK(tensor.device()==h.device() && tensor.scalar_type()==h.scalar_type() && tensor.is_contiguous(),"Type/layout mismatch");
    int m=h.size(0);
    TORCH_CHECK(m>0 && m%8==0 && h.size(1)==1024 && z.sizes()==at::IntArrayRef({m,256}),"M8 H1024 Z256 required");
    TORCH_CHECK(wh.sizes()==at::IntArrayRef({256,1024}) && wz.sizes()==at::IntArrayRef({256,256}),"Weight shape");
    TORCH_CHECK(wht.sizes()==at::IntArrayRef({1024,256}) && wzt.sizes()==at::IntArrayRef({256,256}),"Transposed weight shape");
    TORCH_CHECK(bh.numel()==256 && bz.numel()==256 && residual.sizes()==z.sizes() && out.sizes()==z.sizes(),"Output shape");
    TORCH_CHECK(stats.device()==h.device() && stats.scalar_type()==at::kLong && stats.is_contiguous() && stats.numel()==(m/8)*1*8*6,"Counter shape");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"Gate threshold");
    TORCH_CHECK(row_counts.device()==h.device() && row_counts.scalar_type()==at::kInt && row_counts.is_contiguous() && row_counts.numel()==m*2,"Row count shape");
    c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    #define LAUNCH(S,C) do { \
      int* rows=(S && fast_weights)?row_counts.data_ptr<int>():nullptr; \
      if(rows)parallel_rows<128><<<m,128,0,stream>>>(P(h),P(z),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz,rows); \
      joint<S,C><<<dim3(m/8,1),256,0,stream>>>(P(h),P(z),P(wh),P(wz),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,gh,gz,fast_weights,rows); \
    } while(0)
    if(skip){if(count){LAUNCH(true,true);}else{LAUNCH(true,false);}}
    else{if(count){LAUNCH(false,true);}else{LAUNCH(false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
