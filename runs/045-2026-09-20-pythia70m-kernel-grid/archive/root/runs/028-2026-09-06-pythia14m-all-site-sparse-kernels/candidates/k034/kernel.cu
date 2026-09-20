// K034: shared-memory 32x64x64 tiles, four BF16 MMA warps.
// All projections preserve K16 forward arithmetic; zero16x16 A fragments skip.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

__device__ __forceinline__ unsigned gate_pair(unsigned v,float threshold,bool gate){
    if(!gate)return v;
    bf16 a=__ushort_as_bfloat16(v&65535),b=__ushort_as_bfloat16(v>>16);
    if(__bfloat162float(a)<threshold)a=__float2bfloat16_rn(0.f);
    if(__bfloat162float(b)<threshold)b=__float2bfloat16_rn(0.f);
    return unsigned(__bfloat16_as_ushort(a)) | (unsigned(__bfloat16_as_ushort(b))<<16);
}
__device__ __forceinline__ int swizzle(int row,int word){return row*32+(word^((row&7)*4));}

template<int K,bool Skip>
__device__ void accumulate(const bf16* x,const bf16* w,int row0,int col0,int tid,
 float threshold,bool gate,unsigned* sx,unsigned* sw,float (&c)[4][4]){
    int lane=tid&31,warp=tid/32,row=(warp/2)*16+lane/4;
    #pragma unroll
    for(int base=0;base<K;base+=64){
        for(int i=tid;i<32*32;i+=128){
            int r=i/32,word=i%32;
            unsigned value=*reinterpret_cast<const unsigned*>(x+(row0+r)*K+base+word*2);
            sx[swizzle(r,word)]=gate_pair(value,threshold,gate);
        }
        for(int i=tid;i<64*32;i+=128){
            int r=i/32,word=i%32;
            sw[swizzle(r,word)]=*reinterpret_cast<const unsigned*>(w+(col0+r)*K+base+word*2);
        }
        __syncthreads();
        #pragma unroll
        for(int kk=0;kk<4;++kk){
            int word=kk*8+lane%4;
            unsigned a0=sx[swizzle(row,word)],a1=sx[swizzle(row+8,word)];
            unsigned a2=sx[swizzle(row,word+4)],a3=sx[swizzle(row+8,word+4)];
            bool zero=Skip && __all_sync(0xffffffffu,((a0|a1|a2|a3)&0x7fff7fffu)==0);
            if(!zero){
                #pragma unroll
                for(int nn=0;nn<4;++nn){
                    int wr=(warp%2)*32+nn*8+lane/4;
                    unsigned b0=sw[swizzle(wr,word)],b1=sw[swizzle(wr,word+4)];
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(c[nn][0]),"+f"(c[nn][1]),"+f"(c[nn][2]),"+f"(c[nn][3])
                        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
                }
            }
        }
        __syncthreads();
    }
}

template<bool Skip>
__global__ void project(const bf16* x,const bf16* w,const bf16* bias,bf16* out,int n){
    __shared__ unsigned sx[32*32],sw[64*32];
    float c[4][4]={};int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    int row0=blockIdx.x*32,col0=blockIdx.y*64;
    accumulate<128,Skip>(x,w,row0,col0,tid,0.f,false,sx,sw,c);
    #pragma unroll
    for(int nn=0;nn<4;++nn){
        #pragma unroll
        for(int e=0;e<4;++e){
            int r=row0+(warp/2)*16+lane/4+(e/2)*8;
            int col=col0+(warp%2)*32+nn*8+(lane%4)*2+e%2;
            out[r*n+col]=__float2bfloat16_rn(c[nn][e]+__bfloat162float(bias[col]));
        }
    }
}

template<bool Skip>
__global__ void joint(const bf16* h,const bf16* z,const bf16* wh,const bf16* wz,
 const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,float th,float tz,bool gh,bool gz){
    __shared__ unsigned sx[32*32],sw[64*32];
    int tid=threadIdx.x,lane=tid&31,warp=tid/32,row0=blockIdx.x*32,col0=blockIdx.y*64;
    float ah[4][4]={},az[4][4]={};
    accumulate<512,Skip>(h,wh,row0,col0,tid,th,gh,sx,sw,ah);
    accumulate<128,Skip>(z,wz,row0,col0,tid,tz,gz,sx,sw,az);
    #pragma unroll
    for(int nn=0;nn<4;++nn){
        #pragma unroll
        for(int e=0;e<4;++e){
            int r=row0+(warp/2)*16+lane/4+(e/2)*8;
            int c=col0+(warp%2)*32+nn*8+(lane%4)*2+e%2;
            float hv=__bfloat162float(__float2bfloat16_rn(ah[nn][e]+__bfloat162float(bh[c])));
            float zv=__bfloat162float(__float2bfloat16_rn(az[nn][e]+__bfloat162float(bz[c])));
            float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
            out[r*128+c]=__float2bfloat16_rn(sum+__bfloat162float(residual[r*128+c]));
        }
    }
}

void projection(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor y,bool skip){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128");
    for(const auto& t:{x,w,b,y})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"Type/layout");
    int m=x.size(0),n=w.size(0);
    TORCH_CHECK(m>0 && m%32==0 && w.dim()==2 && w.size(1)==128 && (n==384 || n==512),"Projection shapes");
    TORCH_CHECK(b.numel()==n && y.sizes()==at::IntArrayRef({m,n}),"Output/bias");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    if(skip)project<true><<<dim3(m/32,n/64),128,0,stream>>>(P(x),P(w),P(b),P(y),n);
    else project<false><<<dim3(m/32,n/64),128,0,stream>>>(P(x),P(w),P(b),P(y),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void output(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,
 torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,double th,double tz,bool gh,bool gz,bool skip){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 required");
    for(const auto& t:{h,z,wh,wz,bh,bz,residual,out})TORCH_CHECK(t.device()==h.device() && t.scalar_type()==h.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
    int m=h.size(0);
    TORCH_CHECK(m>0 && m%32==0 && h.size(1)==512 && z.sizes()==at::IntArrayRef({m,128}),"M multiple32, H512 Z128");
    TORCH_CHECK(wh.sizes()==at::IntArrayRef({128,512}) && wz.sizes()==at::IntArrayRef({128,128}),"Weight shapes");
    TORCH_CHECK(bh.numel()==128 && bz.numel()==128 && residual.sizes()==z.sizes() && out.sizes()==z.sizes(),"Outputs");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"Invalid gate");
    c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();
    if(skip)joint<true><<<dim3(m/32,2),128,0,stream>>>(P(h),P(z),P(wh),P(wz),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz);
    else joint<false><<<dim3(m/32,2),128,0,stream>>>(P(h),P(z),P(wh),P(wz),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("projection",&projection);m.def("output",&output);}
