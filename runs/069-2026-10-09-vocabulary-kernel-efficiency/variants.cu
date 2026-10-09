// The source gate is unchanged. G=2 retains two independent M16/K16 masks.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

template<int G,int A> struct Fragment { unsigned x[G][4], w[A][2]; };

template<int G,int A,bool Skip>
__device__ __forceinline__ void operands(const half* x,const half* w,int first,int col,int lane,
    int group,int v,const unsigned (&live)[G],Fragment<G,A>& f){
    int k=group*16+(lane%4)*2;
    #pragma unroll
    for(int g=0;g<G;++g){
        bool active=G==1 || !Skip || (live[g]&(1u<<group));
        int row=first+g*16+lane/4;
        f.x[g][0]=active?*reinterpret_cast<const unsigned*>(x+row*128+k):0;
        f.x[g][2]=active?*reinterpret_cast<const unsigned*>(x+row*128+k+8):0;
        f.x[g][1]=active?*reinterpret_cast<const unsigned*>(x+(row+8)*128+k):0;
        f.x[g][3]=active?*reinterpret_cast<const unsigned*>(x+(row+8)*128+k+8):0;
    }
    #pragma unroll
    for(int a=0;a<A;++a){
        int n=col+a*8+lane/4, base=(group*((v+7)/8)+(col+a*8)/8)*128+lane*2;
        f.w[a][0]=n<v?*reinterpret_cast<const unsigned*>(w+base):0;
        f.w[a][1]=n<v?*reinterpret_cast<const unsigned*>(w+base+64):0;
    }
}

__device__ __forceinline__ void mma(float (&acc)[4], const unsigned (&x)[4],unsigned b0,unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3])
        : "r"(x[0]),"r"(x[1]),"r"(x[2]),"r"(x[3]),"r"(b0),"r"(b1));
}

template<int N,int G,bool Skip,bool Direct,bool Prefetch,bool Count>
__global__ void projection(const half* x,const int* support,const half* w,half* out,long long* stats,int v){
    constexpr int A=N/32;
    int lane=threadIdx.x&31, warp=threadIdx.x/32;
    int first=blockIdx.x*(16*G), col=blockIdx.y*N+warp*A*8;
    unsigned live[G], pending=0;
    #pragma unroll
    for(int g=0;g<G;++g){live[g]=support[first/16+g];pending|=Skip?live[g]:0xffu;}
    float acc[G][A][4]={};
    int issued[G]={}; // Untimed Count specialization records executed MMA paths.
    // Shared-memory allocation is compiled out in the direct-store family.
    __shared__ __align__(16) half stores[Direct?1:G*16*(N+8)];
    if constexpr(Prefetch){
        Fragment<G,A> current,next;
        int group=-1;
        if(pending){group=__ffs(pending)-1;pending&=pending-1;operands<G,A,Skip>(x,w,first,col,lane,group,v,live,current);}
        while(group>=0){
            int upcoming=-1;
            if(pending){upcoming=__ffs(pending)-1;pending&=pending-1;operands<G,A,Skip>(x,w,first,col,lane,upcoming,v,live,next);}
            #pragma unroll
            for(int a=0;a<A;++a){
                #pragma unroll
                for(int g=0;g<G;++g)if(G==1 || !Skip || (live[g]&(1u<<group))){
                    mma(acc[g][a],current.x[g],current.w[a][0],current.w[a][1]);
                    if constexpr(Count)++issued[g];
                }
            }
            if(upcoming>=0)current=next;
            group=upcoming;
        }
    }else{
        // G=1 mirrors the original load/MMA order: direct changes only stores.
        while(pending){
            int group=__ffs(pending)-1;pending&=pending-1;
            int k=group*16+(lane%4)*2;
            unsigned operand[G][4];
            #pragma unroll
            for(int g=0;g<G;++g){
                bool active=G==1 || !Skip || (live[g]&(1u<<group));
                int row=first+g*16+lane/4;
                operand[g][0]=active?*reinterpret_cast<const unsigned*>(x+row*128+k):0;
                operand[g][2]=active?*reinterpret_cast<const unsigned*>(x+row*128+k+8):0;
                operand[g][1]=active?*reinterpret_cast<const unsigned*>(x+(row+8)*128+k):0;
                operand[g][3]=active?*reinterpret_cast<const unsigned*>(x+(row+8)*128+k+8):0;
            }
            #pragma unroll
            for(int a=0;a<A;++a){
                int n=col+a*8+lane/4, base=(group*((v+7)/8)+(col+a*8)/8)*128+lane*2;
                unsigned b0=n<v?*reinterpret_cast<const unsigned*>(w+base):0;
                unsigned b1=n<v?*reinterpret_cast<const unsigned*>(w+base+64):0;
                #pragma unroll
                for(int g=0;g<G;++g)if(G==1 || !Skip || (live[g]&(1u<<group))){
                    mma(acc[g][a],operand[g],b0,b1);
                    if constexpr(Count)++issued[g];
                }
            }
        }
    }
    #pragma unroll
    for(int g=0;g<G;++g){
        #pragma unroll
        for(int a=0;a<A;++a){
            int n=warp*A*8+a*8+(lane%4)*2;
            if constexpr(Direct){
                int c=blockIdx.y*N+n, row=first+g*16+lane/4;
                if(c+1<v){
                    *reinterpret_cast<half2*>(out+row*v+c)=__floats2half2_rn(acc[g][a][0],acc[g][a][1]);
                    *reinterpret_cast<half2*>(out+(row+8)*v+c)=__floats2half2_rn(acc[g][a][2],acc[g][a][3]);
                }
            }else{
                *reinterpret_cast<half2*>(stores+(g*16+lane/4)*(N+8)+n)=__floats2half2_rn(acc[g][a][0],acc[g][a][1]);
                *reinterpret_cast<half2*>(stores+(g*16+8+lane/4)*(N+8)+n)=__floats2half2_rn(acc[g][a][2],acc[g][a][3]);
            }
        }
    }
    if constexpr(!Direct){
        __syncthreads();
        for(int index=threadIdx.x*8;index<G*16*N;index+=blockDim.x*8){
            int row=index/N, n=blockIdx.y*N+index%N, destination=(first+row)*v+n;
            half* source=stores+row*(N+8)+index%N;
            if(n+8<=v && destination%8==0)*reinterpret_cast<uint4*>(out+destination)=*reinterpret_cast<const uint4*>(source);
            else{
                #pragma unroll
                for(int j=0;j<8;++j)if(n+j<v)out[destination+j]=source[j];
            }
        }
    }
    if constexpr(Count){if(threadIdx.x==0){
        #pragma unroll
        for(int g=0;g<G;++g){
            int slot=((first/16+g)*gridDim.y+blockIdx.y)*2;
            stats[slot]=static_cast<long long>(issued[g])*4;
            stats[slot+1]=static_cast<long long>(8*A-issued[g])*4;
        }
    }}
}

template<int N,int G,bool Direct,bool Prefetch>
void dispatch(torch::Tensor x,torch::Tensor s,torch::Tensor w,torch::Tensor out,torch::Tensor stats,bool skip,bool count){
    dim3 grid(x.size(0)/(16*G),(out.size(1)+N-1)/N);
    auto stream=at::cuda::getCurrentCUDAStream();
    #define CALL(S,C) projection<N,G,S,Direct,Prefetch,C><<<grid,128,0,stream>>>(reinterpret_cast<half*>(x.data_ptr()),s.data_ptr<int>(),reinterpret_cast<half*>(w.data_ptr()),reinterpret_cast<half*>(out.data_ptr()),reinterpret_cast<long long*>(stats.data_ptr()),out.size(1))
    if(skip){if(count){CALL(true,true);}else{CALL(true,false);}}
    else{if(count){CALL(false,true);}else{CALL(false,false);}}
    #undef CALL
}

template<int N> void family(torch::Tensor x,torch::Tensor s,torch::Tensor w,torch::Tensor out,torch::Tensor stats,bool skip,bool count,int g,bool direct,bool prefetch){
    if(g==2)dispatch<N,2,false,false>(x,s,w,out,stats,skip,count);
    else if(direct)dispatch<N,1,true,false>(x,s,w,out,stats,skip,count);
    else if(prefetch)dispatch<N,1,false,true>(x,s,w,out,stats,skip,count);
    else TORCH_CHECK(false,"Explicit variant required");
}

void project(torch::Tensor x,torch::Tensor s,torch::Tensor w,torch::Tensor out,torch::Tensor stats,bool skip,bool count,int n,int g,bool direct,bool prefetch){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf && x.dim()==2 && x.size(1)==128,"CUDA FP16 D128 required");
    TORCH_CHECK((n==128 || n==256) && (g==1 || g==2) && x.size(0)>0 && x.size(0)%(16*g)==0,"Unsupported consumer tile");
    TORCH_CHECK(!(direct && prefetch) && !(g==2 && (direct || prefetch)),"One mechanism per family");
    for(const auto& t:{x,w,out})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==at::kHalf && t.is_contiguous(),"Operand type/layout mismatch");
    TORCH_CHECK(out.dim()==2 && out.size(0)==x.size(0) && out.size(1)>0 && out.size(1)%8==0 && w.sizes()==at::IntArrayRef({8,(out.size(1)+7)/8,128}),"Weight/output shape mismatch");
    TORCH_CHECK(s.device()==x.device() && s.scalar_type()==at::kInt && s.is_contiguous() && s.dim()==1 && s.numel()==x.size(0)/16,"Support shape/type mismatch");
    TORCH_CHECK(stats.device()==x.device() && stats.scalar_type()==at::kLong && stats.is_contiguous() && stats.sizes()==at::IntArrayRef({x.size(0)/16,(out.size(1)+n-1)/n,2}),"Counter shape mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    if(n==128)family<128>(x,s,w,out,stats,skip,count,g,direct,prefetch);
    else family<256>(x,s,w,out,stats,skip,count,g,direct,prefetch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("project",&project);}
