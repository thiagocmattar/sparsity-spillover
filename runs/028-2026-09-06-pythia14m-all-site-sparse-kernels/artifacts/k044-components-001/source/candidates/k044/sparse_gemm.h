// Derived from Flash Attention utils.h (Tri Dao, BSD-3-Clause; LICENSE-FLASH).
// Preserve every accumulator's K-step order. Bypass only MMA atoms whose
// complete warp-distributed A or B operand is exactly zero.
#pragma once
#include "kernel_traits.h"
#include "utils.h"

namespace FLASH_NAMESPACE {
template<typename Fragment>
__device__ __forceinline__ bool run028_nonzero(Fragment const& x){
    unsigned bits=0;
    #pragma unroll
    for(int e=0;e<size(x);++e)bits|=unsigned(x(e).raw()) & 0x7fffu;
    return __any_sync(0xffffffffu,bits!=0);
}

// Predicate is warp-uniform because each operand test uses a full-warp vote.
// Keep each accumulator unchanged when the guarded tensor instruction is false.
template<typename A,typename B,typename C>
__device__ __forceinline__ void run028_predicated_atom(A const& a,B const& b,C& c,bool active){
    static_assert(decltype(size(a))::value==8 && decltype(size(b))::value==4 && decltype(size(c))::value==4);
    auto ar=cute::recast<uint32_t>(a);auto br=cute::recast<uint32_t>(b);
    asm volatile("{ .reg .pred run028_p; setp.ne.u32 run028_p, %10, 0; "
        "@run028_p mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}; }"
        : "+f"(c(0)),"+f"(c(1)),"+f"(c(2)),"+f"(c(3))
        : "r"(ar(0)),"r"(ar(1)),"r"(ar(2)),"r"(ar(3)),"r"(br(0)),"r"(br(1)),"r"(int(active)));
}

template<bool Skip,bool Count,typename MMA,typename A,typename B,typename C>
__device__ __forceinline__ void run028_mma(MMA mma,A const& a,B const& b,C& c,int& issued,int& skipped){
    constexpr int M=decltype(size<1>(a))::value,N=decltype(size<1>(b))::value;
    if constexpr(!Skip){
        cute::gemm(mma,a,b,c);
        if constexpr(Count)issued+=M*N;
    }else{
        bool active_a[M],active_b[N];
        #pragma unroll
        for(int m=0;m<M;++m)active_a[m]=run028_nonzero(a(_,m));
        #pragma unroll
        for(int n=0;n<N;++n)active_b[n]=run028_nonzero(b(_,n));
        #pragma unroll
        for(int m=0;m<M;++m){
            #pragma unroll
            for(int n=0;n<N;++n){
                int ns=(m&1)?N-1-n:n;
                bool active=active_a[m] && active_b[ns];auto cv=c(_,m,ns);
                run028_predicated_atom(a(_,m),b(_,ns),cv,active);
                if constexpr(Count){issued+=int(active);skipped+=int(!active);}
            }
        }
    }
}

template<bool Skip,bool Count,bool A_in_regs=false,bool B_in_regs=false,
         typename Tensor0,typename Tensor1,typename Tensor2,typename Tensor3,typename Tensor4,
         typename TiledMma,typename TiledCopyA,typename TiledCopyB,typename ThrCopyA,typename ThrCopyB>
__device__ __forceinline__ void run028_gemm(Tensor0& acc,Tensor1& tCrA,Tensor2& tCrB,Tensor3 const& tCsA,
        Tensor4 const& tCsB,TiledMma tiled_mma,TiledCopyA copy_A,TiledCopyB copy_B,
        ThrCopyA thr_A,ThrCopyB thr_B,int& issued,int& skipped){
    auto av=thr_A.retile_D(tCrA);auto bv=thr_B.retile_D(tCrB);
    if(!A_in_regs)cute::copy(copy_A,tCsA(_,_,_0{}),av(_,_,_0{}));
    if(!B_in_regs)cute::copy(copy_B,tCsB(_,_,_0{}),bv(_,_,_0{}));
    #pragma unroll
    for(int i=0;i<size<2>(tCrA);++i){
        if(i<size<2>(tCrA)-1){
            if(!A_in_regs)cute::copy(copy_A,tCsA(_,_,i+1),av(_,_,i+1));
            if(!B_in_regs)cute::copy(copy_B,tCsB(_,_,i+1),bv(_,_,i+1));
        }
        run028_mma<Skip,Count>(tiled_mma,tCrA(_,_,i),tCrB(_,_,i),acc,issued,skipped);
    }
}

template<bool Skip,bool Count,typename Tensor0,typename Tensor1,typename Tensor2,typename Tensor3,
         typename TiledMma,typename TiledCopy,typename ThrCopy>
__device__ __forceinline__ void run028_gemm_rs(Tensor0& acc,Tensor1& tCrA,Tensor2& tCrB,
        Tensor3 const& tCsB,TiledMma tiled_mma,TiledCopy copy_B,ThrCopy thr_B,int& issued,int& skipped){
    auto bv=thr_B.retile_D(tCrB);
    cute::copy(copy_B,tCsB(_,_,_0{}),bv(_,_,_0{}));
    #pragma unroll
    for(int i=0;i<size<2>(tCrA);++i){
        if(i<size<2>(tCrA)-1)cute::copy(copy_B,tCsB(_,_,i+1),bv(_,_,i+1));
        run028_mma<Skip,Count>(tiled_mma,tCrA(_,_,i),tCrB(_,_,i),acc,issued,skipped);
    }
}
}
