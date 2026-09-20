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
                if(active_a[m] && active_b[ns]){
                    cute::gemm(mma,a(_,m),b(_,ns),c(_,m,ns));
                    if constexpr(Count)++issued;
                }else if constexpr(Count)++skipped;
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
