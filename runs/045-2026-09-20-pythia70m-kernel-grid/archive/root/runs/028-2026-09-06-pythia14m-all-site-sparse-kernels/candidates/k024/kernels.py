"""K024 reverse online-softmax with exact-zero query/K/V tile bypasses.

Unlike K020-23, the fallback uses BF16 tensor-core dot products. It does not
claim to skip individual zeros inside a nonzero tile. All causal keys remain
in the softmax normalizer. Counter values include executed tile padding.
"""
import triton
import triton.language as tl

@triton.jit
def prefix_chunks(V,Prefix,Totals,T:tl.constexpr,NC:tl.constexpr,CHUNK:tl.constexpr):
    block=tl.program_id(0);head=tl.program_id(1)
    rows=block*CHUNK+tl.arange(0,CHUNK);dims=tl.arange(0,32)
    v=tl.load(V+(head*T+rows[:,None])*32+dims[None,:],mask=rows[:,None]<T,other=0).to(tl.float32)
    values=tl.cumsum(v,axis=0)
    tl.store(Prefix+(head*T+rows[:,None])*32+dims[None,:],values,mask=rows[:,None]<T)
    tl.store(Totals+(head*NC+block)*32+dims,tl.sum(v,axis=0))

@triton.jit
def attention(Q,K,V,Prefix,Totals,Out,Stats,T:tl.constexpr,SCALE:tl.constexpr,
              NC:tl.constexpr,NCP:tl.constexpr,CHUNK:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,PREFIX:tl.constexpr,COUNT:tl.constexpr):
    tile=tl.program_id(0);head=tl.program_id(1)
    rows=tile*BM+tl.arange(0,BM);dims=tl.arange(0,32)
    q=tl.load(Q+(head*T+rows[:,None])*32+dims[None,:],mask=rows[:,None]<T,other=0)
    zero=tl.sum(tl.sum((q!=0).to(tl.int32),axis=1),axis=0)==0
    qwork=tl.full((),0,tl.int32);pwork=tl.full((),0,tl.int32)
    if PREFIX and zero:
        earlier=tl.arange(0,NCP);chunk=tile*BM//CHUNK
        totals=tl.load(Totals+(head*NC+earlier[:,None])*32+dims[None,:],mask=(earlier[:,None]<chunk)&(earlier[:,None]<NC),other=0)
        local=tl.load(Prefix+(head*T+rows[:,None])*32+dims[None,:],mask=rows[:,None]<T,other=0)
        out=(local+tl.sum(totals,axis=0)[None,:])/(rows[:,None]+1).to(tl.float32)
    else:
        maximum=tl.full((BM,),float('-inf'),tl.float32)
        normalizer=tl.zeros((BM,),tl.float32)
        acc=tl.zeros((BM,32),tl.float32)
        offsets=tl.arange(0,BN)
        end=tl.cdiv(tl.minimum((tile+1)*BM,T),BN)
        # Reverse causal key-block traversal, matching Flash-style online
        # BF16 exp-probability accumulation rather than normalized-P rounding.
        for step in range(end):
            keys=(end-1-step)*BN+offsets
            k=tl.load(K+(head*T+keys[:,None])*32+dims[None,:],mask=keys[:,None]<T,other=0)
            kz=tl.sum(tl.sum((k!=0).to(tl.int32),axis=1),axis=0)==0
            if kz:
                scores=tl.zeros((BM,BN),tl.float32)
            else:
                scores=tl.dot(q,tl.trans(k))
                if COUNT:qwork+=BN*32
            scores=tl.where((keys[None,:]<=rows[:,None])&(keys[None,:]<T),scores,float('-inf'))
            next_max=tl.maximum(maximum,tl.max(scores,axis=1))
            correction=tl.exp2((maximum-next_max)*(SCALE*1.4426950408889634))
            prob=tl.exp2(scores*(SCALE*1.4426950408889634)-next_max[:,None]*(SCALE*1.4426950408889634))
            v=tl.load(V+(head*T+keys[:,None])*32+dims[None,:],mask=keys[:,None]<T,other=0)
            vz=tl.sum(tl.sum((v!=0).to(tl.int32),axis=1),axis=0)==0
            acc=acc*correction[:,None]
            if not vz:
                acc+=tl.dot(prob.to(tl.bfloat16),v)
                if COUNT:pwork+=BN*32
            normalizer=normalizer*correction+tl.sum(prob,axis=1)
            maximum=next_max
        out=acc/tl.maximum(normalizer[:,None],1.e-30)
    tl.store(Out+(head*T+rows[:,None])*32+dims[None,:],out.to(tl.bfloat16),mask=rows[:,None]<T)
    if COUNT:
        tl.store(Stats+(head*T+rows)*3,qwork,mask=rows<T)
        tl.store(Stats+(head*T+rows)*3+1,pwork,mask=rows<T)
        prefix_flag=tl.full((),1 if PREFIX else 0,tl.int64)*zero.to(tl.int64)
        tl.store(Stats+(head*T+rows)*3+2,prefix_flag,mask=rows<T)
