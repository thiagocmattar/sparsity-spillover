"""Wider feature-union groups and separately materialized tile support.

Both families rebuild all input-dependent metadata on every invocation.
The no-skip controls retain that producer and force all reduction work.
Counters report executed reduction tiles, not measured memory traffic.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def union_index(X, I, C, M:tl.constexpr, K:tl.constexpr, T:tl.constexpr, BM:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.arange(0,K)
    x=tl.load(X+r[:,None]*K+k[None,:],r[:,None]<M,0)
    active=tl.sum(((x.to(tl.float32)>=T)&(x!=0)).to(tl.int32),axis=0)>0
    pos=tl.cumsum(active.to(tl.int32),axis=0)-1
    tl.store(I+tl.program_id(0)*K+pos,k,active)
    tl.store(C+tl.program_id(0),tl.sum(active.to(tl.int32),axis=0))

@tr.jit
def union_consume(X,W,B,Y,I,C,Work,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
                  T:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr,
                  NO_SKIP:tl.constexpr,COUNT:tl.constexpr):
    group=tl.program_id(0); col=tl.program_id(1)
    r=group*BM+tl.arange(0,BM); n=col*BN+tl.arange(0,BN)
    count=K if NO_SKIP else tl.load(C+group)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(0,tl.cdiv(count,BK)):
        p=start*BK+tl.arange(0,BK)
        if NO_SKIP: k=p
        else: k=tl.load(I+group*K+p,p<count,0)
        a=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(p[None,:]<count),0)
        a=tl.where(a.to(tl.float32)>=T,a,0)
        w=tl.load(W+k[:,None]*N+n[None,:],(p[:,None]<count)&(n[None,:]<N),0)
        acc=tl.dot(a,w,acc)
    acc+=tl.load(B+n,n<N,0).to(tl.float32)[None,:]
    tl.store(Y+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))
    if COUNT: tl.store(Work+group*tr.cdiv(N,BN)+col,tl.cdiv(count,BK))

@tr.jit
def tile_prepare(X,G,Mask,M:tl.constexpr,K:tl.constexpr,T:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)*BM+tl.arange(0,BM)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    a=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
    a=tl.where(a.to(tl.float32)>=T,a,0)
    tl.store(G+r[:,None]*K+k[None,:],a,(r[:,None]<M)&(k[None,:]<K))
    active=tl.sum(tl.sum((a!=0).to(tl.int32),axis=1),axis=0)>0
    tl.store(Mask+tl.program_id(0)*tr.cdiv(K,BK)+tl.program_id(1),active)

@tr.jit
def tile_consume(G,W,B,Y,Mask,Work,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
                 BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr,NO_SKIP:tl.constexpr,COUNT:tl.constexpr):
    group=tl.program_id(0); col=tl.program_id(1)
    r=group*BM+tl.arange(0,BM); n=col*BN+tl.arange(0,BN)
    acc=tl.zeros((BM,BN),tl.float32); steps=0
    for tile in range(tr.cdiv(K,BK)):
        active=tl.load(Mask+group*tr.cdiv(K,BK)+tile)
        if NO_SKIP or active:
            k=tile*BK+tl.arange(0,BK)
            a=tl.load(G+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
            w=tl.load(W+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N),0)
            acc=tl.dot(a,w,acc); steps+=1
    acc+=tl.load(B+n,n<N,0).to(tl.float32)[None,:]
    tl.store(Y+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))
    if COUNT: tl.store(Work+group*tr.cdiv(N,BN)+col,steps)

class Linear:
    def __init__(self, linear, threshold, spec, m=2048, no_skip=False):
        self.spec=spec; self.m=m; self.k=linear.in_features; self.n=linear.out_features
        self.w=linear.weight.t().contiguous(); self.bias=linear.bias
        self.t=float(torch.tensor(threshold,dtype=torch.bfloat16))
        self.no_skip=no_skip; self.count=False; self.compiled=[]
        kw={'device':self.w.device}; groups=tr.cdiv(m,spec['bm'])
        self.out=torch.empty((m,self.n),dtype=torch.bfloat16,**kw)
        self.work=torch.empty((groups,tr.cdiv(self.n,spec['bn'])),dtype=torch.int32,**kw)
        if spec['family']=='union':
            self.index=torch.empty((groups,self.k),dtype=torch.int32,**kw)
            self.nnz=torch.empty(groups,dtype=torch.int32,**kw)
        else:
            self.gated=torch.empty((m,self.k),dtype=torch.bfloat16,**kw)
            self.mask=torch.empty((groups,tr.cdiv(self.k,spec['bk'])),dtype=torch.int32,**kw)
    def __call__(self,x):
        assert x.is_contiguous() and x.numel()==self.m*self.k and x.dtype==torch.bfloat16
        s=self.spec; m,k,n=self.m,self.k,self.n
        grid=(tr.cdiv(m,s['bm']),tr.cdiv(n,s['bn']))
        if s['family']=='union':
            a=union_index[(grid[0],)](x,self.index,self.nnz,m,k,self.t,s['bm'],num_warps=8)
            b=union_consume[grid](x,self.w,self.bias,self.out,self.index,self.nnz,self.work,
                m,k,n,self.t,s['bm'],s['bk'],s['bn'],self.no_skip,self.count,num_warps=4)
        else:
            a=tile_prepare[(grid[0],tr.cdiv(k,s['bk']))](x,self.gated,self.mask,m,k,self.t,s['bm'],s['bk'],num_warps=4)
            b=tile_consume[grid](self.gated,self.w,self.bias,self.out,self.mask,self.work,
                m,k,n,s['bm'],s['bk'],s['bn'],self.no_skip,self.count,num_warps=4)
        self.compiled=[a,b]
        return self.out

def expected_work(x, threshold, spec, no_skip=False, n=512):
    """Independent integer count from the actual gated tensor."""
    a=x.reshape(-1,x.shape[-1]).masked_fill(x.reshape(-1,x.shape[-1])<threshold,0)
    m,k=a.shape; bm,bk,bn=spec['bm'],spec['bk'],spec['bn']
    assert m%bm==0 and k%bk==0
    union=(a!=0).reshape(m//bm,bm,k).any(1)
    if no_skip: steps=torch.full((m//bm,),k//bk,device=a.device,dtype=torch.int64)
    elif spec['family']=='union':steps=(union.sum(-1)+bk-1)//bk
    else:steps=union.reshape(m//bm,k//bk,bk).any(-1).sum(-1)
    return steps[:,None].expand(-1,tr.cdiv(n,bn))
