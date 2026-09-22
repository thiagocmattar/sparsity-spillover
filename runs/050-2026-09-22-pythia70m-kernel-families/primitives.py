"""Different sparse representations; every dynamic conversion is in __call__."""
import torch
import triton as tr
import triton.language as tl
from functools import lru_cache
from support import RUN,sha

@tr.jit
def gate(X,Y,S:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);x=tl.load(X+i,i<S,0)
 tl.store(Y+i,tl.where(x.to(tl.float32)>=T,x,0),i<S)

@tr.jit
def combine(H,Z,R,Y,S:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B)
 h=tl.load(H+i,i<S,0).to(tl.float32);z=tl.load(Z+i,i<S,0).to(tl.float32)
 r=tl.load(R+i,i<S,0).to(tl.float32)
 s=(h+z).to(tl.bfloat16).to(tl.float32)
 tl.store(Y+i,s+r,i<S)

@tr.jit
def pack(X,P,C,K:tl.constexpr,T:tl.constexpr):
 row=tl.program_id(0);tile=tl.program_id(1);a=tl.arange(0,256);idx=tile*256+a
 x=tl.load(X+row*K+idx);x=tl.where(x.to(tl.float32)>=T,x,0)
 active=x!=0;pos=tl.cumsum(active.to(tl.int32))-1
 bits=(x.to(tl.uint16,bitcast=True).to(tl.uint32)<<16)|idx.to(tl.uint32)
 tl.store(P+row*K+tile*256+pos,bits,active)
 tl.store(C+row*(K//256)+tile,tl.sum(active.to(tl.int32),0))

@tr.jit
def masked(X,W,Bias,Y,Part,K:tl.constexpr,N:tl.constexpr,M:tl.constexpr,T:tl.constexpr,
           BN:tl.constexpr,SPLIT:tl.constexpr,BK:tl.constexpr):
 r=tl.program_id(0);n=tl.program_id(1)*BN+tl.arange(0,BN);s=tl.program_id(2)
 acc=tl.full((BN,),0,tl.float32)
 for tile in range(s,tr.cdiv(K,BK),SPLIT):
  k=tile*BK+tl.arange(0,BK);x=tl.load(X+r*K+k,k<K,0).to(tl.float32)
  x=tl.where(x>=T,x,0);w=tl.load(W+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N)&(x[:,None]!=0),0).to(tl.float32)
  acc+=tl.sum(x[:,None]*w,0)
 if SPLIT==1:
  acc+=tl.load(Bias+n,n<N,0).to(tl.float32);tl.store(Y+r*N+n,acc,n<N)
 else:tl.store(Part+s*M*N+r*N+n,acc,n<N)

@tr.jit
def reduce_parts(P,Bias,Y,M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);a=tl.full((B,),0,tl.float32)
 for s in range(SPLIT):a+=tl.load(P+s*M*N+i,i<M*N,0)
 a+=tl.load(Bias+i%N).to(tl.float32);tl.store(Y+i,a,i<M*N)

@tr.jit
def union_index(X,Idx,C,M:tl.constexpr,K:tl.constexpr,T:tl.constexpr,GM:tl.constexpr,PK:tl.constexpr):
 group=tl.program_id(0);r=group*GM+tl.arange(0,GM);k=tl.arange(0,PK)
 x=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0).to(tl.float32)
 act=tl.sum(((x>=T)&(x!=0)).to(tl.int32),0)>0
 pos=tl.cumsum(act.to(tl.int32))-1
 tl.store(Idx+group*K+pos,k,act);tl.store(C+group,tl.sum(act.to(tl.int32),0))

@tr.jit
def gathered(X,W,Bias,Idx,C,Y,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,T:tl.constexpr,GM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr,NO_SKIP:tl.constexpr=False):
 group=tl.program_id(0);r=group*GM+tl.arange(0,16);n=tl.program_id(1)*BN+tl.arange(0,BN)
 count=K if NO_SKIP else tl.load(C+group);acc=tl.full((16,BN),0,tl.float32)
 for start in range(0,tl.cdiv(count,BK)):
  p=start*BK+tl.arange(0,BK)
  if NO_SKIP:k=p
  else:k=tl.load(Idx+group*K+p,p<count,0)
  a=tl.load(X+r[:,None]*K+k[None,:],(tl.arange(0,16)[:,None]<GM)&(r[:,None]<M)&(p[None,:]<count),0)
  a=tl.where(a.to(tl.float32)>=T,a,0)
  w=tl.load(W+k[:,None]*N+n[None,:],(p[:,None]<count)&(n[None,:]<N),0)
  acc=tl.dot(a,w,acc)
 acc+=tl.load(Bias+n,n<N,0).to(tl.float32)[None,:]
 tl.store(Y+r[:,None]*N+n[None,:],acc,(tl.arange(0,16)[:,None]<GM)&(r[:,None]<M)&(n[None,:]<N))

@tr.jit
def dense_gate(X,W,Bias,Y,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,T:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
 r=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
 acc=tl.full((BM,BN),0,tl.float32)
 for start in range(tr.cdiv(K,BK)):
  k=start*BK+kk;a=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
  a=tl.where(a.to(tl.float32)>=T,a,0)
  w=tl.load(W+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N),0)
  acc=tl.dot(a,w,acc)
 acc+=tl.load(Bias+n,n<N,0).to(tl.float32)[None,:]
 tl.store(Y+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))

@tr.jit
def split24(X,A,E,M:tl.constexpr,K:tl.constexpr,T:tl.constexpr,B:tl.constexpr):
 group=tl.program_id(0)*B+tl.arange(0,B);o=tl.arange(0,4)
 i=group[:,None]*4+o[None,:];x=tl.load(X+i,i<M*K,0)
 x=tl.where(x.to(tl.float32)>=T,x,0);order=tl.cumsum((x!=0).to(tl.int32),1)
 tl.store(A+i,tl.where(order<=2,x,0),i<M*K)
 tl.store(E+i,tl.where(order>2,x,0),i<M*K)

@tr.jit
def overflow_dot(E,W,Y,Z,Bias,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,BM:tl.constexpr,BK:tl.constexpr,BN:tl.constexpr):
 r=tl.program_id(0)*BM+tl.arange(0,BM);n=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,BK)
 acc=tl.full((BM,BN),0,tl.float32)
 for start in range(tr.cdiv(K,BK)):
  k=start*BK+kk;a=tl.load(E+r[:,None]*K+k[None,:],(r[:,None]<M)&(k[None,:]<K),0)
  if tl.sum(tl.sum((a!=0).to(tl.int32),0),0)>0:
   w=tl.load(W+k[:,None]*N+n[None,:],(k[:,None]<K)&(n[None,:]<N),0)
   acc=tl.dot(a,w,acc)
 acc+=tl.load(Y+r[:,None]*N+n[None,:],(r[:,None]<M)&(n[None,:]<N),0).to(tl.float32)
 acc+=tl.load(Bias+n,n<N,0).to(tl.float32)[None,:]
 tl.store(Z+r[:,None]*N+n[None,:],acc,(r[:,None]<M)&(n[None,:]<N))

@lru_cache(None)
def extension():
 from torch.utils.cpp_extension import load
 return load(name='r050_pack_'+sha(RUN/'packed.cu')[:12],sources=[str(RUN/'packed.cu')],extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'],verbose=True)

def candidates(include_ablations=False):
 out=[{'id':'dense_native','family':'dense'}, {'id':'dense_fused','family':'dense'},
      {'id':'dense_dot','family':'dense','bm':32,'bk':32}]
 for rows in (2,4):
  for v in (4,8,16):out.append({'id':f'a_r{rows}_v{v}','family':'a','rows':rows,'v':v})
 for split in (1,2,4):
  for bn in (64,128):out.append({'id':f'b_s{split}_n{bn}','family':'b','split':split,'bn':bn})
 for gm in (4,8,16):
  for bk in (32,64):out.append({'id':f'c_m{gm}_k{bk}','family':'c','gm':gm,'bk':bk})
 for bm in (16,32,64):
  for bk in (32,64):out.append({'id':f'd_m{bm}_k{bk}','family':'d','bm':bm,'bk':bk})
 # Six bounded refinements after A/C passed the initial matched screen.
 for gm,bn in ((1,64),(2,64),(4,128),(8,128),(16,128)):
  out.append({'id':f'c_m{gm}_k32_n{bn}','family':'c','gm':gm,'bk':32,'bn':bn,'stage':'refinement'})
 out.append({'id':'a_r8_v4','family':'a','rows':8,'v':4,'stage':'refinement'})
 if include_ablations:
  out += [{**s,'id':s['id']+'_noskip','no_skip':True,'stage':'ablation'} for s in out if s['family']=='c']
 return out

class Linear:
 def __init__(self,linear,threshold,spec,m=2048):
  self.spec=spec;self.raw=linear;self.w=linear.weight.t().contiguous();self.bias=linear.bias
  self.k,self.n=self.w.shape;self.m=m
  self.t=float(torch.tensor(threshold,dtype=torch.bfloat16)) if threshold is not None else -float('inf')
  self.compiled=[]
  kw={'device':self.w.device};self.out=torch.empty((m,self.n),dtype=torch.bfloat16,**kw)
  f=spec['family']
  if f=='a':
   self.p=torch.empty((m,self.k),dtype=torch.int32,**kw);self.count=torch.empty((m,self.k//256),dtype=torch.int32,**kw);self.ext=extension()
  if f=='b':self.part=torch.empty((spec['split'],m,self.n),dtype=torch.float32,**kw)
  if f=='c':
   g=tr.cdiv(m,spec['gm']);self.idx=torch.empty((g,self.k),dtype=torch.int32,**kw);self.count=torch.empty((g,),dtype=torch.int32,**kw)
  if f=='d':
   from structured import extension as structured_extension
   self.a=torch.empty((m,self.k//2),dtype=torch.bfloat16,**kw)
   self.e=torch.empty((m,self.k),dtype=torch.bfloat16,**kw)
   self.meta=torch.empty((m*self.k//16,),dtype=torch.int16,**kw)
   self.y=torch.empty((m,self.n),dtype=torch.float32,**kw)
   self.structured=structured_extension()
  if spec['id']=='dense_fused':self.gated=torch.empty((m,self.k),dtype=torch.bfloat16,**kw)
 def __call__(self,x):
  x=x.reshape(self.m,self.k);s=self.spec;f=s['family'];m,k,n=self.m,self.k,self.n
  if s['id']=='dense_native':return torch.nn.functional.linear(x.masked_fill(x<self.t,0),self.raw.weight,self.bias)
  if s['id']=='dense_fused':
   self.compiled=[gate[(tr.cdiv(m*k,1024),)](x,self.gated,m*k,self.t,1024)]
   return torch.nn.functional.linear(self.gated,self.raw.weight,self.bias)
  if f=='dense':self.compiled=[dense_gate[(tr.cdiv(m,s['bm']),tr.cdiv(n,64))](x,self.w,self.bias,self.out,m,k,n,self.t,s['bm'],64,s['bk'])]
  elif f=='a':
   self.compiled=[pack[(m,k//256)](x,self.p,self.count,k,self.t,num_warps=4)]
   self.ext.consume(self.p,self.count,self.w,self.bias,self.out,s['v'],s['rows'])
  elif f=='b':
   self.compiled=[masked[(m,tr.cdiv(n,s['bn']),s['split'])](x,self.w,self.bias,self.out,self.part,k,n,m,self.t,s['bn'],s['split'],32,num_warps=4)]
   if s['split']>1:self.compiled.append(reduce_parts[(tr.cdiv(m*n,256),)](self.part,self.bias,self.out,m,n,s['split'],256))
  elif f=='c':
   index_kernel=union_index[(tr.cdiv(m,s['gm']),)](x,self.idx,self.count,m,k,self.t,s['gm'],tr.next_power_of_2(k),num_warps=8)
   bn=s.get('bn',64)
   compute_kernel=gathered[(tr.cdiv(m,s['gm']),tr.cdiv(n,bn))](x,self.w,self.bias,self.idx,self.count,self.out,m,k,n,self.t,s['gm'],s['bk'],bn,s.get('no_skip',False),num_warps=4)
   self.compiled=[index_kernel,compute_kernel]
  elif f=='d':
   self.structured.run(x,self.raw.weight,self.a,self.e,self.meta,self.y,self.t)
   self.compiled=[overflow_dot[(tr.cdiv(m,s['bm']),tr.cdiv(n,64))](self.e,self.w,self.y,self.out,self.bias,m,k,n,s['bm'],s['bk'],64)]
  else:raise ValueError(s)
  return self.out
