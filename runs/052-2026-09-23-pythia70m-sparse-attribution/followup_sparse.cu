// Eight evidence-led SIMT follow-ups: aligned vector loads and per-row CTA compaction.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

template<int V>
__device__ __forceinline__ void accumulate(const __nv_bfloat16* weight,float value,float (&sum)[V]) {
 unsigned words[V/2];
 if constexpr(V==2)words[0]=*reinterpret_cast<const unsigned*>(weight);
 if constexpr(V==4){uint2 p=*reinterpret_cast<const uint2*>(weight);words[0]=p.x;words[1]=p.y;}
 if constexpr(V>=8){
  #pragma unroll
  for(int j=0;j<V/8;j++){uint4 p=*reinterpret_cast<const uint4*>(weight+j*8);words[j*4]=p.x;words[j*4+1]=p.y;words[j*4+2]=p.z;words[j*4+3]=p.w;}
 }
 #pragma unroll
 for(int j=0;j<V/2;j++) {
  float lo=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(words[j])));
  float hi=__bfloat162float(__ushort_as_bfloat16(static_cast<unsigned short>(words[j]>>16)));
  sum[j*2]=fmaf(value,lo,sum[j*2]);sum[j*2+1]=fmaf(value,hi,sum[j*2+1]);
 }
}

template<int V,int ROWS,bool NO_SKIP>
__global__ void vector_warps(const __nv_bfloat16* x,const __nv_bfloat16* w,
 const __nv_bfloat16* bias,__nv_bfloat16* out,int* counts,int m,int k,float threshold) {
 int lane=threadIdx.x&31,row=blockIdx.x*ROWS+(threadIdx.x>>5);
 if(row>=m)return;
 int col=blockIdx.y*32*V+lane*V,nnz=0;float sum[V]={};
 for(int start=0;start<k;start+=32){
  float value=__bfloat162float(x[(long long)row*k+start+lane]);
  value=value>=threshold?value:0.f;
  unsigned active=__ballot_sync(0xffffffff,value!=0.f);nnz+=__popc(active);
  unsigned todo=NO_SKIP?0xffffffff:active;
  while(todo){int source=__ffs(todo)-1;float v=__shfl_sync(0xffffffff,value,source);
   accumulate<V>(w+(long long)(start+source)*512+col,v,sum);todo&=todo-1;}
 }
 if(lane==0)counts[row*gridDim.y+blockIdx.y]=nnz;
 #pragma unroll
 for(int j=0;j<V;j++)out[(long long)row*512+col+j]=__float2bfloat16_rn(sum[j]+__bfloat162float(bias[col+j]));
}

template<int K,int V,bool NO_SKIP>
__global__ void compact_row(const __nv_bfloat16* x,const __nv_bfloat16* w,
 const __nv_bfloat16* bias,__nv_bfloat16* out,int* counts,float threshold) {
 constexpr int THREADS=512/V,CHUNKS=K/32;
 __shared__ unsigned masks[CHUNKS];
 __shared__ int offsets[CHUNKS],indices[K],nnz;
 __shared__ float values[K];
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,row=blockIdx.x;
 const __nv_bfloat16* input=x+(long long)row*K;
 for(int chunk=warp;chunk<CHUNKS;chunk+=THREADS/32){
  float value=__bfloat162float(input[chunk*32+lane]);
  unsigned mask=__ballot_sync(0xffffffff,value>=threshold && value!=0.f);
  if(lane==0)masks[chunk]=mask;
 }
 __syncthreads();
 if(warp==0){
  int c1=lane<CHUNKS?__popc(masks[lane]):0;
  int c2=lane+32<CHUNKS?__popc(masks[lane+32]):0;
  int p1=c1,p2=c2;
  #pragma unroll
  for(int step=1;step<32;step*=2){
   int a=__shfl_up_sync(0xffffffff,p1,step),b=__shfl_up_sync(0xffffffff,p2,step);
   if(lane>=step){p1+=a;p2+=b;}
  }
  int first=__shfl_sync(0xffffffff,p1,31),second=__shfl_sync(0xffffffff,p2,31);
  if(lane<CHUNKS)offsets[lane]=p1-c1;
  if(lane+32<CHUNKS)offsets[lane+32]=first+p2-c2;
  if(lane==0){nnz=first+second;counts[row]=nnz;}
 }
 __syncthreads();
 for(int feature=tid;feature<K;feature+=THREADS){
  unsigned mask=masks[feature/32],bit=1u<<(feature&31);
  if(NO_SKIP || (mask&bit)){
   int position=NO_SKIP?feature:offsets[feature/32]+__popc(mask&(bit-1));
   float value=__bfloat162float(input[feature]);
   indices[position]=feature;values[position]=value>=threshold?value:0.f;
  }
 }
 __syncthreads();
 int count=NO_SKIP?K:nnz,col=tid*V;float sum[V]={};
 for(int i=0;i<count;i++)accumulate<V>(w+(long long)indices[i]*512+col,values[i],sum);
 #pragma unroll
 for(int j=0;j<V;j++)out[(long long)row*512+col+j]=__float2bfloat16_rn(sum[j]+__bfloat162float(bias[col+j]));
}

void r052_followup(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,
 torch::Tensor counts,double threshold,int load_v,int rows,bool compact,bool no_skip){
 for(auto t:{x,w,bias,out,counts}){
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.device()==x.device(),"contiguous tensors on one CUDA device required");
 }
 for(auto t:{x,w,bias,out})TORCH_CHECK(t.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(counts.scalar_type()==at::kInt,"int32 counts required");
 TORCH_CHECK(x.dim()==2 && w.dim()==2 && out.dim()==2,"matrix operands required");
 int m=x.size(0),k=x.size(1);
 TORCH_CHECK((k==512 || k==2048) && w.size(0)==k && w.size(1)==512 && out.size(0)==m && out.size(1)==512 && bias.numel()==512,"fixed70M shapes required");
 auto stream=at::cuda::getCurrentCUDAStream();
 #define ARGS reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),counts.data_ptr<int>()
 #define C(K,V,S) compact_row<K,V,S><<<m,512/V,0,stream>>>(ARGS,threshold)
 #define CS(K,V) if(no_skip){C(K,V,true);}else{C(K,V,false);}
 #define CV(V) if(k==512){CS(512,V);}else{CS(2048,V);}
 #define W(V,R,S) vector_warps<V,R,S><<<dim3((m+R-1)/R,512/(32*V)),R*32,0,stream>>>(ARGS,m,k,threshold)
 #define WS(V,R) if(no_skip){W(V,R,true);}else{W(V,R,false);}
 if(compact){
  TORCH_CHECK(counts.dim()==2 && counts.size(0)==m && counts.size(1)==1,"compact count shape");
  if(load_v==2){CV(2);}else if(load_v==4){CV(4);}else if(load_v==8){CV(8);}else if(load_v==16){CV(16);}else TORCH_CHECK(false,"vector width");
 }else{
  TORCH_CHECK((load_v==4 || load_v==8) && (rows==4 || rows==8),"warp configuration");
  TORCH_CHECK(counts.dim()==2 && counts.size(0)==m && counts.size(1)==512/(32*load_v),"warp count shape");
  if(load_v==4 && rows==4){WS(4,4);}else if(load_v==4 && rows==8){WS(4,8);}else if(load_v==8 && rows==4){WS(8,4);}else{WS(8,8);}
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("followup",&r052_followup);}
