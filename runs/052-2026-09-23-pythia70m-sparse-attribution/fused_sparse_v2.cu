// Run052: CTA-local active-K compaction/WMMA and warp-ballot SIMT.
// All gates, scans, metadata and output writes execute on the current stream.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
namespace wm = nvcuda::wmma;

template<int GM, int BN, bool NO_SKIP>
__global__ void gathered(const __nv_bfloat16* x, const __nv_bfloat16* w,
 const __nv_bfloat16* bias, __nv_bfloat16* out, int* observed, int m, int k, int n, float threshold) {
 constexpr int THREADS=BN*2;
 __shared__ unsigned masks[64];
 __shared__ int offsets[64], indices[2048], active_count;
 __shared__ __align__(32) __nv_bfloat16 a[16*16], b[16*BN];
 __shared__ __align__(32) float result[16*BN];
 int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
 int first_row=blockIdx.x*GM, first_col=blockIdx.y*BN;
 for(int chunk=warp;chunk<k/32;chunk+=THREADS/32) {
  int feature=chunk*32+lane;
  bool active=false;
  #pragma unroll
  for(int r=0;r<GM;r++) if(first_row+r<m) {
   float value=__bfloat162float(x[(long long)(first_row+r)*k+feature]);
   active |= value>=threshold && value!=0.f;
  }
  unsigned mask=__ballot_sync(0xffffffff,active);
  if(lane==0)masks[chunk]=mask;
 }
 __syncthreads();
 if(tid==0) {
  int total=0;
  for(int chunk=0;chunk<k/32;chunk++){offsets[chunk]=total;total+=__popc(masks[chunk]);}
  active_count=total;
  observed[blockIdx.x*gridDim.y+blockIdx.y]=total;
 }
 __syncthreads();
 for(int feature=tid;feature<k;feature+=THREADS) {
  unsigned mask=masks[feature/32], bit=1u<<(feature&31);
  if(NO_SKIP)indices[feature]=feature;
  else if(mask&bit)indices[offsets[feature/32]+__popc(mask&(bit-1))]=feature;
 }
 __syncthreads();
 int count=NO_SKIP?k:active_count;
 wm::fragment<wm::accumulator,16,16,16,float> acc;
 wm::fill_fragment(acc,0.f);
 for(int start=0;start<count;start+=16) {
  for(int i=tid;i<256;i+=THREADS) {
   int row=i/16,p=start+i%16;
   __nv_bfloat16 value=__float2bfloat16_rn(0.f);
   if(row<GM && first_row+row<m && p<count) {
    value=x[(long long)(first_row+row)*k+indices[p]];
    if(__bfloat162float(value)<threshold)value=__float2bfloat16_rn(0.f);
   }
   a[i]=value;
  }
  for(int i=tid;i<16*BN;i+=THREADS) {
   int p=start+i/BN,col=first_col+i%BN;
   b[i]=(p<count && col<n)?w[(long long)indices[p]*n+col]:__float2bfloat16_rn(0.f);
  }
  __syncthreads();
  wm::fragment<wm::matrix_a,16,16,16,__nv_bfloat16,wm::row_major> af;
  wm::fragment<wm::matrix_b,16,16,16,__nv_bfloat16,wm::row_major> bf;
  wm::load_matrix_sync(af,a,16);
  wm::load_matrix_sync(bf,b+warp*16,BN);
  wm::mma_sync(acc,af,bf,acc);
  __syncthreads();
 }
 wm::store_matrix_sync(result+warp*16,acc,BN,wm::mem_row_major);
 __syncthreads();
 for(int i=tid;i<GM*BN;i+=THREADS) {
  int row=first_row+i/BN,col=first_col+i%BN;
  if(row<m && col<n)out[(long long)row*n+col]=__float2bfloat16_rn(result[i]+__bfloat162float(bias[col]));
 }
}

template<int V,bool NO_SKIP>
__global__ void ballot_rows(const __nv_bfloat16* x,const __nv_bfloat16* w,
 const __nv_bfloat16* bias,__nv_bfloat16* out,int* observed,int m,int k,int n,float threshold) {
 int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);
 if(row>=m)return;
 int col=blockIdx.y*32*V+lane*V,nnz=0;
 float acc[V]={};
 for(int start=0;start<k;start+=32) {
  float value=__bfloat162float(x[(long long)row*k+start+lane]);
  value=value>=threshold?value:0.f;
  unsigned active=__ballot_sync(0xffffffff,value!=0.f);
  nnz+=__popc(active);
  unsigned todo=NO_SKIP?0xffffffff:active;
  while(todo) {
   int source=__ffs(todo)-1;
   float v=__shfl_sync(0xffffffff,value,source);
   int feature=start+source;
   #pragma unroll
   for(int q=0;q<V;q++)if(col+q<n)acc[q]=fmaf(v,__bfloat162float(w[(long long)feature*n+col+q]),acc[q]);
   todo &= todo-1;
  }
 }
 if(lane==0)observed[row*gridDim.y+blockIdx.y]=nnz;
 #pragma unroll
 for(int q=0;q<V;q++)if(col+q<n)out[(long long)row*n+col+q]=__float2bfloat16_rn(acc[q]+__bfloat162float(bias[col+q]));
}

void check(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,torch::Tensor counts) {
 for(auto t:{x,w,bias,out,counts}) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(),"contiguous CUDA tensors required");
  TORCH_CHECK(t.device()==x.device(),"one device required");
 }
 for(auto t:{x,w,bias,out})TORCH_CHECK(t.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(counts.scalar_type()==at::kInt,"int32 counts required");
 TORCH_CHECK(x.dim()==2 && w.dim()==2 && out.dim()==2,"matrix operands required");
 TORCH_CHECK(x.size(1)>0 && x.size(1)<=2048 && x.size(1)%32==0,"K must be 32-aligned and <=2048");
 TORCH_CHECK(w.size(0)==x.size(1) && out.size(0)==x.size(0) && out.size(1)==w.size(1) && bias.numel()==w.size(1),"shape mismatch");
}

void r052_gather(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,torch::Tensor counts,double threshold,int gm,int bn,bool no_skip) {
 check(x,w,bias,out,counts);
 int m=x.size(0),k=x.size(1),n=w.size(1);
 TORCH_CHECK((gm==4 || gm==8 || gm==16) && (bn==64 || bn==128),"unsupported shape");
 TORCH_CHECK(counts.dim()==2 && counts.size(0)==(m+gm-1)/gm && counts.size(1)==(n+bn-1)/bn,"count shape");
 auto stream=at::cuda::getCurrentCUDAStream();
 #define G(M,N,S) gathered<M,N,S><<<dim3((m+M-1)/M,(n+N-1)/N),N*2,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),counts.data_ptr<int>(),m,k,n,threshold)
 #define GS(M,N) if(no_skip){G(M,N,true);}else{G(M,N,false);}
 if(gm==4 && bn==64){GS(4,64);}else if(gm==4 && bn==128){GS(4,128);}
 else if(gm==8 && bn==64){GS(8,64);}else if(gm==8 && bn==128){GS(8,128);}
 else if(gm==16 && bn==64){GS(16,64);}else if(gm==16 && bn==128){GS(16,128);}
 else TORCH_CHECK(false,"unsupported shape");
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void r052_ballot(torch::Tensor x,torch::Tensor w,torch::Tensor bias,torch::Tensor out,torch::Tensor counts,double threshold,int v,bool no_skip) {
 check(x,w,bias,out,counts);
 int m=x.size(0),k=x.size(1),n=w.size(1);
 TORCH_CHECK(v==4 || v==8,"vector width");
 TORCH_CHECK(counts.dim()==2 && counts.size(0)==m && counts.size(1)==(n+32*v-1)/(32*v),"count shape");
 auto stream=at::cuda::getCurrentCUDAStream();
 #define B(V,S) ballot_rows<V,S><<<dim3((m+3)/4,(n+32*V-1)/(32*V)),128,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),counts.data_ptr<int>(),m,k,n,threshold)
 if(v==4){if(no_skip){B(4,true);}else{B(4,false);}}
 else{if(no_skip){B(8,true);}else{B(8,false);}}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gather",&r052_gather);m.def("ballot",&r052_ballot);}
