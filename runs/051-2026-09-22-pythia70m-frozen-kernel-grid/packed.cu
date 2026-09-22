// Run050 lossless packed-row streaming. Inspired by TwELL's public MIT format;
// independently implemented with worst-case storage and FP32 products.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

template<int V,int ROWS> __global__ void consume(const unsigned* p,const int* counts,
 const __nv_bfloat16* w,const __nv_bfloat16* bias,__nv_bfloat16* out,int m,int k,int n) {
 int lane=threadIdx.x&31,row=blockIdx.x*ROWS+(threadIdx.x>>5);
 if(row>=m)return;
 int col=(blockIdx.y*32+lane)*V;
 float acc[V]={};
 for(int t=0;t<k/256;t++) {
  int count=counts[row*(k/256)+t];
  for(int start=0;start<count;start+=32) {
   unsigned item= start+lane<count ? p[(long long)row*k+t*256+start+lane]:0;
   for(int j=0;j<min(32,count-start);j++) {
    unsigned packed=__shfl_sync(0xffffffff,item,j);
    int idx=packed&65535;
    float v=__bfloat162float(__ushort_as_bfloat16(packed>>16));
    __nv_bfloat16 values[V];
    #pragma unroll
    for(int q=0;q<V/4;q++) {
     if(col+q*4+3<n) reinterpret_cast<uint2*>(values)[q]=*reinterpret_cast<const uint2*>(w+(long long)idx*n+col+q*4);
    }
    #pragma unroll
    for(int o=0;o<V;o++) if(col+o<n) acc[o]=fmaf(v,__bfloat162float(values[o]),acc[o]);
   }
  }
 }
 #pragma unroll
 for(int o=0;o<V;o++) if(col+o<n)out[(long long)row*n+col+o]=__float2bfloat16_rn(acc[o]+__bfloat162float(bias[col+o]));
}
void launch(torch::Tensor p,torch::Tensor counts,torch::Tensor w,torch::Tensor bias,torch::Tensor out,int v,int rows){
 TORCH_CHECK(p.is_cuda()&&w.is_cuda()&&out.is_cuda(),"CUDA required");
 TORCH_CHECK(p.scalar_type()==at::kInt&&counts.scalar_type()==at::kInt,"integer packing");
 TORCH_CHECK(w.scalar_type()==at::kBFloat16&&bias.scalar_type()==at::kBFloat16&&out.scalar_type()==at::kBFloat16,"BF16 required");
 TORCH_CHECK(p.is_contiguous()&&counts.is_contiguous()&&w.is_contiguous()&&out.is_contiguous(),"contiguous required");
 int m=p.size(0),k=p.size(1),n=w.size(1);
 TORCH_CHECK(k%256==0&&k<=65536&&w.size(0)==k&&out.size(0)==m&&out.size(1)==n,"shape contract");
 auto stream=at::cuda::getCurrentCUDAStream();
 #define RUN(V,R) consume<V,R><<<dim3((m+R-1)/R,(n+32*V-1)/(32*V)),32*R,0,stream>>>(reinterpret_cast<unsigned*>(p.data_ptr()),counts.data_ptr<int>(),reinterpret_cast<__nv_bfloat16*>(w.data_ptr()),reinterpret_cast<__nv_bfloat16*>(bias.data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),m,k,n)
 if(rows==2){if(v==4){RUN(4,2);}else if(v==8){RUN(8,2);}else{TORCH_CHECK(v==16);RUN(16,2);}}
 else if(rows==4){if(v==4){RUN(4,4);}else if(v==8){RUN(8,4);}else{TORCH_CHECK(v==16);RUN(16,4);}}
 else{TORCH_CHECK(rows==8&&v==4);RUN(4,8);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("consume",&launch);}
