"""Separate the common all-short-row path from the matrix-instruction fallback."""
import subprocess
import sys
from io_utils import RUN,read,write,record


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


FAST=r'''
template<bool Count>
__global__ void scalar_only(const bf16* h,const bf16* z,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,long long* stats,
    float th,float tz,bool gh,bool gz,unsigned char* fallback){
    __shared__ int any_complex;
    int lane=threadIdx.x&31,warp=threadIdx.x/32,row=blockIdx.x*8+warp;
    if(threadIdx.x==0)any_complex=0;
    __syncthreads();
    auto ha=inspect<2048>(h,row,lane,th,gh);
    auto za=inspect<512>(z,row,lane,tz,gz);
    if(lane==0 && (ha.count>SHORT_LIMIT || za.count>SHORT_LIMIT))atomicOr(&any_complex,1);
    __syncthreads();
    if(threadIdx.x==0)fallback[blockIdx.x*2+blockIdx.y]=any_complex;
    // No outputs or counts are written for fallback blocks: work is not duplicated.
    if(any_complex)return;
    #pragma unroll
    for(int e=0;e<8;++e){
        int col=blockIdx.y*256+e*32+lane;
        bf16 hv=short_linear(ha,wht,bh,col),zv=short_linear(za,wzt,bz,col);
        float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
        out[row*512+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*512+col]));
    }
    counters<Count>(stats,warp,lane,0,512,0,128,ha.count*256,za.count*256);
}
'''


def main():
    for identifier,parent,limit in [('opt029','opt019',2),('opt030','opt021',8)]:
        source=RUN/'candidates'/parent;folder=RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        cu=(source/'joint.cu').read_text()
        cu=once(cu,'template<bool Skip,bool Count>\n__global__ void joint(',FAST.replace('SHORT_LIMIT',str(limit))+'\ntemplate<bool Skip,bool Count>\n__global__ void joint(')
        cu=once(cu,'bool gh,bool gz,bool fast_weights){\n    __shared__',
                'bool gh,bool gz,bool fast_weights,const unsigned char* fallback){\n    if(fallback && !fallback[blockIdx.x*2+blockIdx.y])return;\n    __shared__')
        cu=once(cu,'double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count){',
                'double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count,torch::Tensor fallback){')
        cu=once(cu,'c10::cuda::CUDAGuard guard(h.device());',
                'TORCH_CHECK(fallback.device()==h.device() && fallback.scalar_type()==at::kByte && fallback.is_contiguous() && fallback.numel()==m/8*2,"Fallback flags");\n    c10::cuda::CUDAGuard guard(h.device());')
        line=next(x for x in cu.splitlines() if '#define LAUNCH(S,C)' in x)
        replacement='''    #define LAUNCH(S,C) do { \\
      unsigned char* flags=(S && fast_weights)?fallback.data_ptr<unsigned char>():nullptr; \\
      if(flags)scalar_only<C><<<dim3(m/8,2),256,0,stream>>>(P(h),P(z),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,gh,gz,flags); \\
      joint<S,C><<<dim3(m/8,2),256,0,stream>>>(P(h),P(z),P(wh),P(wz),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,gh,gz,fast_weights,flags); \\
    } while(0)'''
        cu=once(cu,line,replacement)
        (folder/'joint.cu').write_text(cu,newline='\n')
        py=(source/'joint.py').read_text().replace(parent,identifier)
        py=once(py,'self.out=torch.empty_like(r);','self.flags=torch.empty((r.shape[0]//8,2),device=r.device,dtype=torch.uint8);self.out=torch.empty_like(r);')
        py=once(py,'self.skip,self.fast_weights,self.count)','self.skip,self.fast_weights,self.count,self.flags)')
        (folder/'joint.py').write_text(py,newline='\n')
        candidate=(source/'candidate.py').read_text().replace(parent,identifier)
        candidate=once(candidate,"'sparse_sites':['h','z']","'sparse_sites':['h','z'],'split_scalar_kernel':True")
        (folder/'candidate.py').write_text(candidate,newline='\n')
        spec=read(source/'spec.json')
        spec.update(kind='split-short-rows',short_limit=limit,origin_manifest=record(source/'manifest.json'),
            hypothesis='A small scalar-only kernel avoids the registers/shared-memory required by rare MMA fallbacks; extra launches and repeat inspection may erase gains.',
            work_accounting='All-short blocks write complete outputs/counters; remaining blocks write nothing before the unchanged fallback. Fresh flags and both launches are timed.')
        write(folder/'spec.json',spec)
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)


if __name__=='__main__':main()
