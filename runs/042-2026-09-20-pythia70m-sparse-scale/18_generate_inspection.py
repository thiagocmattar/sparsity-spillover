"""Inspect each activation row once, then reuse it across output-column blocks."""
import re
import subprocess
import sys
from io_utils import RUN, read, write, record


def once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)


def main():
    for identifier,origin in [('opt012','opt001'),('opt013','opt002'),('opt014','opt006')]:
        source=RUN/'candidates'/origin
        folder=RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        spec=read(source/'spec.json')
        cu=(source/'joint.cu').read_text()
        boundary='__device__ __forceinline__ bf16 short_linear'
        kernel='''static_assert(sizeof(TinyRow)==36,"Workspace byte shape");
__global__ void inspect_rows(const bf16* h,const bf16* z,TinyRow* summaries,int m,
                            float th,float tz,bool gh,bool gz){
    int row=blockIdx.x*8+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=m)return;
    auto ha=inspect<2048>(h,row,lane,th,gh);
    auto za=inspect<512>(z,row,lane,tz,gz);
    if(lane==0){summaries[row*2]=ha;summaries[row*2+1]=za;}
}
'''
        cu=once(cu,boundary,kernel+boundary)
        cu=once(cu,'bf16* out,long long* stats,float th',
                'bf16* out,long long* stats,const TinyRow* summaries,float th')
        cu=once(cu,'auto ha=inspect<2048>(h,row,lane,th,gh);auto za=inspect<512>(z,row,lane,tz,gz);',
                'auto ha=summaries[row*2];auto za=summaries[row*2+1];')
        cu=once(cu,'torch::Tensor out,torch::Tensor stats,',
                'torch::Tensor out,torch::Tensor stats,torch::Tensor summaries,')
        cu=once(cu,'#define LAUNCH(S,C)',
            'TORCH_CHECK(summaries.device()==h.device() && summaries.scalar_type()==at::kByte && summaries.is_contiguous() && summaries.numel()==m*2*36,"Inspection workspace shape");\n'
            '    if(skip && fast_weights)inspect_rows<<<(m+7)/8,256,0,stream>>>(P(h),P(z),reinterpret_cast<TinyRow*>(summaries.data_ptr()),m,th,tz,gh,gz);\n'
            '    #define LAUNCH(S,C)')
        cu=once(cu,'reinterpret_cast<long long*>(stats.data_ptr()),th,tz',
                'reinterpret_cast<long long*>(stats.data_ptr()),reinterpret_cast<const TinyRow*>(summaries.data_ptr()),th,tz')
        (folder/'joint.cu').write_text(cu,newline='\n')
        py=(source/'joint.py').read_text().replace('run042_'+origin,'run042_'+identifier)
        py=once(py,'self.out=torch.empty_like(r);',
                'self.summaries=torch.empty((r.shape[0],2,36),device=r.device,dtype=torch.uint8);self.out=torch.empty_like(r);')
        py=once(py,'r,self.out,self.stats,','r,self.out,self.stats,self.summaries,')
        (folder/'joint.py').write_text(py,newline='\n')
        candidate=(source/'candidate.py').read_text().replace(origin,identifier)
        candidate=candidate.replace("'sparse_sites':['h','z']", "'inspection':'One fresh scan per row, reused across column blocks','sparse_sites':['h','z']")
        (folder/'candidate.py').write_text(candidate,newline='\n')
        spec.update(kind='reused-inspection',origin_candidate=origin,
                    origin_manifest=record(source/'manifest.json'),
                    timed_work='Inspection launch, workspace writes/reads and sparse projection all included')
        write(folder/'spec.json',spec)
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__=='__main__': main()
