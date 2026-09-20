"""Inspect h and z together to reduce block barriers; preserve scalar arithmetic."""
from io_utils import RUN, write, record


PARALLEL = r'''
template<int Threads>
__global__ void parallel_rows(const bf16* h,const bf16* z,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,
    float th,float tz,bool gh,bool gz,int* row_counts){
    __shared__ int counts[2][Threads/32],indices[2][8],cursor[2];
    __shared__ TinyRow summary[2];
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid/32;
    int nh=0,nz=0;
    #pragma unroll
    for(int base=0;base<2048;base+=Threads){
        float value=gated(h,row*2048+base+tid,th,gh);
        if(value!=0.f)nh+=(!(fabsf(value)>=0x1p-50f && fabsf(value)<=0x1p50f))?9:1;
    }
    #pragma unroll
    for(int base=0;base<512;base+=Threads){
        float value=gated(z,row*512+base+tid,tz,gz);
        if(value!=0.f)nz+=(!(fabsf(value)>=0x1p-50f && fabsf(value)<=0x1p50f))?9:1;
    }
    #pragma unroll
    for(int offset=16;offset>0;offset/=2){
        nh+=__shfl_down_sync(0xffffffffu,nh,offset);
        nz+=__shfl_down_sync(0xffffffffu,nz,offset);
    }
    if(lane==0){counts[0][warp]=nh;counts[1][warp]=nz;}
    __syncthreads();
    if(tid==0){
        int total_h=0,total_z=0;
        #pragma unroll
        for(int w=0;w<Threads/32;++w){total_h+=counts[0][w];total_z+=counts[1][w];}
        summary[0].count=min(total_h,9);summary[1].count=min(total_z,9);
        row_counts[row*2]=summary[0].count;row_counts[row*2+1]=summary[1].count;
        cursor[0]=cursor[1]=0;
    }
    __syncthreads();
    if(summary[0].count>8 || summary[1].count>8)return;
    #pragma unroll
    for(int base=0;base<2048;base+=Threads){
        int index=base+tid;float value=gated(h,row*2048+index,th,gh);
        if(value!=0.f)indices[0][atomicAdd(cursor,1)]=index;
    }
    #pragma unroll
    for(int base=0;base<512;base+=Threads){
        int index=base+tid;float value=gated(z,row*512+index,tz,gz);
        if(value!=0.f)indices[1][atomicAdd(cursor+1,1)]=index;
    }
    __syncthreads();
    if(tid<2){
        for(int i=1;i<summary[tid].count;++i){
            int value=indices[tid][i],j=i-1;
            while(j>=0 && indices[tid][j]>value){indices[tid][j+1]=indices[tid][j];--j;}
            indices[tid][j+1]=value;
        }
        for(int i=0;i<summary[tid].count;++i){
            int index=indices[tid][i];summary[tid].indices[i]=index;
            summary[tid].values[i]=tid==0?gated(h,row*2048+index,th,gh):gated(z,row*512+index,tz,gz);
        }
    }
    __syncthreads();
    TinyRow ha=summary[0],za=summary[1];
    for(int col=tid;col<512;col+=Threads){
        bf16 hv=short_linear(ha,wht,bh,col),zv=short_linear(za,wzt,bz,col);
        float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
        out[row*512+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*512+col]));
    }
}
'''


def main():
    source = RUN/'candidates/opt063'
    for i,threads in enumerate((128,64,256)):
        identifier = f'opt{70+i:03d}'
        folder = RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        cu = (source/'joint.cu').read_text()
        start = cu.index('template<int K,int Threads>')
        end = cu.index('template<bool Skip,bool Count>\n__global__ void joint',start)
        cu = cu[:start] + PARALLEL + '\n' + cu[end:]
        assert 'parallel_rows<128><<<m,128' in cu
        cu = cu.replace('parallel_rows<128><<<m,128',f'parallel_rows<{threads}><<<m,{threads}')
        (folder/'joint.cu').write_text(cu.rstrip()+'\n',newline='\n')
        py = (source/'joint.py').read_text().replace('run042_opt063_',f'run042_{identifier}_')
        (folder/'joint.py').write_text(py.rstrip()+'\n',newline='\n')
        code = (source/'candidate.py').read_text().replace('opt063',identifier).replace('opt032','opt064')
        code = code.replace("'parallel_inspection_threads':128",f"'parallel_inspection_threads':{threads}")
        (folder/'candidate.py').write_text(code.rstrip()+'\n',newline='\n')
        write(folder/'spec.json', {'M':8,'N':256,'K':16,'short_limit':8,
              'kind':'joint-hz-row-inspection','threads':threads,
              'parent_manifest':record(RUN/'candidates/opt064/manifest.json'),
              'source_joint':record(source/'manifest.json'),
              'hypothesis':'Inspect h and z together with four block barriers; skip index gathering when either row needs the unchanged MMA fallback.',
              'constraints':'Same ascending-index FP32 scalar arithmetic and BF16 rounding; same actual work counters including mixed-group duplicate work.'})
        write(folder/'manifest.json', {'candidate':identifier,
              'files':[record(p) for p in sorted(folder.glob('*'))],
              'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
