"""Parallel per-row inspection followed by the unchanged rare-row MMA fallback."""
from io_utils import RUN, write, record

FAST = r'''
template<int K,int Threads>
__device__ __forceinline__ TinyRow parallel_inspect(const bf16* x,int row,float threshold,bool gate,
    int* counts,int* first,int* second,TinyRow* summary){
    int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    int count=0,i0=K,i1=K;
    #pragma unroll
    for(int base=0;base<K;base+=Threads){
        int index=base+tid;float value=gated(x,row*K+index,threshold,gate);
        if(value!=0.f){
            count+=(!(fabsf(value)>=0x1p-50f && fabsf(value)<=0x1p50f))?3:1;
            if(i0==K)i0=index;else if(i1==K)i1=index;
        }
    }
    int c=count,f=i0;
    #pragma unroll
    for(int offset=16;offset>0;offset/=2){
        c+=__shfl_down_sync(0xffffffffu,c,offset);
        f=min(f,__shfl_down_sync(0xffffffffu,f,offset));
    }
    int warp_first=__shfl_sync(0xffffffffu,f,0);
    int s=i0==warp_first?i1:i0;
    #pragma unroll
    for(int offset=16;offset>0;offset/=2)s=min(s,__shfl_down_sync(0xffffffffu,s,offset));
    if(lane==0){counts[warp]=c;first[warp]=f;second[warp]=s;}
    __syncthreads();
    if(tid==0){
        TinyRow out;out.count=0;int a=K,b=K;
        #pragma unroll
        for(int w=0;w<Threads/32;++w){out.count+=counts[w];a=min(a,first[w]);}
        #pragma unroll
        for(int w=0;w<Threads/32;++w)b=min(b,first[w]==a?second[w]:first[w]);
        out.count=min(out.count,3);
        if(out.count>0 && out.count<=2){out.i0=a;out.v0=gated(x,row*K+a,threshold,gate);}
        if(out.count==2){out.i1=b;out.v1=gated(x,row*K+b,threshold,gate);}
        *summary=out;
    }
    __syncthreads();
    TinyRow out=*summary;
    __syncthreads();
    return out;
}

template<int Threads>
__global__ void parallel_rows(const bf16* h,const bf16* z,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,
    float th,float tz,bool gh,bool gz,int* row_counts){
    __shared__ int counts[Threads/32],first[Threads/32],second[Threads/32];
    __shared__ TinyRow summary;
    int row=blockIdx.x;
    TinyRow ha=parallel_inspect<2048,Threads>(h,row,th,gh,counts,first,second,&summary);
    TinyRow za=parallel_inspect<512,Threads>(z,row,tz,gz,counts,first,second,&summary);
    if(threadIdx.x==0){row_counts[row*2]=ha.count;row_counts[row*2+1]=za.count;}
    if(ha.count>2 || za.count>2)return;
    for(int col=threadIdx.x;col<512;col+=Threads){
        bf16 hv=short_linear(ha,wht,bh,col),zv=short_linear(za,wzt,bz,col);
        float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
        out[row*512+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*512+col]));
    }
}
'''


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    origin = RUN/'candidates/opt019'
    for identifier, threads in [('opt041',64), ('opt042',128), ('opt043',256)]:
        folder = RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        cu = (origin/'joint.cu').read_text()
        cu = once(cu, 'template<bool Skip,bool Count>\n__global__ void joint', FAST + '\ntemplate<bool Skip,bool Count>\n__global__ void joint')
        cu = once(cu, 'float th,float tz,bool gh,bool gz,bool fast_weights){', 'float th,float tz,bool gh,bool gz,bool fast_weights,const int* row_counts){')
        needle = 'int lane=threadIdx.x&31,warp=threadIdx.x/32,first_row=blockIdx.x*8;'
        cu = once(cu, needle, needle + '''
    if(row_counts){
        bool done=true;
        #pragma unroll
        for(int i=0;i<8;++i)done=done && row_counts[(first_row+i)*2]<=2 && row_counts[(first_row+i)*2+1]<=2;
        if(done){
            counters<Count>(stats,warp,lane,0,512,0,128,
                row_counts[(first_row+warp)*2]*256,row_counts[(first_row+warp)*2+1]*256);
            return;
        }
    }
''')
        cu = once(cu, 'double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count){', 'double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count,torch::Tensor row_counts){')
        needle = 'c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();'
        cu = once(cu, needle, '''TORCH_CHECK(row_counts.device()==h.device() && row_counts.scalar_type()==at::kInt && row_counts.is_contiguous() && row_counts.numel()==m*2,"Row count shape");
    ''' + needle)
        start = cu.index('    #define LAUNCH(S,C)')
        end = cu.index('\n    if(skip)', start)
        cu = cu[:start] + f'''    #define LAUNCH(S,C) do {{ \\
      int* rows=(S && fast_weights)?row_counts.data_ptr<int>():nullptr; \\
      if(rows)parallel_rows<{threads}><<<m,{threads},0,stream>>>(P(h),P(z),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz,rows); \\
      joint<S,C><<<dim3(m/8,2),256,0,stream>>>(P(h),P(z),P(wh),P(wz),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,gh,gz,fast_weights,rows); \\
    }} while(0)''' + cu[end:]
        (folder/'joint.cu').write_text(cu, newline='\n')
        py = (origin/'joint.py').read_text().replace('run042_opt019_', 'run042_' + identifier + '_')
        py = once(py, 'self.out=torch.empty_like(r);', 'self.row_counts=torch.empty((r.shape[0],2),device=r.device,dtype=torch.int32);self.out=torch.empty_like(r);')
        py = once(py, 'self.skip,self.fast_weights,self.count)', 'self.skip,self.fast_weights,self.count,self.row_counts)')
        (folder/'joint.py').write_text(py, newline='\n')
        code = f'''"""Parallel sparse-row inspection on the qualified composed parent."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_{identifier}_base',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    joint=module('run042_{identifier}_joint',HERE/'joint.py')
    for layer in model.gpt_neox.layers:layer._run026_joint=joint.Joint(layer._run026_joint)
    return {{**metadata,'identity':'{identifier}','parallel_inspection_threads':{threads},
            'counter_note':'Same M8/N256 conceptual fallback potential; exact issued MMA and scalar counts. Fresh inspection and both launches are timed.'}}
'''
        (folder/'candidate.py').write_text(code, newline='\n')
        write(folder/'spec.json', {'M':8,'N':256,'K':16,'short_limit':2,
              'kind':'parallel-row-inspection','threads':threads,
              'parent_manifest':record(RUN/'candidates/opt032/manifest.json'),
              'source_joint':record(origin/'manifest.json'),
              'hypothesis':'Parallel row scans replace the serial 64-step warp scan on all-short row groups; unchanged fallback handles every other group.',
              'arithmetic':'Same ascending first-two indices, FP32 multiply/FMA, BF16 branch and residual rounding as parent. No input result caching.'})
        write(folder/'manifest.json', {'candidate':identifier,
              'files':[record(p) for p in sorted(folder.glob('*'))],
              'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
