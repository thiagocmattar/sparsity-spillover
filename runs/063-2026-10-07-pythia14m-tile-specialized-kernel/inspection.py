"""Untimed model capture and standalone hot-resident support cost diagnostic."""
from functools import lru_cache
import os
import random
import torch
from io_utils import RUN,sha,write

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    return load(name='run063_inspection_'+sha(RUN/'inspection.cu')[:12],sources=[str(RUN/'inspection.cu')],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)

def collect(model,ids,dest):
    import replay
    operands=[];originals=[]
    class Capture:
        def __init__(self,original,index):self.original,self.index=original,index
        def __call__(self,h,z,residual):
            operands.extend([(self.index,'h',h.reshape(-1,512).contiguous().clone(),self.original.th),
                             (self.index,'z',z.reshape(-1,128).contiguous().clone(),self.original.tz)])
            return self.original(h,z,residual)
    try:
        for i,layer in enumerate(model.gpt_neox.layers):
            originals.append((layer,layer._run026_joint));layer._run026_joint=Capture(layer._run026_joint,i)
        with torch.inference_mode():replay.scaffold.forward(model,ids)
    finally:
        for layer,joint in originals:layer._run026_joint=joint
    ext=extension();rows=[];rng=random.Random(2504)
    with torch.inference_mode():
        for layer,site,x,threshold in operands:
            buffers={mode:torch.empty(x.shape[0],device=x.device,dtype=torch.int32) for mode in ['generic','corner']}
            graphs={}
            for mode,out in buffers.items():
                for _ in range(5):ext.inspect_only(x,out,threshold,mode=='generic')
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(100):ext.inspect_only(x,out,threshold,mode=='generic')
                graphs[mode]=graph
            samples=[]
            for repeat in range(10):
                order=['generic','corner'];rng.shuffle(order)
                for mode in order:
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();graphs[mode].replay();b.record();b.synchronize()
                    samples.append({'repeat':repeat,'mode':mode,'microseconds_per_call':a.elapsed_time(b)*1000/100})
            # Each corner bit must equal the OR of independently inspected rows.
            a=buffers['generic'].reshape(-1,8).to(torch.int64)
            pooled=a[:,0].clone()
            for j in range(1,8):pooled.bitwise_or_(a[:,j])
            b=buffers['corner'].reshape(-1,8).to(torch.int64)
            if not torch.equal(b,pooled[:,None].expand_as(b)):raise ValueError('Standalone support masks disagree')
            rows.append({'layer':layer,'site':site,'shape':list(x.shape),'samples':samples,'mask_OR_equal':True})
    write(dest,{'status':'complete','rows':rows,'coverage':'one fixed timing input; six layers and both h/z operands',
                'method':'CUDA-event graph of 100 standalone support calls, ten randomized paired repeats, both launches 256 threads',
                'limits':'hot-resident component diagnostic; generic per-row support OR equals corner support; no packing/projection/full-model cost; not additive to end-to-end latency'})
