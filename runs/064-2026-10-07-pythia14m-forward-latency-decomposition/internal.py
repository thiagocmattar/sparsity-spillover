"""SM-local timestamp distributions; not an additive whole-device budget."""
from functools import lru_cache
from pathlib import Path
import hashlib
import torch

ROOT=Path(__file__).parent

@lru_cache(None)
def extension(specialized):
    from torch.utils.cpp_extension import load
    path=ROOT/('clocked_joint.cu' if specialized else 'clocked_generic.cu')
    return load(name='run064_clock_'+hashlib.sha256(path.read_bytes()).hexdigest()[:12],sources=[str(path)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


def probe(joint,h,z,r,repeats):
    import kernel
    flat=[v.reshape(-1,v.shape[-1]).contiguous() for v in [h,z,r]];h,z,r=flat
    out=torch.empty_like(r);stats=torch.empty_like(joint.stats);ticks=torch.empty((r.shape[0]//8,4,5),device=r.device,dtype=torch.int64)
    args=[h,z,joint.w2.weight,joint.wo.weight,joint.wht,joint.wzt,joint.w2.bias,joint.wo.bias,r,out,stats,
          joint.th,joint.tz,joint.gh,joint.gz,joint.skip,joint.fast_weights,False]
    base=kernel.extension(joint.specialized).forward;clocked=extension(joint.specialized).forward
    base(*args);reference=out.clone();clocked(*args,ticks)
    if not torch.equal(reference,out):raise ValueError('Clock instrumentation changed projection output')
    cycles=[];latencies=[]
    for repeat in range(repeats):
        pair={}
        for name in (['base','clocked'] if repeat%2==0 else ['clocked','base']):
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            first.record();(base(*args) if name=='base' else clocked(*args,ticks));last.record();last.synchronize()
            pair[name]=first.elapsed_time(last)*1000
            if name=='clocked':cycles.append((ticks[:,:,1:]-ticks[:,:,:-1]).cpu().tolist())
        latencies.append(pair)
    gh=kernel.gated_value(h,joint.gh,joint.th);gz=kernel.gated_value(z,joint.gz,joint.tz)
    densities={}
    for name,value in [('h',gh),('z',gz)]:
        tiles=value.reshape(-1,8,value.shape[-1]//16,16)
        densities[name]={'tiles':tiles.shape[0]*tiles.shape[2],'empty_tiles':int((tiles==0).all(dim=1).all(dim=-1).sum()),'elements':value.numel(),'exact_zeros':int((value==0).sum())}
    return {'stage_names':['support_and_prepare','h_accumulate','z_accumulate','round_bias_residual_store'],
            'unit':'SM clock cycles per projection warp; overlapping CTA/warp work, not wall time',
            'cycles':cycles,'kernel_us':latencies,'exact_output':True,'density':densities}
