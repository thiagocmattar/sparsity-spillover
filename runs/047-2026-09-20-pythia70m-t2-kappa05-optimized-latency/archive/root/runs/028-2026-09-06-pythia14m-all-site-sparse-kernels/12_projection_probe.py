"""Isolated synthetic K021 projection accuracy and skip-control measurements."""
import argparse
import json
import time
import traceback
import numpy as np
import torch
from common import RUN,module,runtime,record,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[];samples=[]
    try:
        runtime(torch)
        candidate=module('run028_projection_probe',RUN/'candidates/k021/candidate.py')
        backend=candidate.extension()
        write_json(dest/'manifest.json',{'sources':[record(RUN/name) for name in ['12_projection_probe.py','common.py','candidates/k021/candidate.py','candidates/k021/projection.cu']],
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'measurement':'synthetic projections; not trained-model evidence'})
        with torch.inference_mode():
            for n in [384,512]:
                w=torch.randn((n,128),device='cuda',dtype=torch.bfloat16)*.1
                wt=w.T.contiguous();b=torch.randn(n,device='cuda',dtype=torch.bfloat16)*.1
                for zero_fraction in [0.,.5,.7,.9,.99]:
                    x=torch.randn((2048,128),device='cuda',dtype=torch.bfloat16)
                    x[torch.rand(x.shape,device=x.device)<zero_fraction]=0
                    out=torch.empty((2048,n),device=x.device,dtype=x.dtype)
                    ref=torch.nn.functional.linear(x,w,b)
                    backend.forward(x,wt,b,out,True);sparse=out.clone()
                    backend.forward(x,wt,b,out,False)
                    delta=sparse.float()-ref.float()
                    row={'n':n,'requested_zero_fraction':zero_fraction,'zeros':int((x==0).sum()),'elements':x.numel(),
                        'skip_toggle_bitwise_equal':torch.equal(sparse,out),'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/ref.float().norm()),'finite':bool(torch.isfinite(sparse).all())}
                    rows.append(row)
                    calls={'native':lambda:torch.nn.functional.linear(x,w,b),'skip':lambda:backend.forward(x,wt,b,out,True),'no_skip':lambda:backend.forward(x,wt,b,out,False)}
                    for fn in calls.values():
                        for _ in range(3):fn()
                    rng=np.random.default_rng(2812)
                    for repeat in range(7):
                        for mode in rng.permutation(list(calls)):
                            torch.cuda.synchronize();before=time.perf_counter();calls[mode]();torch.cuda.synchronize()
                            samples.append({'n':n,'zero_fraction':zero_fraction,'repeat':repeat,'mode':str(mode),'host_ms':1000*(time.perf_counter()-before)})
                    write_json(dest/'accuracy.json',rows);write_json(dest/'timing.json',samples)
                    print(json.dumps(row),flush=True)
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
