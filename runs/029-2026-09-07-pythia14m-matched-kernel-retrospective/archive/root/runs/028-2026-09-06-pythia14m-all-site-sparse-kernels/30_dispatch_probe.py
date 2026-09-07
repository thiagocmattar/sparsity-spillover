"""Record actual CUDA dispatch names rather than infer them from API names."""
import argparse
import shutil
import time
import traceback
import torch
from common import RUN, module, runtime, record, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch)
        source=RUN/'30_dispatch_probe.py';shutil.copyfile(source,dest/source.name)
        write_json(dest/'manifest.json',{'source':record(source),'torch':torch.__version__,'torch_git':torch.version.git_version,
            'gpu':torch.cuda.get_device_name(),'scope':'synthetic CUDA dispatch diagnostic, not performance estimates'})
        with torch.inference_mode():
            candidate=module('run028_dispatch_candidate',RUN/'candidates/k026/candidate.py')
            fns={'native':lambda q,k,v:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=32**-.5),
                 'k026_dense':candidate.Attention(False),'k026_sparse':candidate.Attention(True)}
            for t in [129,2048]:
                q,k,v=[torch.randn((1,4,t,32),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                for name,fn in fns.items():
                    call=(lambda:fn(q,k,v)) if name=='native' else (lambda:fn(q,k,v,32**-.5))
                    call();torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                        call();torch.cuda.synchronize()
                    prof.export_chrome_trace(str(dest/f'{name}-{t}.json'))
                    events=[{'name':event.name,'device_type':str(event.device_type),'device_time_total':event.device_time_total} for event in prof.events() if 'CUDA' in str(event.device_type)]
                    rows.append({'mode':name,'length':t,'cuda_events':events});write_json(dest/'dispatch.json',rows)
                    print(rows[-1],flush=True)
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
