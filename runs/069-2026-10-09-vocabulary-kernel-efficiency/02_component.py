"""Qualified arithmetic, independent row masks, counters and graph replay."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
from common import RUN, config, runtime, support, write
from kernel import extension, original
from measurement import Graph, compare


def resource_reports(old,new):
    # Inspect the actual loaded binaries, including the unchanged legacy control.
    # Local-memory allocation alone does not establish a spill/stall bottleneck.
    tool=shutil.which('cuobjdump') or str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda-12.8'))/'bin/cuobjdump')
    resources=[]
    for name,module in (('legacy',old),('variants',new)):
        binary=Path(module.__file__)
        with binary.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        try:
            process=subprocess.run([tool,'--dump-resource-usage',str(binary)],capture_output=True,text=True,check=False,timeout=60)
            code=process.returncode;output=process.stdout+process.stderr
        except (OSError,subprocess.TimeoutExpired) as exc:
            code=None;output=type(exc).__name__+': '+str(exc)
        resources.append({'implementation':name,'binary':str(binary),'binary_sha256':digest,
                          'returncode':code,'resource_usage':output})
    return resources


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    args=parser.parse_args();cfg=config();runtime(cfg['runtime_seed'])
    started=time.monotonic();rows=[]
    old=original.extension();new=extension()
    with torch.inference_mode():
        for kappa in (0.1,0.5):
            for case in ('random','dead','live','independent','full_vocabulary'):
                m,v=64,50304 if case=='full_vocabulary' else 1032
                raw=torch.randn(m,128,device='cuda',dtype=torch.float32)
                if case=='dead':raw[::16,::16]=0
                if case=='live':raw[::16,::16]=kappa+1
                if case=='independent':
                    raw[::16,::16]=kappa+1
                    raw[::32,::16]=0  # adjacent M16 masks differ maximally
                s=support(raw,kappa)
                live=raw[::16,::16]>kappa
                x=(raw.reshape(m//16,16,8,16)*live[:,None,:,None]).reshape(m,128).half()
                w=torch.randn(v,128,device='cuda',dtype=torch.float16)
                packed=original.pack_weight(w)
                reference=torch.nn.functional.linear(x,w)
                for variant in cfg['variants']:
                    n=variant['n'];columns=math.ceil(v/n)
                    out=torch.empty(m,v,device='cuda',dtype=torch.float16)
                    stats=torch.empty(m//16,columns,2,device='cuda',dtype=torch.int64)
                    for skip in (False,True):
                        def operation(xx,ss,count=False):
                            if variant['family']=='legacy':old.project(xx,ss,packed,out,stats,skip,count,16,n)
                            else:new.project(xx,ss,packed,out,stats,skip,count,n,variant['groups'],variant['family']=='direct',variant['family']=='prefetch')
                            return out
                        actual=operation(x,s,True)
                        check=compare(reference,actual,cfg['numerical_bounds'])
                        hits=stats.sum((0,1)).tolist()
                        active=int(live.sum()) if skip else m//16*8
                        expected=active*columns*(n//8)
                        potential=m//16*8*columns*(n//8)
                        if not check['pass'] or hits != [expected,potential-expected]:
                            raise ValueError('Arithmetic or independent-mask counter failure: '+str((variant,case,check,hits)))
                        graph=Graph(operation,[x,s]);graph.stage([x,s])
                        replay=compare(reference,graph(),cfg['numerical_bounds'])
                        if not replay['pass']:raise ValueError('Graph replay failure')
                        rows.append({'kappa':kappa,'case':case,'variant':variant['name'],'skip':skip,
                                     'check':check,'replay':replay,'issued':hits[0],'potential':potential})
                        del graph
    report={'status':'pass','gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,
            'elapsed_seconds':time.monotonic()-started,'checks':len(rows),'rows':rows}
    write(RUN/f'prelaunch/components-{args.tag}.json',report)
    write(RUN/f'prelaunch/resources-{args.tag}.json',resource_reports(old,new))
    print({k:v for k,v in report.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
