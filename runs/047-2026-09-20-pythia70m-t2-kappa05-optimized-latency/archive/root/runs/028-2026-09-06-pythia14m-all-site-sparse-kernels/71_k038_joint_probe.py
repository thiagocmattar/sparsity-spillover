"""K038 gated h/z and residual primitive check; synthetic, not qualification."""
import argparse
import shutil
import time
import traceback
from types import SimpleNamespace
import torch
from common import RUN,module,runtime,record,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch)
        paths=[RUN/'71_k038_joint_probe.py',RUN/'common.py']+[f for f in (RUN/'candidates/k038').iterdir() if f.is_file()]
        for path in paths:
            target=dest/'source'/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(dest/'manifest.json',{'sources':[record(f) for f in paths],'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'scope':'synthetic K038 gate/linear/residual arithmetic; no full-model claim'})
        candidate=module('run028_joint_tensor_probe',RUN/'candidates/k038/candidate.py')
        with torch.inference_mode():
            candidate.extension()
            wh=torch.nn.Linear(512,128,device='cuda',dtype=torch.bfloat16)
            wz=torch.nn.Linear(128,128,device='cuda',dtype=torch.bfloat16)
            for fraction in [0.,.5,.99,1.]:
                h=torch.randn((1,2048,512),device='cuda',dtype=torch.bfloat16)
                z=torch.randn((1,2048,128),device='cuda',dtype=torch.bfloat16)
                residual=torch.randn_like(z)
                for x in [h,z]:x.masked_fill_(torch.rand_like(x.float())<fraction,0)
                h[0,0,:4]=torch.tensor([.5,.49609375,-.5,0.],device='cuda',dtype=torch.bfloat16)
                z[0,0,:4]=h[0,0,:4]
                for gh,gz,threshold in [(False,False,0.),(True,False,0.),(True,True,.5)]:
                    gated_h=torch.where(h>=threshold,h,0) if gh else h
                    gated_z=torch.where(z>=threshold,z,0) if gz else z
                    reference=(wh(gated_h)+wz(gated_z))+residual
                    previous=SimpleNamespace(w2=wh,wo=wz,gh=gh,gz=gz,th=threshold,tz=threshold,skip=True)
                    fn=candidate.Joint(previous);out=fn(h,z,residual).clone();fn.skip=False;unskipped=fn(h,z,residual).clone()
                    delta=out.float()-reference.float()
                    row={'zero_fraction_target':fraction,'gate_h':gh,'gate_z':gz,'threshold':threshold,
                        'bitwise_native':torch.equal(out,reference),'bitwise_skip_toggle':torch.equal(out,unskipped),
                        'different_elements':int((out!=reference).sum()),'max_abs':float(delta.abs().max()),
                        'relative_l2':float(delta.norm()/reference.float().norm())}
                    rows.append(row);write_json(dest/'components.json',rows);print(row,flush=True)
                    if not row['bitwise_skip_toggle']:raise ValueError('Skip toggle changes output')
                    torch.testing.assert_close(out,reference,atol=.25,rtol=.02)
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'native_bitwise_all':all(r['bitwise_native'] for r in rows),
            'skip_toggle_bitwise_all':all(r['bitwise_skip_toggle'] for r in rows),'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
