"""K045 synthetic arithmetic and independently computed hybrid work counts."""
import argparse
import shutil
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from common import RUN, module, runtime, record, write_json


def accounting(values, hybrid=True, skip=True):
    values=np.asarray(values).reshape(-1,values.shape[-1]);m,k=values.shape
    potential=(m//8)*(k//16)*16
    if not skip:return potential,0,0
    counts=np.count_nonzero(values,axis=1)
    safe=np.all((values==0)|((np.abs(values)>=2.**-50)&(np.abs(values)<=2.**50)),axis=1)
    simple=(counts<=2)&safe if hybrid else np.zeros(m,dtype=bool)
    remaining=values.copy();remaining[simple]=0
    active=np.any(remaining.reshape(m//8,8,k//16,16)!=0,axis=(1,3))
    issued=int(active.sum())*16
    return issued,potential-issued,int(counts[simple].sum())*128


def sum_work_counts(stats):
    if stats.ndim<2 or stats.shape[-1]!=6:raise ValueError('Six named fields on final counter axis required')
    return stats.sum(dim=tuple(range(stats.ndim-1))).cpu().tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch)
        paths=[RUN/'98_k045_joint_probe_v2.py',RUN/'common.py']+[f for f in (RUN/'candidates/k045').iterdir() if f.is_file()]
        for path in paths:
            target=dest/'source'/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(dest/'manifest.json',{'sources':[record(f) for f in paths],'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'seed':2801,'scope':'39 synthetic cases; native output and skip/count controls; independent MMA/SIMT accounting'})
        candidate=module('run028_k045_joint_probe',RUN/'candidates/k045/candidate.py')
        with torch.inference_mode():
            candidate.extension()
            wh=torch.nn.Linear(512,128,device='cuda',dtype=torch.bfloat16)
            wz=torch.nn.Linear(128,128,device='cuda',dtype=torch.bfloat16)
            patterns=['dense','random99','zero','one','two','three','mixed','h-simple','z-simple','same-lane-two','tiny','large','no-fast']
            for pattern in patterns:
                h=torch.randn((1,2048,512),device='cuda',dtype=torch.bfloat16)
                z=torch.randn((1,2048,128),device='cuda',dtype=torch.bfloat16)
                residual=torch.randn_like(z)
                for x in [h,z]:
                    if pattern=='random99':x.masked_fill_(torch.rand_like(x.float())<.99,0)
                    elif pattern not in {'dense','no-fast'}:
                        saved=x.clone();x.zero_()
                        for row in range(2048):
                            n={'zero':0,'one':1,'two':2,'three':3,'same-lane-two':2,'tiny':1,'large':1}.get(pattern,row%5)
                            if pattern=='h-simple':n=1 if x is h else 5
                            if pattern=='z-simple':n=1 if x is z else 5
                            for j in range(n):
                                col=(row*13+j*(32 if pattern=='same-lane-two' else 37))%x.shape[-1]
                                x[0,row,col]=saved[0,row,col]
                        if pattern=='tiny':x.mul_(2.**-60)
                        if pattern=='large':x.mul_(2.**60)
                for gh,gz,threshold in [(False,False,0.),(True,False,0.),(True,True,.5)]:
                    gated_h=torch.where(h>=threshold,h,0) if gh else h
                    gated_z=torch.where(z>=threshold,z,0) if gz else z
                    reference=(wh(gated_h)+wz(gated_z))+residual
                    previous=SimpleNamespace(w2=wh,wo=wz,gh=gh,gz=gz,th=threshold,tz=threshold,skip=True)
                    fn=candidate.Joint(previous)
                    if pattern=='no-fast':fn.fast_weights=False
                    out=fn(h,z,residual).clone();fn.count=True;counted=fn(h,z,residual).clone()
                    observed=sum_work_counts(fn.stats)
                    hc=accounting(gated_h.float().cpu().numpy(),fn.fast_weights)
                    zc=accounting(gated_z.float().cpu().numpy(),fn.fast_weights)
                    expected=[hc[0],hc[1],zc[0],zc[1],hc[2],zc[2]]
                    fn.skip=False;unskipped=fn(h,z,residual).clone()
                    dense_counts=sum_work_counts(fn.stats)
                    expected_dense=[2048//8*32*16,0,2048//8*8*16,0,0,0]
                    row={'pattern':pattern,'gate_h':gh,'gate_z':gz,'threshold':threshold,
                        'numerically_equal_native':torch.equal(out,reference),'numerically_equal_skip_toggle':torch.equal(out,unskipped),
                        'numerically_equal_count_toggle':torch.equal(out,counted),'max_abs':float((out.float()-reference.float()).abs().max()),
                        'observed_counts':observed,'expected_counts':expected,'dense_counts':dense_counts,'expected_dense_counts':expected_dense}
                    rows.append(row);write_json(dest/'components.json',rows);print(row,flush=True)
                    if not all(row[key] for key in ['numerically_equal_native','numerically_equal_skip_toggle','numerically_equal_count_toggle']):
                        raise ValueError('Hybrid arithmetic discrepancy')
                    if observed!=expected or dense_counts!=expected_dense:raise ValueError('Work counter discrepancy')
        if len(rows)!=39:raise ValueError('Incomplete coverage')
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
