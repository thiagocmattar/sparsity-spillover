"""Dense attention schedules: causal, zero, sparse and nonzero synthetic checks."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import traceback
import torch
import replay
from io_utils import RUN, module, read, record, verify, write


def main(identifier):
    folder=RUN/'candidates'/identifier
    manifest=read(folder/'manifest.json')
    for row in manifest['files']: verify(row)
    implementation=module('operator_'+identifier,folder/'attention/candidate.py')
    result={'status':'running','candidate':manifest,'checker':record(Path(__file__)),'checks':[]}
    torch.manual_seed(2801)
    try:
        with torch.inference_mode():
            op=implementation.Attention(skip=False,shortcut=False)
            for name in ('zero','zero_q','dense_small','dense','sparse','gated'):
                q,k,v=[torch.randn(1,8,2048,64,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                if name=='zero': q.zero_();k.zero_();v.zero_()
                if name=='zero_q': q.zero_()
                if name=='dense_small': q.mul_(.1);k.mul_(.1);v.mul_(.1)
                if name=='sparse': q[:,:,:1024]=0;k[:,:,:512]=0;v[:,:,:1024]=0
                if name=='gated':
                    for x in (q,k,v): x.masked_fill_(x.abs()<.5,0)
                actual=op(q,k,v,.125).clone()
                expected=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=.125)
                error=(actual.float()-expected.float()).abs()
                relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20))
                passed=bool(torch.isfinite(actual).all()) and bool((error<=.015625+.02*expected.float().abs()).all()) and relative<=.02
                row={'case':name,'maximum_absolute_error':float(error.max()),'relative_l2':relative,'pass':passed}
                result['checks'].append(row);print(row,flush=True)
                assert passed,row
                # Future tokens must not influence the already computed first half.
                k[:,:,1024:].add_(2);v[:,:,1024:].sub_(3)
                changed=op(q,k,v,.125).clone()
                assert torch.equal(actual[:,:,:1024],changed[:,:,:1024]),'Causal prefix changed'
            result['status']='passed'
    except Exception:
        result['status']='failed';result['traceback']=traceback.format_exc();raise
    finally:
        write(RUN/'artifacts/development'/f'operator-{identifier}.json',result)

if __name__=='__main__': main(sys.argv[1])
