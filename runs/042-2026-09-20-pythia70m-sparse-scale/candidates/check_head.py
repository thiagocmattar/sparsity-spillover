"""Full-vocabulary output and input-refresh checks for dense head schedules."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import traceback
import torch
import replay
from io_utils import RUN,module,read,write,record,verify

def main(identifier):
    folder=RUN/'candidates'/identifier;manifest=read(folder/'manifest.json')
    for row in manifest['files']:verify(row)
    implementation=module('operator_'+identifier,folder/'head.py')
    result={'status':'running','candidate':manifest,'checker':record(Path(__file__)),'checks':[]}
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        with torch.inference_mode():
            linear=torch.nn.Linear(512,50304,bias=False,device='cuda',dtype=torch.bfloat16)
            op=implementation.Head(linear,read(folder/'spec.json')['index'])
            for case in ('zero','dense','sparse','changed_input'):
                x=torch.randn(1,2048,512,device='cuda',dtype=torch.bfloat16)
                if case=='zero':x.zero_()
                if case=='sparse':x.masked_fill_(x.abs()<.5,0)
                if case=='changed_input':x.mul_(2)
                expected=linear(x);actual=op(x).clone()
                error=(actual.float()-expected.float()).abs()
                relative=float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20))
                passed=bool(torch.isfinite(actual).all()) and bool((error<=.03125+.02*expected.float().abs()).all()) and relative<=.02
                row={'case':case,'shape':list(actual.shape),'maximum_absolute_error':float(error.max()),'relative_l2':relative,'bitwise_equal':bool(torch.equal(actual,expected)),'pass':passed}
                result['checks'].append(row);print(row,flush=True);assert passed,row
            result['status']='passed'
    except Exception:
        result['status']='failed';result['traceback']=traceback.format_exc();raise
    finally:write(RUN/'artifacts/development'/f'operator-{identifier}.json',result)

if __name__=='__main__':main(sys.argv[1])
