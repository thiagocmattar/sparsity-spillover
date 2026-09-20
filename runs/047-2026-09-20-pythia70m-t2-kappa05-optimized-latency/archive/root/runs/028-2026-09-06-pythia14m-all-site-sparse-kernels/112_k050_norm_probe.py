"""Paired LayerNorm/Gate arithmetic and graph-refresh qualification."""
import argparse
import shutil
import time
import traceback
import torch
from common import RUN,module,runtime,record,write_json
from sparsity_research.sites import FixedOneSidedThreshold


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);args=p.parse_args()
    if not args.attempt.replace('-','').isalnum():p.error('Simple attempt identity required')
    dest=RUN/'artifacts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[]
    try:
        runtime(torch);torch.manual_seed(2850)
        paths=[RUN/'112_k050_norm_probe.py',RUN/'common.py']+[f for f in (RUN/'candidates/k050').iterdir() if f.is_file()]
        for path in paths:
            target=dest/'source'/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(dest/'manifest.json',{'sources':[record(f) for f in paths],'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'seed':2850,'scope':'28 paired LayerNorm/gate patterns plus graph input refresh'})
        candidate=module('run028_k050_norm_probe',RUN/'candidates/k050/candidate.py')
        with torch.inference_mode():
            candidate.extension()
            for pattern in ['random','zero','constant','small','large','boundary','tail']:
                size=7 if pattern=='tail' else 2048
                x=torch.randn((1,size,128),device='cuda',dtype=torch.bfloat16)
                if pattern in {'zero','boundary'}:x.zero_()
                if pattern=='constant':x.fill_(3.)
                if pattern=='small':x.mul_(2.**-20)
                if pattern=='large':x.mul_(2.**20)
                first=torch.nn.LayerNorm(128,device='cuda',dtype=torch.bfloat16)
                second=torch.nn.LayerNorm(128,device='cuda',dtype=torch.bfloat16)
                for norm in [first,second]:
                    norm.weight.copy_(torch.randn_like(norm.weight));norm.bias.copy_(torch.randn_like(norm.bias))
                for ga,gm,threshold in [(False,False,0.),(True,False,0.),(False,True,.01),(True,True,.5)]:
                    if pattern=='boundary':
                        values=torch.tensor([threshold-.0078125,threshold,threshold+.0078125,-0.],device='cuda',dtype=torch.bfloat16).repeat(32)
                        first.bias.copy_(values);second.bias.copy_(values.flip(0))
                    gate_a=FixedOneSidedThreshold(threshold) if ga else None
                    gate_m=FixedOneSidedThreshold(threshold) if gm else None
                    reference_a=first(x);reference_m=second(x)
                    if ga:reference_a=gate_a(reference_a)
                    if gm:reference_m=gate_m(reference_m)
                    pair=candidate.NormPair(first,second,gate_a,gate_m)
                    actual_a=pair.start(x).clone();actual_m=pair.finish(x).clone()
                    row={'pattern':pattern,'rows':size,'gate_a':ga,'gate_m':gm,'threshold':threshold,
                        'numerically_equal_a':torch.equal(actual_a,reference_a),'numerically_equal_m':torch.equal(actual_m,reference_m),
                        'zero_mask_equal_a':torch.equal(actual_a==0,reference_a==0),'zero_mask_equal_m':torch.equal(actual_m==0,reference_m==0),
                        'max_abs_a':float((actual_a.float()-reference_a.float()).abs().max()),
                        'max_abs_m':float((actual_m.float()-reference_m.float()).abs().max())}
                    rows.append(row);write_json(dest/'components.json',rows);print(row,flush=True)
                    if not all(row[k] for k in ['numerically_equal_a','numerically_equal_m','zero_mask_equal_a','zero_mask_equal_m']):
                        raise ValueError('Paired LayerNorm or gate discrepancy')
            # Replaying the same captured addresses must recompute from their new contents.
            pair=candidate.NormPair(first,second);static=x.clone()
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):pair.start(static);pair.finish(static)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):a=pair.start(static);m=pair.finish(static)
            for _ in range(3):
                static.copy_(torch.randn_like(static));graph.replay()
                if not torch.equal(a,first(static)) or not torch.equal(m,second(static)):
                    raise ValueError('Graph replay retained stale normalization values')
            if len(rows)!=28:raise ValueError('Incomplete primitive coverage')
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'graph_refresh_cases':3,'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','cases':len(rows),'error':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started});raise


if __name__=='__main__':main()
