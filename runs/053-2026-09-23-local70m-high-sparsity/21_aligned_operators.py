"""Numerical, independent work-count and changed-input graph checks."""
import argparse
import time
import traceback
from local_support import RUN, load, write, event, sha

def main():
    p=argparse.ArgumentParser(); p.add_argument('--attempt',required=True); args=p.parse_args()
    assert args.attempt.replace('-','').isalnum()
    dest=RUN/'artifacts/attempts'/args.attempt; dest.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); result={'status':'running','checks':[],'source_sha256':sha(RUN/'wide_sparse.py')}
    try:
        import torch
        from wide_sparse import Linear, expected_work
        cfg=load(RUN/'aligned-tile-config.json'); torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        assert torch.cuda.get_device_name()==load(RUN/'config.json')['gpu']
        with torch.inference_mode():
            for k in (512,2048):
                weight=torch.nn.Linear(k,512,device='cuda',dtype=torch.bfloat16)
                x=torch.zeros((64,k),device='cuda',dtype=torch.bfloat16)
                for spec in cfg['candidates']:
                    for no_skip in (False,True):
                        op=Linear(weight,.5,spec,m=64,no_skip=no_skip)
                        for case in ('zero','short','mixed','dense','boundary','empty_tiles'):
                            x.zero_()
                            if case=='short':x[:,::(k//8)]=.75
                            if case=='mixed':x[::3].normal_(); x[1::3,::(k//8)]=.75
                            if case=='dense':x.normal_()
                            if case=='boundary':x[:,:3]=torch.tensor([.498046875,.5,.50390625],device='cuda',dtype=torch.bfloat16)
                            if case=='empty_tiles':x[:,32:48]=.75
                            ref=weight(x.masked_fill(x<.5,0))
                            op.count=True; actual=op(x).clone()
                            assert torch.equal(op.work,expected_work(x,.5,spec,no_skip)),(k,spec,case)
                            delta=(actual.float()-ref.float()).abs()
                            rel=float(torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float()).clamp_min(1e-12))
                            valid=bool(torch.isfinite(actual).all() and (delta<=cfg['bounds']['atol']+cfg['bounds']['rtol']*ref.float().abs()).all()) and rel<=cfg['bounds']['relative_l2']
                            op.count=False
                            valid=valid and torch.equal(actual,op(x))
                            row={'k':k,'candidate':spec['id'],'no_skip':no_skip,'case':case,'pass':valid,'max_abs':float(delta.max()),'relative_l2':rel,'work':op.work.cpu().tolist()}
                            result['checks'].append(row)
                            assert valid,row
                        torch.cuda.synchronize()
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):output=op(x)
                        for dense in (False,True):
                            if dense:x.normal_()
                            else:x.zero_()
                            graph.replay()
                            torch.testing.assert_close(output,weight(x.masked_fill(x<.5,0)),atol=cfg['bounds']['atol'],rtol=cfg['bounds']['rtol'])
                        del graph
                        event(dest,'operators',k=k,candidate=spec['id'],no_skip=no_skip,elapsed_seconds=time.monotonic()-start)
            result.update(status='passed',changed_input_graph_checks=32)
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['elapsed_seconds']=time.monotonic()-start; write(dest/'result.json',result)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
