"""GPU qualification of initial complete operator paths, including dynamic graphs."""
import argparse,time,traceback
from support import RUN,write,event,sha,source_hashes


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    import bootstrap
    bootstrap.setup();bootstrap.verify_baseline()
    import torch
    import primitives as ops
    from work_counters import Work
    started=time.monotonic();rows=[];extension=None
    with torch.inference_mode():
        for spec in ops.candidates(include_ablations=True):
            for k in (512,2048):
                item={'candidate':spec,'k':k,'cases':[],'pass':False}
                try:
                    layer=torch.nn.Linear(k,512,device='cuda',dtype=torch.bfloat16)
                    op=ops.Linear(layer,.1,spec,m=64)
                    if spec['family'] in ('e','f'):extension=op.ext
                    for case in ('zero','dense','sparse','equality','overflow','disjoint_support'):
                        x=torch.randn(64,k,device='cuda',dtype=torch.bfloat16)*.2
                        if case=='zero':x.zero_()
                        elif case=='sparse':x.masked_fill_(torch.rand_like(x.float())<.98,0)
                        elif case=='equality':
                            x.fill_(float(torch.tensor(.1,dtype=torch.bfloat16)));x[:,::3]=torch.nextafter(x[:,::3],torch.full_like(x[:,::3],-float('inf')))
                        elif case=='overflow':x.fill_(.5);x[:,1::4]=-.2
                        elif case=='disjoint_support':
                            x.zero_();x[torch.arange(64,device='cuda'),torch.arange(64,device='cuda')]=.5
                        gated=x.masked_fill(x<.1,0);expected=torch.nn.functional.linear(gated,layer.weight,layer.bias)
                        actual=op(x).clone();torch.cuda.synchronize()
                        err=(actual.float()-expected.float()).abs()
                        relative=float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(expected.float()).clamp_min(1e-20))
                        okay=bool(torch.isfinite(actual).all()) and not bool((err>.25+.02*expected.float().abs()).any()) and relative<=.02
                        work=Work();work.add('test',op,gated)
                        item['cases'].append({'case':case,'pass':okay,'max_abs':float(err.max()),'relative_l2':relative,'work':work.result()})
                    sample=x.clone();op(sample);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):output=op(sample)
                    for value in (0.,.5,0.):
                        sample.fill_(value);graph.replay();torch.cuda.synchronize()
                        expected=torch.nn.functional.linear(sample.masked_fill(sample<.1,0),layer.weight,layer.bias)
                        assert torch.allclose(output,expected,atol=.25,rtol=.02),'Graph reused stale metadata'
                    item['pass']=all(r['pass'] for r in item['cases']);item['changed_input_graph_pass']=True
                except Exception as exc:item.update(error=str(exc),traceback=traceback.format_exc())
                rows.append(item);write(dest/'operators.json',rows)
                event(dest,'operator',candidate=spec['id'],k=k,qualified=item['pass'],elapsed_seconds=time.monotonic()-started)
        if extension is not None:
            from compiler_evidence import retain
            retain(extension,dest/'compiler')
        eligible=[s['id'] for s in ops.candidates() if all(r['pass'] for r in rows if r['candidate']['id']==s['id'])]
        write(dest/'result.json',{'status':'complete','eligible':eligible,'results':rows,'elapsed_seconds':time.monotonic()-started,
                                 'source_hashes':source_hashes()})
        assert all(s['id'] in eligible for s in ops.candidates() if s['family']=='dense'),'Dense correctness control failed'
        assert any(s['id'] in eligible for s in ops.candidates() if s['family'] in ('e','f')),'No new candidate qualified; inspect retained failures before screening'


if __name__=='__main__':main()
