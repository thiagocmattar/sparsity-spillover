"""Training-only component screen; every scan/gate/pack is inside its timer."""
import argparse
import gc
import random
import sys
import time
import traceback
from local_support import RUN, load, write, event, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--attempt',required=True);args=p.parse_args()
    cfg=load(RUN/'wider-output-config.json'); assert args.condition in cfg['conditions']
    assert args.attempt.replace('-','').isalnum()
    dest=RUN/'artifacts/attempts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();result={'status':'running','condition':args.condition,'config':cfg,'source_sha256':sha(RUN/'wide_sparse.py')}
    compiler={};work=[]; numerics=[]; completed=0; losses=[]; structure={}
    try:
        import numpy as np
        import torch
        import transformers
        from wide_sparse import Linear, expected_work
        assert torch.cuda.get_device_name()==load(RUN/'config.json')['gpu']
        torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        base=RUN/'deps/run049';sys.path.insert(0,str(base))
        import replay
        import install as base_install
        from io_utils import module
        from sparsity_research.pythia import load_checkpoint_pythia
        from sparsity_research.metrics import ActivationAccumulator,weight_statistics
        grid=RUN/'deps/run051'
        module('support',grid/'support.py')
        old=module('run053_prior_primitives',grid/'primitives.py')
        policy=load(grid/'selection-final.json')['policies']['sparse_c']
        old_specs={s['id']:s for s in old.candidates(include_ablations=True)}
        catalog=load(RUN/'provenance/inputs.json')
        row=next(r for r in catalog['checkpoints'] if r['id']==args.condition)
        for item in row['files']+row['provenance']+[catalog['development']]:
            path=RUN/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        result.update(checkpoint=row,data_identity=catalog['development'])
        data=np.memmap(RUN/catalog['development']['path'],mode='r',dtype=np.int32)
        captured={};params={}
        with torch.inference_mode():
            model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/row['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
            model.set_attn_implementation('sdpa');model.config.use_cache=False
            base_install.install(model,'native_hz')
            accumulator=ActivationAccumulator((0.,.001,.01))
            write(dest/'weights.json',weight_statistics(model))
            class Observe:
                def __init__(self,op,index):self.op,self.index=op,index
                def __call__(self,h,z,residual):
                    for site,value in [('h',h),('z',z)]:captured[f'{site}.{self.index}']=value.reshape(2048,-1).contiguous().clone()
                    return self.op(h,z,residual)
            for index,layer in enumerate(model.gpt_neox.layers):
                op=layer._run026_joint
                params[f'h.{index}']=(op.w2,op.th);params[f'z.{index}']=(op.wo,op.tz)
                layer._run026_joint=Observe(op,index)
            sample_path=dest/'samples.jsonl'
            with sample_path.open('w',encoding='utf-8') as samples:
                import json
                for block in range(cfg['blocks']):
                    ids=torch.tensor(data[block*2048:(block+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                    logits=replay.scaffold.forward(model,ids)
                    losses.append(float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,50304),ids[:,1:].reshape(-1))))
                    del logits
                    for site,(linear,threshold) in params.items():
                        x=captured[site];t=float(torch.tensor(threshold,dtype=torch.bfloat16))
                        gated=x.masked_fill(x<t,0);assert bool(torch.isfinite(gated).all())
                        accumulator.update({site.replace('.', '.layer_'):gated},torch=torch)
                        active=gated!=0;info=structure.setdefault(site,{})
                        for gm in (1,16,32,64):
                            histogram=torch.bincount(active.reshape(2048//gm,gm,-1).any(1).sum(-1),minlength=x.shape[1]+1)
                            key=f'union{gm}_nnz_hist'
                            if key not in info:info[key]=histogram
                            else:info[key]+=histogram
                        reference=torch.nn.functional.linear(gated,linear.weight,linear.bias)
                        modes={name:old.Linear(linear,threshold,old_specs[name]) for name in ('dense_native','dense_fused','dense_dot')}
                        modes['prior']=old.Linear(linear,threshold,old_specs[policy[site]])
                        for spec in cfg['candidates']:
                            for off in (False,True):
                                modes[spec['id']+('_no_skip' if off else '')]=Linear(linear,threshold,spec,no_skip=off)
                        graphs={}
                        for name,op in modes.items():
                            actual=op(x)
                            diff=actual.float()-reference.float()
                            rel=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
                            valid=bool(torch.isfinite(actual).all() and (diff.abs()<=cfg['bounds']['atol']+cfg['bounds']['rtol']*reference.float().abs()).all()) and rel<=cfg['bounds']['relative_l2']
                            numerics.append({'block':block,'site':site,'mode':name,'pass':valid,'max_abs':float(diff.abs().max()),'relative_l2':rel})
                            if isinstance(op,Linear):
                                op.count=True;counted=op(x).clone()
                                expected=expected_work(x,t,op.spec,op.no_skip)
                                assert torch.equal(op.work,expected),(block,site,name)
                                work.append({'block':block,'site':site,'mode':name,'executed_reduction_tiles':int(op.work.sum()),
                                             'potential_reduction_tiles':op.work.numel()*(op.k//op.spec['bk']),
                                             'nnz':int((x.masked_fill(x<t,0)!=0).sum()),'elements':x.numel()})
                                op.count=False;assert torch.equal(counted,op(x))
                            for _ in range(2):op(x)
                            torch.cuda.synchronize()
                            graph=torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                for _ in range(cfg['graph_repeats']):out=op(x)
                            graph.replay();graphs[name]=graph
                            if hasattr(op,'compiled'):
                                compiler[site+':'+name]=[{'registers':getattr(k,'n_regs',None),'spills':getattr(k,'n_spills',None),'metadata':str(getattr(k,'metadata',None))} for k in op.compiled]
                        rng=random.Random(cfg['seed']+block*100+list(params).index(site))
                        for repeat in range(cfg['passes']):
                            order=list(graphs);rng.shuffle(order)
                            for name in order:
                                torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                                started=time.perf_counter();a.record();graphs[name].replay();b.record();b.synchronize()
                                record={'block':block,'split':'development' if block<cfg['development_blocks'] else 'confirmation',
                                        'site':site,'mode':name,'repeat':repeat,
                                        'host_us':(time.perf_counter()-started)*1e6/cfg['graph_repeats'],
                                        'cuda_us':a.elapsed_time(b)*1000/cfg['graph_repeats']}
                                samples.write(json.dumps(record)+'\n')
                        del out,graph,graphs,modes,op,actual,reference,diff
                        gc.collect()
                    completed=block+1;samples.flush()
                    elapsed=time.monotonic()-start
                    event(dest,'screen',condition=args.condition,blocks=completed,target_blocks=cfg['blocks'],
                          loss=sum(losses)/len(losses),blocks_per_second=completed/elapsed,
                          remaining_seconds=(cfg['blocks']-completed)*elapsed/completed)
            result.update(status='complete',all_numerically_qualified=all(r['pass'] for r in numerics),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),completed_blocks=completed)
            write(dest/'activation-statistics.json',{'coverage':{'split':'training','blocks':completed,'input_tokens':completed*2048},
                  'per_site_layer':accumulator.rows(),'pooled_by_site':accumulator.pooled_by_site(),
                  'structure':{site:{key:value.cpu().tolist() for key,value in entries.items()} for site,entries in structure.items()}})
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc(),completed_blocks=completed)
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        write(dest/'numerics.json',numerics);write(dest/'work.json',work);write(dest/'compiler.json',compiler);write(dest/'result.json',result)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
