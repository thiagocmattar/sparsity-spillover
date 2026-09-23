"""Diagnose the retained qualification failure without changing its bounds."""
import argparse
import sys
import time
import traceback
from local_support import RUN, load, write, sha, event

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--attempt',required=True);a=p.parse_args()
    dest=RUN/'artifacts/attempts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    result={'status':'running','purpose':'Numerical diagnosis and missing full inventory for the unchanged failed candidate.'}
    started=time.monotonic()
    try:
        import numpy as np
        import torch
        import transformers
        torch.manual_seed(2801);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        sys.path.insert(0,str(RUN/'deps/run049'))
        import replay
        import install as base_install
        from sparsity_research.pythia import load_checkpoint_pythia
        import local_policy
        import full_diagnostics
        catalog=load(RUN/'provenance/inputs.json');row=next(r for r in catalog['checkpoints'] if r['id']==a.condition)
        for item in row['files']+row['provenance']+[catalog['validation']]:
            path=RUN/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        data=np.memmap(RUN/catalog['validation']['path'],dtype=np.int32,mode='r');assert divmod(len(data),2048)==(338,1444)
        result.update(checkpoint=row,data_identity=catalog['validation'],policy_sha256=sha(RUN/'provenance/local-policy.json'))
        records={};models={};logits={}
        class Observe:
            def __init__(self,core,key):self.core,self.key=core,key
            def __call__(self,h,z,residual):
                out=self.core(h,z,residual)
                records[self.key]={'h':h.clone(),'z':z.clone(),'output':out.clone()}
                return out
        with torch.inference_mode():
            ids=torch.tensor(data[78*2048:79*2048].copy(),device='cuda',dtype=torch.long)[None]
            for mode in ('reference','candidate'):
                model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/row['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa');model.config.use_cache=False
                if mode=='reference':base_install.install(model,'native_hz')
                else:local_policy.install(model,'candidate',base_install)
                for i,layer in enumerate(model.gpt_neox.layers):layer._run026_joint=Observe(layer._run026_joint,(mode,i))
                logits[mode]=replay.scaffold.forward(model,ids).clone();models[mode]=model
            def difference(x,y):
                d=(x.float()-y.float()).abs(); bad=x!=y
                indices=torch.nonzero(bad.reshape(-1),as_tuple=False).flatten()[:8]
                return {'different_elements':int(bad.sum()),'max_abs':float(d.max()),
                        'examples':[{'flat_index':int(i),'reference':float(x.reshape(-1)[i]),'candidate':float(y.reshape(-1)[i])} for i in indices]}
            layers=[]
            for i,layer in enumerate(models['candidate'].gpt_neox.layers):
                core=layer._run026_joint.core;ref=records['reference',i];cand=records['candidate',i]
                item={'layer':i,'joint_output':difference(ref['output'],cand['output']),'sites':{}}
                for site in ('h','z'):
                    raw=ref[site].reshape(2048,-1).contiguous();other=cand[site].reshape_as(raw)
                    threshold=float(torch.tensor(getattr(core,'t'+site),dtype=torch.bfloat16))
                    linear=getattr(core,'w2' if site=='h' else 'wo')
                    expected=torch.nn.functional.linear(raw.masked_fill(raw<threshold,0),linear.weight,linear.bias)
                    actual=getattr(core,site)(raw).clone()
                    item['sites'][site]={'input':difference(raw,other),
                        'gate_membership_changes':int(((raw>=threshold)!=(other>=threshold)).sum()),
                        'projection_on_identical_reference_input':difference(expected,actual)}
                layers.append(item)
            delta=(logits['reference'].float()-logits['candidate'].float()).abs()
            result.update(block=78,logits=difference(logits['reference'],logits['candidate']),
                          elementwise_bound_violations=int((delta>(.25+.02*logits['reference'].float().abs())).sum()),layers=layers)
            write(dest/'numerical-trace.json',{**result,'status':'trace_complete'})
            for model in models.values():
                for layer in model.gpt_neox.layers:layer._run026_joint=layer._run026_joint.core
            records.clear();logits.clear();del delta,ref,cand,raw,other,expected,actual
            quality=load(RUN/'artifacts/attempts/model-001-reference-c25/quality.json')
            full_diagnostics.collect(models['candidate'],data,replay,dest,
                lambda stage,**fields:event(dest,stage,condition=a.condition,**fields),quality['loss'])
            from io_utils import module
            profiler=module('run053_diagnosis_profile',RUN/'deps/run049/profiling.py')
            runner=replay.dense.DenseRunner(lambda x:replay.scaffold.forward(models['reference'],x),ids.clone(),'graph')
            runner.prepare()
            # Full candidate inventory temporarily wrapped it. Restore before profiling.
            for layer in models['candidate'].gpt_neox.layers:layer._run026_joint=layer._run026_joint.core
            cr=replay.dense.DenseRunner(lambda x:replay.scaffold.forward(models['candidate'],x),ids.clone(),'graph');cr.prepare()
            profiler.collect(models,{'reference_graph':runner,'candidate_graph':cr},[ids],dest)
            result['status']='complete'
    except Exception as exc:result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write(dest/'result.json',result)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
