"""Development-only CUDA qualification on synthetic and real attention operands."""
import argparse
import json
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN, ROOT, R27, manifest, module, runtime, write_json, read_json, record, verify_record
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    p.add_argument('--conditions',nargs='+',default=['c25','c30'])
    p.add_argument('--inputs',type=int,default=2)
    p.add_argument('--synthetic-only',action='store_true')
    p.add_argument('--candidate',default='k020',choices=['k020','k022'])
    a=p.parse_args()
    if not a.attempt.replace('-','').isalnum() or not 1<=a.inputs<=64: p.error('Invalid identity/count')
    dest=RUN/'artifacts'/a.attempt
    dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    rows=[]; timings=[]
    def emit(stage,**fields):
        r={'stage':stage,'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',r)
        with (dest/'events.jsonl').open('a') as f: f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    try:
        runtime(torch)
        candidate=module('run028_attention_probe',RUN/f'candidates/{a.candidate}/candidate.py')
        functions={f'prefix{int(prefix)}-round{int(rnd)}':candidate.Attention(prefix,rnd)
                   for prefix in [False,True] for rnd in [False,True]}
        source_paths=[RUN/name for name in ['01_probe.py','common.py','config.json','candidates/k020/candidate.py','candidates/k020/kernel.cu']]
        if a.candidate!='k020':source_paths+=list((RUN/f'candidates/{a.candidate}').glob('*.py'))+list((RUN/f'candidates/{a.candidate}').glob('*.cu'))
        write_json(dest/'manifest.json',{'arguments':vars(a),'sources':[record(path) for path in source_paths],
            'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'measurement':'development component only, not qualified full-model speedup'})
        emit('compile')
        candidate.extension()
        def case(identity,q,k,v):
            scale=32**-.5
            reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
            qzeros=int((q==0).all(-1).sum());qrows=q.numel()//32
            row={'identity':identity,'shape':list(q.shape),'zero_query_rows':qzeros,'query_rows':qrows,
                 'q_zero_elements':int((q==0).sum()),'k_zero_elements':int((k==0).sum()),'v_zero_elements':int((v==0).sum()),'elements':q.numel(),'variants':{}}
            for label,fn in functions.items():
                actual=fn(q,k,v,scale,count=True).clone()
                diff=(actual.float()-reference.float())
                row['variants'][label]={'max_abs':float(diff.abs().max()),
                    'relative_l2':float(diff.norm()/reference.float().norm().clamp_min(1e-30)),
                    'bitwise_equal':torch.equal(actual,reference),'finite':bool(torch.isfinite(actual).all()),
                    'qk_fma_terms':int(fn.stats[...,0].sum()),'pv_fma_terms':int(fn.stats[...,1].sum()),
                    'prefix_query_rows':int(fn.stats[...,2].sum())}
            rows.append(row)
            write_json(dest/'components.json',rows)
            if q.shape[2]==2048:
                calls={'native':lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)}
                calls.update({label:(lambda op=fn:op(q,k,v,scale)) for label,fn in functions.items()})
                for fn in calls.values():
                    for _ in range(3):fn()
                order_rng=np.random.default_rng(2802)
                for rep in range(7):
                    for label in order_rng.permutation(list(calls)):
                        torch.cuda.synchronize();before=time.perf_counter()
                        calls[label]()
                        torch.cuda.synchronize()
                        timings.append({'identity':identity,'repeat':rep,'mode':str(label),'host_ms':1000*(time.perf_counter()-before)})
                write_json(dest/'timings.json',timings)
            emit('component',identity=identity,zero_query_rows=qzeros,query_rows=qrows,variants=row['variants'])
        with torch.inference_mode():
            for length in [17,129,2048]:
                q,k,v=[torch.randn((1,4,length,32),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                case(f'synthetic-dense-{length}',q,k,v)
                q.zero_();v[:,:,::3]=0
                case(f'synthetic-zero-q-{length}',q,k,v)
                q[:,:,::17,0]=.5;k[:,:,:,1:]=0
                case(f'synthetic-mixed-{length}',q,k,v)
                v.zero_()
                case(f'synthetic-zero-v-{length}',q,k,v)
            if not a.synthetic_only:
                source=manifest()
                cache=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
                original=module('run028_previous_adapter',R27/'adapter.py')
                for condition in a.conditions:
                    checkpoint=next(r for r in source['checkpoints'] if r['id']==condition)
                    for r in checkpoint['files']+checkpoint['provenance']:verify_record(r)
                    write_json(dest/f'checkpoint-{condition}.json',checkpoint)
                    model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                    model.set_attn_implementation('sdpa');model.config.use_cache=False
                    original.install(model,'fusion_dense')
                    captured={};handles=[]
                    for layer_id,layer in enumerate(model.gpt_neox.layers):
                        for site in ['q_post','k_post','v']:
                            def hook(mod,args,out,index=layer_id,key=site):captured[index,key]=out.detach().clone()
                            handles.append(getattr(layer.attention,site+'_site').register_forward_hook(hook))
                    for idx in range(a.inputs):
                        ids=torch.tensor(cache[idx].copy(),device='cuda',dtype=torch.long)[None]
                        logits=model(input_ids=ids,use_cache=False).logits
                        loss=float(torch.nn.functional.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1)))
                        emit('captured',condition=condition,input=idx,loss=loss)
                        del logits
                        for layer_id in range(6):
                            case(f'{condition}-input{idx}-layer{layer_id}',*[captured[layer_id,s] for s in ['q_post','k_post','v']])
                    for h in handles:h.remove()
                    del model,captured
                    torch.cuda.empty_cache()
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'timing_samples':len(timings),'elapsed_seconds':time.monotonic()-started})
        emit('complete',cases=len(rows),timing_samples=len(timings))
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc(),'elapsed_seconds':time.monotonic()-started})
        emit('failed',error=str(exc));raise


if __name__=='__main__':main()
