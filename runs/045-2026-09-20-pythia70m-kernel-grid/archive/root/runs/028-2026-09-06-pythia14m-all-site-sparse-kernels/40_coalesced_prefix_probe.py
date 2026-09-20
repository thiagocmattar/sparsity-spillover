"""K031 primitive equivalence and issued/skipped MMA evidence, development only."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN,ROOT,R27,module,manifest,runtime,record,verify_record,write_json,read_json
from sparsity_research.pythia import load_checkpoint_pythia

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--synthetic-only',action='store_true');a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity required')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();rows=[];timings=[]
    def emit(stage,**fields):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        runtime(torch);candidate=module('run028_k031_probe',RUN/'candidates/k031/candidate.py')
        paths=[RUN/'40_coalesced_prefix_probe.py',RUN/'common.py']+[f for f in (RUN/'candidates/k031').iterdir() if f.is_file()]
        for path in paths:
            target=dest/'source'/path.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        write_json(dest/'manifest.json',{'arguments':vars(a),'sources':[record(f) for f in paths],
            'upstream':read_json(RUN/'runtime/vendor/inventory.json'),'torch':torch.__version__,'torch_git':torch.version.git_version,
            'gpu':torch.cuda.get_device_name(),'counter_unit':'16x8x16 MMA atoms including padding, 2048 FMA-equivalent terms per atom',
            'scope':'synthetic and two training-block captured operands; not full-model qualification'})
        emit('compile');candidate.extension()
        functions={'same_kernel_dense':candidate.Attention(False),'sparse':candidate.Attention(True),'sparse_no_prefix':candidate.Attention(True,False)}
        def case(identity,q,k,v):
            reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=32**-.5)
            results={};outputs={}
            for label,fn in functions.items():
                out=fn(q,k,v,32**-.5,count=True).clone();outputs[label]=out
                delta=out.float()-reference.float();counts=fn.stats.sum((0,1,2,3)).cpu().tolist()
                results[label]={'max_abs':float(delta.abs().max()),'relative_l2':float(delta.norm()/reference.float().norm().clamp_min(1.e-30)),
                    'bitwise_equal_native':torch.equal(out,reference),'finite':bool(torch.isfinite(out).all()),
                    'qk_issued_mmas':counts[0],'qk_skipped_mmas':counts[1],'pv_issued_mmas':counts[2],'pv_skipped_mmas':counts[3],
                    'gate_mismatches_at_0p5':int(((out>=.5)!=(reference>=.5)).sum()),
                    'prefix_counters':dict(zip(['qk_bypassed_mmas','pv_reused_mmas','query_split_rows'],fn.prefix_stats.sum((0,1,2,3)).cpu().tolist()))}
            row={'identity':identity,'shape':list(q.shape),'variants':results,'skip_toggle_bitwise_equal':torch.equal(outputs['same_kernel_dense'],outputs['sparse']),
                'zeros':{key:int((value==0).sum()) for key,value in zip(['q','k','v'],[q,k,v])}}
            rows.append(row);write_json(dest/'components.json',rows)
            if q.shape[2]==2048:
                calls={'native':lambda:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=32**-.5)}
                calls.update({name:(lambda fn=fn:fn(q,k,v,32**-.5)) for name,fn in functions.items()})
                for fn in calls.values():
                    for _ in range(3):fn()
                rng=np.random.default_rng(2826)
                for repeat in range(7):
                    for name in rng.permutation(list(calls)):
                        torch.cuda.synchronize();before=time.perf_counter();calls[name]();torch.cuda.synchronize()
                        timings.append({'identity':identity,'repeat':repeat,'mode':str(name),'host_ms':1000*(time.perf_counter()-before)})
                write_json(dest/'timings.json',timings)
            emit('component',**row)
        with torch.inference_mode():
            for t in [2048]:
                q,k,v=[torch.randn((1,4,t,32),device='cuda',dtype=torch.bfloat16) for _ in range(3)]
                case(f'dense-{t}',q,k,v);q.zero_();case(f'zero-q-{t}',q,k,v)
                q[:,:,::17,0]=.5;k[:,:,:,1:]=0;v[:,:,::3]=0;case(f'mixed-{t}',q,k,v)
                v.zero_();case(f'zero-v-{t}',q,k,v)
                q.zero_();v.copy_(torch.randint(-64,65,v.shape,device='cuda').to(torch.bfloat16)*.5)
                v[0,0,0,0]=64.;case('safe-grid-zero-q',q,k,v)
                v[0,0,0,0]=.25;case('unsafe-grid-zero-q',q,k,v)
            if not a.synthetic_only:
                inputs=manifest();data=np.memmap(verify_record(inputs['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
                original=module('run028_native_probe_previous',R27/'adapter.py')
                for condition in ['c25','c30']:
                    checkpoint=next(r for r in inputs['checkpoints'] if r['id']==condition)
                    for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
                    write_json(dest/f'checkpoint-{condition}.json',checkpoint)
                    net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                    net.set_attn_implementation('sdpa');net.config.use_cache=False;original.install(net,'fusion_dense')
                    captured={};handles=[]
                    for index,layer in enumerate(net.gpt_neox.layers):
                        for site in ['q_post','k_post','v']:
                            def hook(mod,args,out,index=index,site=site):captured[index,site]=out.detach().clone()
                            handles.append(getattr(layer.attention,site+'_site').register_forward_hook(hook))
                    for i in range(2):
                        ids=torch.tensor(data[i].copy(),device='cuda',dtype=torch.long)[None];net(input_ids=ids,use_cache=False)
                        for layer in range(6):case(f'{condition}-input{i}-layer{layer}',*[captured[layer,s] for s in ['q_post','k_post','v']])
                    for handle in handles:handle.remove()
                    del net,captured;torch.cuda.empty_cache()
        write_json(dest/'result.json',{'status':'complete','cases':len(rows),'skip_toggle_bitwise_all':all(r['skip_toggle_bitwise_equal'] for r in rows),
            'native_bitwise_all':all(v['bitwise_equal_native'] for r in rows for v in r['variants'].values()),'elapsed_seconds':time.monotonic()-started})
        emit('complete',cases=len(rows))
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});emit('failed',error=str(exc));raise

if __name__=='__main__':main()
