"""Direct same-process Ph/P2 latency pairs with independent eager anchors."""
import argparse
import json
import os
import platform
import time
import traceback
from io_utils import RUN, read, write, record, verify, module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',choices=['0','0p1'],required=True)
    parser.add_argument('--replicate',type=int,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.attempt.replace('-','').isalnum():parser.error('Simple unique attempt name required')
    cfg=read(RUN/'config.json')
    if not 1<=args.replicate<=cfg['process_replicates']:parser.error('Replicate outside contract')
    dest=RUN/'artifacts/attempts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'status':'running','arguments':vars(args),'config':record(RUN/'config.json')}
    def emit(stage,**fields):
        row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,
             'elapsed_seconds':time.monotonic()-started,**fields}
        write(dest/'status.json',row)
        with (dest/'events.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
        print(json.dumps(row),flush=True)
    try:
        emit('imports')
        import replay
        import numpy as np
        import torch
        import transformers
        from sparsity_research.sites import TOPOLOGIES,Topology
        TOPOLOGIES['HZ']=Topology('HZ',('h','z'))
        from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
        for name,actual in [('python','.'.join(platform.python_version().split('.')[:2])),
                ('torch',torch.__version__.split('+')[0]),('transformers',transformers.__version__),
                ('numpy',np.__version__),('cuda',torch.version.cuda)]:
            if actual!=cfg['runtime'][name]:raise RuntimeError(f'Pinned runtime mismatch: {name}={actual}')
        if torch.cuda.get_device_name()!=cfg['gpu']:raise RuntimeError('Exact RTX5090 hardware required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise RuntimeError('Use default cuBLAS workspace')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        manifest=read(RUN/'provenance/inputs.json')
        checkpoints={key:next(row for row in manifest['checkpoints'] if row['id']==identifier)
                     for key,identifier in zip(['ph','p2'],cfg['pair_conditions'][args.pair])}
        for checkpoint in checkpoints.values():
            for row in checkpoint['files']+checkpoint['provenance']:verify(row)
        validation=np.memmap(verify(manifest['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444):raise ValueError('Full validation coverage mismatch')
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False)
        if args.smoke:indices=indices[:4]
        inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        candidate=replay.final_row('k050',read(RUN/'provenance/candidates.json')['configurations'])
        if candidate['status']!='eligible':raise ValueError('Frozen K050 is not eligible')
        result.update(checkpoints=checkpoints,configuration=candidate)
        models={};groups={};timed={};implementation={}
        with torch.inference_mode():
            for pressure,checkpoint in checkpoints.items():
                group={}
                for mode in ['native','candidate']:
                    emit('loading',pressure=pressure,mode=mode)
                    model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                    model.set_attn_implementation('sdpa');model.config.use_cache=False
                    if mode=='candidate':implementation[pressure]=replay.install(model,candidate)
                    models[pressure+'_'+mode]=model
                    if mode=='native':
                        runner=replay.dense.DenseRunner(lambda ids,m=model:m(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                        runner.prepare();group['native']=runner
                    emit('capture',pressure=pressure,mode=mode)
                    runner=replay.dense.DenseRunner(lambda ids,m=model:replay.scaffold.forward(m,ids),inputs[0].clone(),'graph')
                    runner.prepare();group[mode+'_graph']=runner
                    timed[pressure+'_'+mode]=runner
                groups[pressure]=group
            result['runtime']={'gpu':torch.cuda.get_device_name(),'gpu_properties':str(torch.cuda.get_device_properties(0)),
                'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
                'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,
                'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                'topologies':{key:topology_metadata(value) for key,value in models.items()},
                'setup_seconds':{pressure:{key:value.setup_seconds for key,value in group.items()} for pressure,group in groups.items()},
                'timer':'randomized same-input/pass paired synchronized host full-logit graph replay; equal staging excluded',
                'eager_anchor':'each checkpoint has its own unmodified native forward'}
            result['implementation_coverage']=implementation
            write(dest/'manifest.json',result)
            for group in groups.values():
                for runner in group.values():
                    for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize()
            emit('timing',inputs=len(inputs),passes=2 if args.smoke else cfg['timing_passes'])
            samples=replay.dense.paired_probe(timed,inputs,passes=2 if args.smoke else cfg['timing_passes'],seed=cfg['timing_seed']+args.replicate-1)
            timing=replay.dense.timing_summary(samples,reference='ph_candidate')
            write(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary':timing})
            result['timing']=timing
            blocks=8 if args.smoke else cfg['validation_blocks'];quality={}
            for pressure,group in groups.items():
                emit('validation',pressure=pressure,target_blocks=blocks)
                stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
                quality[pressure]=replay.dense.compare_inputs(group,stream,cfg['numerical_bounds'],progress=lambda p=pressure,**fields:emit('validation',pressure=p,**fields))
                quality[pressure].update(blocks=blocks,documents=500 if blocks==338 else None,
                    input_tokens=blocks*2048,excluded_tail_tokens=1444 if blocks==338 else None)
                bad=next((row for row in quality[pressure]['gates']['candidate_graph'] if not row['pass']),None)
                if bad:
                    i=bad['input_index'];ids=torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                    for runner in group.values():runner.stage(ids)
                    ref=group['native']().clone();actual=group['candidate_graph']()
                    errors=(actual.float()-ref.float()).abs().flatten();top=torch.topk(errors,16)
                    write(dest/f'failure-{pressure}.json',{'block':i,'input_ids':ids.cpu().tolist(),
                        'flat_logit_indices':top.indices.cpu().tolist(),'absolute_errors':top.values.cpu().tolist(),
                        'native':ref.flatten()[top.indices].float().cpu().tolist(),
                        'candidate':actual.flatten()[top.indices].float().cpu().tolist(),'gate':bad})
                    del ref,actual,errors
            write(dest/'quality.json',quality)
            result.update(quality=quality,qualified=all(all(row['pass'].values()) for row in quality.values()),
                loss={pressure:row['loss'] for pressure,row in quality.items()})
            if args.replicate==1 and not args.smoke:
                diagnostic=module('run059_frozen_diagnostics',replay.R28/'115_hybrid_diagnostics.py')
                for pressure,checkpoint in checkpoints.items():
                    emit('diagnostics',pressure=pressure)
                    diagnostic.collect(models[pressure+'_candidate'],models[pressure+'_native'],validation,
                        dest/f'diagnostics-{pressure}.json',lambda stage,p=pressure,**fields:emit(stage,pressure=p,**fields),
                        checkpoint['canonical_logical_products']['architecture_maximum'],blocks=blocks)
            result.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete',qualified=result['qualified'],loss=result['loss'],timing=timing)
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        emit('failed',error=str(exc))
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write(dest/'result.json',result)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
