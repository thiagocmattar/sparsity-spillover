"""Same-process generic/specialized full-model pairs and eager qualification."""
import argparse
import platform
import os
import time
import traceback
from io_utils import RUN, read, write, record, verify, module


def diagnostic_modules():
    # Frozen helpers prepend their own directories to sys.path. Resolve these
    # run-owned diagnostics by path, as learned from Run061's retained retry.
    return (module('run063_diagnostics',RUN/'diagnostics.py'),
            module('run063_inspection',RUN/'inspection.py'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair',choices=['0','0p1'],required=True)
    parser.add_argument('--replicate',type=int,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    cfg=read(RUN/'config.json')
    if not args.attempt.replace('-','').isalnum() or not 1<=args.replicate<=cfg['process_replicates']:
        parser.error('Unique simple attempt and a declared replicate required')
    dest=RUN/'artifacts/attempts'/args.attempt
    dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'status':'running','arguments':vars(args),'config':record(RUN/'config.json'),
            'scientific_config':cfg,'run_id':cfg['run_id'],
            'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}

    def emit(stage,**fields):
        row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,
             'elapsed_seconds':time.monotonic()-started,**fields}
        write(dest/'status.json',row)
        with (dest/'events.jsonl').open('a',encoding='utf-8',newline='\n') as handle:
            import json
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
        print(__import__('json').dumps(row),flush=True)

    try:
        emit('imports')
        import replay
        import numpy as np
        import torch
        import transformers
        import topology
        topology.register()
        from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
        for name,actual in [('python','.'.join(platform.python_version().split('.')[:2])),
                ('torch',torch.__version__.split('+')[0]),('transformers',transformers.__version__),
                ('numpy',np.__version__),('cuda',torch.version.cuda)]:
            if actual!=cfg['runtime'][name]:
                raise RuntimeError(f'Pinned runtime mismatch: {name}={actual}')
        if torch.cuda.get_device_name()!=cfg['gpu']:
            raise RuntimeError('Exact RTX5090 hardware required for this cohort')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
            raise RuntimeError('Default cuBLAS workspace required')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        manifest=read(RUN/'provenance/inputs.json')
        result['source_checkout']=read(RUN/'provenance/deployment.json')['source_checkout']
        checkpoints={key:next(c for c in manifest['checkpoints'] if c['id']==identifier)
                     for key,identifier in zip(['tile'],cfg['pair_conditions'][args.pair])}
        for checkpoint in checkpoints.values():
            for row in checkpoint['files']+checkpoint['provenance']:
                verify(row)
        validation=np.memmap(verify(manifest['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444):
            raise ValueError('Full validation coverage mismatch')
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False)
        if args.smoke:
            indices=indices[:4]
        inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        models={};groups={};timed={};implementations={}
        with torch.inference_mode():
            for family,checkpoint in checkpoints.items():
                group={}
                for mode in cfg['modes']:
                    emit('loading',family=family,mode=mode)
                    model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                    model.set_attn_implementation('sdpa');model.config.use_cache=False
                    if mode!='native':
                        implementations[family+'_'+mode]=replay.install(model,mode)
                    models[family+'_'+mode]=model
                    if mode=='native':
                        runner=replay.dense.DenseRunner(lambda ids,m=model:m(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                        runner.prepare();group['native']=runner
                    emit('capture',family=family,mode=mode)
                    runner=replay.dense.DenseRunner(lambda ids,m=model:replay.scaffold.forward(m,ids),inputs[0].clone(),'graph')
                    runner.prepare();group[mode+'_graph']=runner
                    timed[family+'_'+mode]=runner
                groups[family]=group
            result.update(checkpoints=checkpoints,implementation_coverage=implementations,
                runtime={'gpu':torch.cuda.get_device_name(),'gpu_properties':str(torch.cuda.get_device_properties(0)),
                'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
                'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,
                'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                'topologies':{key:topology_metadata(value) for key,value in models.items()},
                'setup_seconds':{f:{k:r.setup_seconds for k,r in g.items()} for f,g in groups.items()},
                'timer':'synchronized host full-logit graph replay; equal input staging/setup excluded; recurring gates/support/packing included',
                'source_files':[record(RUN/n) for n in ['kernel.py','joint.cu','generic.cu','inspection.cu','inspection.py','diagnostics.py','hz_adapter.py','02_benchmark.py','replay.py','tile_gate.py','topology.py']],
                'eager_anchor':'each checkpoint has its own native uncached forward'})
            write(dest/'manifest.json',result)
            for group in groups.values():
                for runner in group.values():
                    for ids in inputs:
                        runner.stage(ids);runner()
            torch.cuda.synchronize()
            emit('timing',inputs=len(inputs),passes=2 if args.smoke else cfg['timing_passes'])
            with (dest/'samples.jsonl').open('x',encoding='utf-8',newline='\n') as sink:
                def save(row):
                    sink.write(__import__('json').dumps(row,allow_nan=False)+'\n');sink.flush()
                samples=replay.dense.paired_probe(timed,inputs,passes=2 if args.smoke else cfg['timing_passes'],
                    seed=cfg['timing_seed']+args.replicate-1,sink=save)
            timing=replay.dense.timing_summary(samples,reference='tile_generic_sparse')
            write(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary':timing})
            blocks=8 if args.smoke else cfg['validation_blocks'];quality={}
            for family,group in groups.items():
                emit('validation',family=family,target_blocks=blocks)
                validation_started=time.monotonic()
                def validation_progress(f=family,**fields):
                    elapsed=time.monotonic()-validation_started
                    done=fields['blocks']
                    emit('validation',family=f,**fields,input_tokens_per_second=done*2048/elapsed,
                         remaining_seconds=(blocks-done)*elapsed/done)
                stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
                quality[family]=replay.dense.compare_inputs(group,stream,cfg['numerical_bounds'],
                    progress=validation_progress)
                quality[family].update(blocks=blocks,documents=500 if blocks==338 else None,
                    input_tokens=blocks*2048,excluded_tail_tokens=1444 if blocks==338 else None)
                for mode,checks in quality[family]['gates'].items():
                    bad=next((row for row in checks if not row['pass']),None)
                    if bad:
                        write(dest/f'failure-{family}-{mode}.json',bad)
            write(dest/'quality.json',quality)
            result.update(quality=quality,qualified=all(all(q['pass'].values()) for q in quality.values()),
                          loss={f:q['loss'] for f,q in quality.items()},timing=timing)
            if not result['qualified']:
                raise ValueError('Full numerical qualification failed; stop before diagnostics')
            if args.replicate==1 and not args.smoke:
                diagnostics,inspection=diagnostic_modules()
                checkpoint=checkpoints['tile']
                for mode in ['generic_sparse','tile_sparse']:
                    emit('diagnostics',mode=mode)
                    diagnostics.collect(models['tile_'+mode],models['tile_native'],validation,
                        dest/f'diagnostics-{mode}.json',lambda stage,**fields:emit(stage,mode=mode,**fields),
                        checkpoint['architecture'],blocks=blocks)
                inspection.collect(models['tile_tile_sparse'],inputs[0],dest/'inspection.json')
            result.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                          evidence_label='valid' if result['qualified'] else 'invalid')
            write(dest/'manifest.json',result)
            write(dest/'result.json',result)
            emit('complete',qualified=result['qualified'],loss=result['loss'],timing=timing)
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        emit('failed',error=str(exc))
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        write(dest/'manifest.json',result)
        write(dest/'result.json',result)
    return 0 if result['status']=='complete' and result['qualified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
