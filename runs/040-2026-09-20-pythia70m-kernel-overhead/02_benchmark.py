"""One fresh-process matched graph evaluation with full eager-anchored validation."""
import argparse
import json
import os
import platform
import time
import traceback
from io_utils import RUN, read, write, record, verify, module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',required=True)
    p.add_argument('--condition',required=True)
    p.add_argument('--replicate',required=True,type=int)
    p.add_argument('--attempt',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--final',action='store_true')
    p.add_argument('--development',action='store_true')
    a=p.parse_args()
    if a.development and (a.final or a.smoke): p.error('Development, smoke and final have distinct coverage')
    from controls import MODES
    if a.candidate not in MODES and not (a.candidate.startswith('opt') and len(a.candidate)==6 and a.candidate[3:].isdigit()):
        p.error('Unknown mode or candidate identifier')
    if not a.attempt.replace('-','').isalnum(): p.error('Simple unique attempt name required')
    cfg=read(RUN/'config.json')
    if not 1<=a.replicate<=cfg['process_replicates']:p.error('Replicate outside contract')
    manifest=read(RUN/'provenance/inputs.json')
    checkpoint=next(r for r in manifest['checkpoints'] if r['id']==a.condition)
    dest=RUN/'artifacts/attempts'/a.attempt
    dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'status':'running','arguments':vars(a),'condition':a.condition,
            'candidate':a.candidate,'checkpoint':checkpoint,'config':record(RUN/'config.json'),
            'run_sources': [record(p) for p in sorted(RUN.glob('*.py'))],
            'archive':record(RUN/'provenance/archive.json'),'controls':record(RUN/'controls.py'),
            'port_sources': {p.relative_to(RUN).as_posix(): record(p) for p in sorted((RUN/'kernel').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}}
    def emit(stage,**fields):
        row={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,
             'elapsed_seconds':time.monotonic()-started,**fields}
        write(dest/'status.json',row)
        with (dest/'events.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    try:
        emit('imports')
        import replay
        import numpy as np
        import torch
        import transformers
        from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata
        for name,actual in [('python','.'.join(platform.python_version().split('.')[:2])),
                            ('torch',torch.__version__.split('+')[0]),('transformers',transformers.__version__),
                            ('numpy',np.__version__),('cuda',torch.version.cuda)]:
            if actual!=cfg['runtime'][name]:raise RuntimeError(f'Pinned runtime mismatch: {name}={actual}')
        if torch.cuda.get_device_name()!=cfg['gpu']:raise RuntimeError('Exact RTX5090 hardware required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise RuntimeError('Use canonical default cuBLAS workspace')
        torch.manual_seed(cfg['runtime_seed'])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        for row in checkpoint['files']+checkpoint['provenance']:verify(row)
        data_record=manifest['development' if a.development else 'validation']
        result['evaluation_split']='training-development' if a.development else 'validation'
        result['data_identity']=data_record
        validation=np.memmap(verify(data_record),dtype=np.int32,mode='r')
        if not a.development and divmod(len(validation),2048)!=(338,1444):raise ValueError('Validation coverage mismatch')
        indices=(np.arange(cfg['development_blocks']) if a.development else
                 np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False))
        if len(validation)<(int(indices.max())+1)*2048:raise ValueError('Input selection exceeds retained cache')
        if a.smoke:indices=indices[:4]
        inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        if a.candidate not in MODES:
            candidate_manifest=read(RUN/'candidates'/a.candidate/'manifest.json')
            result['optimization_candidate']=candidate_manifest
            if not a.development:
                selection=read(RUN/'provenance/final-selection.json')
                if selection['candidate']!=a.candidate or selection['manifest']!=record(RUN/'candidates'/a.candidate/'manifest.json'):
                    raise ValueError('Only the development-selected frozen candidate may enter final validation')
        result['configuration']={'mode':a.candidate}
        models={};runners={}
        with torch.inference_mode():
            for mode in ['native','candidate'] + ([] if a.candidate=='full' else ['frozen']):
                emit('loading',mode=mode)
                model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,RUN/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa');model.config.use_cache=False
                if mode!='native':
                    metadata=replay.install(model,a.candidate if mode=='candidate' else 'full')
                    if mode=='candidate':result['implementation_coverage']=metadata
                models[mode]=model
                if mode=='native':
                    runner=replay.dense.DenseRunner(lambda ids,m=model:m(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                    runner.prepare();runners['native']=runner
                emit('capture',mode=mode)
                runner=replay.dense.DenseRunner(lambda ids,m=model:replay.scaffold.forward(m,ids),inputs[0].clone(),'graph')
                runner.prepare();runners[mode+'_graph']=runner
            runtime={'gpu':torch.cuda.get_device_name(),'gpu_properties':str(torch.cuda.get_device_properties(0)),
                     'python':platform.python_version(),'torch':torch.__version__,'torch_git':torch.version.git_version,
                     'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                     'device_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
                     'cpu_threads':torch.get_num_threads(),'topology':topology_metadata(models['native']),
                     'setup_seconds':{k:v.setup_seconds for k,v in runners.items()},
                     'timer':'paired synchronized host full-logit graph replay; equal input staging excluded',
                     'eager_anchor':'unmodified native forward; not the speedup denominator'}
            write(dest/'manifest.json',{**result,'runtime':runtime})
            result['runtime']=runtime
            free_bytes,total_bytes=torch.cuda.mem_get_info()
            result['runtime']['free_gpu_bytes_after_capture']=free_bytes
            if free_bytes < 8*1024**3: raise RuntimeError('Less than 8GiB GPU headroom after capture')
            # Refresh every timing input before measuring: no input-dependent result cache.
            for runner in runners.values():
                for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize()
            passes=2 if a.smoke else cfg['development_passes'] if a.development else cfg['timing_passes']
            emit('timing',inputs=len(inputs),passes=passes)
            samples=replay.dense.paired_probe({k:v for k,v in runners.items() if k.endswith('_graph')},inputs,
                    passes=passes,seed=cfg['timing_seed']+a.replicate-1)
            timing=replay.dense.timing_summary(samples,reference='native_graph')
            write(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary':timing})
            result['timing']=timing
            blocks=8 if a.smoke else cfg['development_blocks'] if a.development else cfg['validation_blocks']
            emit('validation',target_blocks=blocks)
            stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
            quality=replay.dense.compare_inputs(runners,stream,cfg['numerical_bounds'],progress=lambda **f:emit('validation',**f))
            quality.update(blocks=blocks,documents=500 if blocks==338 else None,input_tokens=blocks*2048,
                           excluded_tail_tokens=1444 if blocks==338 else None)
            write(dest/'quality.json',quality)
            result.update(qualified=all(quality['pass'].values()),qualification=quality['pass'],
                          loss=quality['loss'],loss_delta=quality['loss_delta'],validation_blocks=blocks)
            bad=next((r for r in quality['gates']['candidate_graph'] if not r['pass']),None)
            if bad:
                i=bad['input_index'];ids=torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None]
                for runner in runners.values():runner.stage(ids)
                ref=runners['native']().clone();actual=runners['candidate_graph']()
                errors=(actual.float()-ref.float()).abs().flatten();top=torch.topk(errors,16)
                write(dest/'failure-example.json',{'block':i,'input_ids':ids.cpu().tolist(),
                      'flat_logit_indices':top.indices.cpu().tolist(),'absolute_errors':top.values.cpu().tolist(),
                      'native':ref.flatten()[top.indices].float().cpu().tolist(),
                      'candidate':actual.flatten()[top.indices].float().cpu().tolist(),'gate':bad})
                del ref,actual,errors
            if a.final and a.replicate==1:
                diagnostic=module('run040_final_diagnostics',RUN/'diagnostics.py')
                emit('diagnostics',loss=quality['loss'])
                diagnostic.collect(models['candidate'],models['native'],validation,dest/'diagnostics.json',emit,
                    checkpoint['canonical_logical_products']['architecture_maximum'],blocks=blocks)
            if a.candidate=='full' and not a.smoke:
                from profiling import collect
                collect(models,runners,inputs[:4],dest)
            result.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete',qualified=result['qualified'],loss=result['loss'],timing=timing)
    except Exception as exc:
        result.update(status='unsupported' if str(exc).startswith('UNSUPPORTED:') else 'failed',
                      error=str(exc),traceback=traceback.format_exc())
        emit(result['status'],error=str(exc))
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        write(dest/'result.json',result)
    return 0 if result['status']=='complete' and result.get('qualified') else 1


if __name__=='__main__':
    raise SystemExit(main())
