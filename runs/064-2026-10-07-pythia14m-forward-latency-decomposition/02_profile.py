"""Six fresh same-GPU processes, full qualification and complete forward map."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
BASE=Path(os.environ.get('RUN064_BASE',str(ROOT.parent/'063-2026-10-07-pythia14m-tile-specialized-kernel')))
sys.path.insert(0,str(BASE))
from io_utils import read,write,record,verify,module

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['0','0p1'],required=True)
    parser.add_argument('--replicate',type=int,required=True);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--tag',default='001');args=parser.parse_args()
    cfg=read(ROOT/'config.json'); old=read(BASE/'config.json')
    dest=ROOT/'artifacts/attempts'/f'{"smoke" if args.smoke else "full"}-k{args.pair}-r{args.replicate}-{args.tag}'
    dest.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    result={'status':'running','run_id':cfg['run_id'],'config':cfg,'arguments':vars(args),'source_config':record(ROOT/'config.json',ROOT)}
    def emit(stage,**fields):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-start,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields}
        write(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    try:
        emit('imports')
        import numpy as np
        import torch
        import transformers
        import replay
        import topology
        topology.register()
        stages=module('run064_stages',ROOT/'stages.py')
        profiling=module('run064_profiling',ROOT/'profiling.py')
        from sparsity_research.pythia import load_checkpoint_pythia
        for name,actual in [('python','.'.join(platform.python_version().split('.')[:2])),('torch',torch.__version__.split('+')[0]),('transformers',transformers.__version__),('numpy',np.__version__),('cuda',torch.version.cuda)]:
            if old['runtime'][name]!=actual:raise ValueError(f'Runtime mismatch {name}: {actual}')
        if torch.cuda.get_device_name()!=cfg['gpu']:raise ValueError('RTX5090 required')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise ValueError('Default workspace required')
        torch.manual_seed(cfg['runtime_seed']);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        manifest=read(BASE/'provenance/inputs.json')
        checkpoint=next(c for c in manifest['checkpoints'] if c['id']==old['pair_conditions'][args.pair][0])
        for row in checkpoint['files']+checkpoint['provenance']:verify(row)
        validation=np.memmap(verify(manifest['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444):raise ValueError('Coverage mismatch')
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False)
        if args.smoke:indices=indices[:4]
        inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        models={};baseline={};mapped={};tapes={};group={};metadata={}
        with torch.inference_mode():
            for mode in cfg['modes']:
                emit('load_capture',mode=mode)
                model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,BASE/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa');model.config.use_cache=False
                if mode!='native':metadata[mode]=replay.install(model,mode)
                models[mode]=model
                if mode=='native':
                    eager=replay.dense.DenseRunner(lambda ids,m=model:m(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native');eager.prepare();group['native']=eager
                runner=replay.dense.DenseRunner(lambda ids,m=model:replay.scaffold.forward(m,ids),inputs[0].clone(),'graph');runner.prepare()
                baseline[mode]=runner;group['baseline_'+mode]=runner
                tape=stages.Tape(events=True);tapes[mode]=tape
                runner=replay.dense.DenseRunner(lambda ids,m=model,t=tape,o=mode!='native':stages.forward(m,ids,t,o),inputs[0].clone(),'graph');runner.prepare()
                mapped[mode]=runner;group['mapped_'+mode]=runner
            result.update(checkpoint=checkpoint,validation=manifest['validation'],source_deployment=read(ROOT/'prelaunch/deployment.json'),
                implementation=metadata,runtime={'gpu':torch.cuda.get_device_name(),'gpu_uuid':str(torch.cuda.get_device_properties(0).uuid),
                'properties':str(torch.cuda.get_device_properties(0)),'python':platform.python_version(),'torch':torch.__version__,
                'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda},
                source_files=[record(p,ROOT) for p in sorted(ROOT.glob('*.py'))]+[record(p,ROOT) for p in sorted(ROOT.glob('*.cu'))],
                setup_seconds={name:r.setup_seconds for name,r in group.items()})
            write(dest/'manifest.json',result)
            for collection in [baseline,mapped]:
                for runner in collection.values():
                    for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize()
            emit('baseline_timing')
            with (dest/'baseline-samples.jsonl').open('x') as sink:
                samples=replay.dense.paired_probe(baseline,inputs,passes=2 if args.smoke else cfg['timing_passes'],seed=cfg['timing_seed']+args.replicate-1,sink=lambda row:sink.write(json.dumps(row)+'\n'))
            write(dest/'baseline-timing.json',{'indices':indices.tolist(),'samples':samples})
            emit('stage_timing')
            with (dest/'stage-samples.jsonl').open('x') as sink:
                def save(row):
                    tape=tapes[row['mode']];values=tape.times();total=tape.events[0].elapsed_time(tape.events[-1])*1000
                    if abs(sum(values)-total)>0.02:raise ValueError('Stage timing budget does not close')
                    sink.write(json.dumps({**row,'stage_names':tape.names,'stage_us':values,'partition_total_us':total})+'\n')
                replay.dense.paired_probe(mapped,inputs,passes=2 if args.smoke else cfg['timing_passes'],seed=cfg['timing_seed']+args.replicate-1,sink=save)
            blocks=8 if args.smoke else 338
            emit('qualification',target_blocks=blocks)
            vs=time.monotonic()
            def progress(**fields):
                elapsed=time.monotonic()-vs;done=fields['blocks']
                emit('qualification',**fields,input_tokens_per_second=done*2048/elapsed,remaining_seconds=(blocks-done)*elapsed/done)
            stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
            quality=replay.dense.compare_inputs(group,stream,cfg['numerical_bounds'],progress=progress)
            quality.update(blocks=blocks,documents=500 if blocks==338 else None,excluded_tail_tokens=1444 if blocks==338 else None,input_tokens=blocks*2048)
            write(dest/'quality.json',quality)
            result.update(qualified=all(quality['pass'].values()),loss=quality['loss'])
            if not result['qualified']:raise ValueError('Numerical qualification failed')
            if args.replicate==1 and not args.smoke:
                emit('traces')
                traces=[]
                for mode in cfg['modes']:
                    for i,ids in enumerate(inputs[:cfg['trace_inputs']]):
                        runner=baseline[mode];runner.stage(ids);runner();torch.cuda.synchronize()
                        graph=profiling.capture(runner,dest/f'trace-{mode}-i{i}-graph.json',cfg['trace_replays'])
                        tape=stages.Tape(annotations=True)
                        eager=profiling.capture(lambda:stages.forward(models[mode],ids,tape,mode!='native'),dest/f'trace-{mode}-i{i}-eager.json',1)
                        traces.append({'mode':mode,'input_index':i,'graph':graph,'eager':eager,'mapping':profiling.map_graph(graph,eager)})
                write(dest/'traces.json',traces)
                emit('diagnostics')
                diagnostics=module('run064_diagnostics',BASE/'diagnostics.py')
                for mode in ['generic_sparse','tile_sparse']:
                    diagnostics.collect(models[mode],models['native'],validation,dest/f'diagnostics-{mode}.json',lambda stage,**fields:emit(stage,mode=mode,**fields),checkpoint['architecture'],blocks=blocks)
                internal=module('run064_internal',ROOT/'internal.py');internal_rows=[]
                for mode in cfg['modes'][1:]:
                    emit('internal_cycles',mode=mode)
                    model=models[mode];saved={};originals=[]
                    for layer_index,layer in enumerate(model.gpt_neox.layers):
                        original=layer._run026_joint;originals.append(original)
                        def save_joint(h,z,r,j=original,l=layer_index):
                            saved[l]=(h.clone(),z.clone(),r.clone());return j(h,z,r)
                        layer._run026_joint=save_joint
                    try:
                        for i,ids in enumerate(inputs[:cfg['internal_inputs']]):
                            replay.scaffold.forward(model,ids)
                            for layer_index,joint in enumerate(originals):
                                row=internal.probe(joint,*saved[layer_index],cfg['internal_repeats'])
                                internal_rows.append({'mode':mode,'input_index':i,'layer':layer_index,**row})
                    finally:
                        for layer,joint in zip(model.gpt_neox.layers,originals):layer._run026_joint=joint
                write(dest/'internal-cycles.json',internal_rows)
            result.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),evidence_label='valid')
            emit('complete',qualified=True,loss=result['loss'])
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc))
    finally:
        result.update(elapsed_seconds=time.monotonic()-start,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write(dest/'manifest.json',result);write(dest/'result.json',result)
    return 0 if result['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
