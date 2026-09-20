"""One checkpoint/process; paired stock, fusion-only, sparse and no-skip modes."""
import argparse
import gc
import json
import os
import platform
import time
import traceback
import numpy as np
import torch
import transformers
import adapter
from diagnostics import collect
from run027_common import RUN,ROOT,R25,R26,dense,inputs,read_json,write_json,record,verify_record
from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',required=True)
    parser.add_argument('--replicate',type=int,required=True)
    parser.add_argument('--attempt',required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.attempt.replace('-','').isalnum() or not 1<=args.replicate<=3: parser.error('Invalid identity')
    cfg=read_json(RUN/'config.json')
    manifest=inputs()
    checkpoint=next(r for r in manifest['checkpoints'] if r['id']==args.condition)
    for row in checkpoint['files']+checkpoint['provenance']: verify_record(row)
    dest=RUN/'artifacts'/args.attempt
    dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'status':'running','arguments':vars(args),'condition':args.condition,
        'family':checkpoint['family'],'dose':checkpoint['dose'],'historical':checkpoint['historical'],
        'canonical_logical_products':checkpoint['canonical_logical_products'],'qualified':{},'timing':{}}

    def emit(stage,**fields):
        value={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':stage,
               'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',value)
        with (dest/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(value,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(value),flush=True)

    try:
        runtime=read_json(R25/'config.json')['runtime']
        if (torch.__version__.split('+')[0],transformers.__version__,np.__version__)!=(runtime['torch'],runtime['transformers'],runtime['numpy']):
            raise ValueError('Pinned runtime differs')
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG'): raise ValueError('Use default workspace')
        torch.manual_seed(2703)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        validation=np.memmap(verify_record(manifest['inputs']['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444): raise ValueError('Validation changed')
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False)
        if args.smoke: indices=indices[:4]
        timing_inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        models={}; runners={}
        with torch.inference_mode():
            for mode in cfg['modes']:
                emit('loading',mode=mode)
                model=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                model.set_attn_implementation('sdpa'); model.config.use_cache=False
                topology=topology_metadata(model)
                if mode!='native': adapter.install(model,mode)
                models[mode]=model
                runners[mode]=dense.DenseRunner(lambda ids,net=model:net(input_ids=ids,use_cache=False).logits,timing_inputs[0].clone(),'native')
                runners[mode].prepare()
            source_files=[p for p in RUN.glob('*') if p.suffix in {'.py','.cu','.json'}]
            write_json(dest/'manifest.json',{'checkpoint':checkpoint,'inputs':manifest['inputs'],
                'sources':[record(p) for p in source_files]+manifest['kernel_sources'],
                'source_helpers':[record(R25/name) for name in ['autoresearch/dense_probe/probe.py','measurement.py','run025_common.py','config.json']]
                    +[record(p) for p in (ROOT/'src').rglob('*.py')],
                'topology':topology,'python':platform.python_version(),'torch':torch.__version__,
                'transformers':transformers.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),
                'arguments':vars(args),'timer':'synchronized full forward/full logits; equal resident input staging excluded'})
            for runner in runners.values():
                for ids in timing_inputs: runner.stage(ids); runner()
            torch.cuda.synchronize()
            passes=2 if args.smoke else cfg['timing_passes']
            emit('timing',inputs=len(timing_inputs),passes=passes)
            samples=dense.paired_probe(runners,timing_inputs,passes=passes,seed=cfg['timing_seed']+args.replicate-1)
            summaries={ref:dense.timing_summary(samples,reference=ref) for ref in cfg['modes']}
            write_json(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary_by_reference':summaries})
            result['timing']=summaries
            blocks=8 if args.smoke else cfg['validation_blocks']
            emit('validation',blocks=blocks)
            inputs_iter=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(blocks))
            quality=dense.compare_inputs(runners,inputs_iter,read_json(R25/'config.json')['calibration'],
                progress=lambda **fields:emit('validation',**fields))
            quality.update(documents=500 if not args.smoke else None,blocks=blocks,input_tokens=blocks*2048,excluded_tail=1444)
            write_json(dest/'quality.json',quality)
            result.update(qualified=quality['pass'],loss=quality['loss'],loss_delta=quality['loss_delta'],
                validation_blocks=blocks,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if args.replicate==1 and not args.smoke:
                emit('diagnostics',loss=quality['loss'])
                collect(models['native'],validation,dest/'diagnostics.json',emit)
            result['status']='complete'
            emit('complete',qualified=quality['pass'],loss=quality['loss'],timing=summaries['native'])
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        emit('failed',error=str(exc)); raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        write_json(dest/'result.json',result)


if __name__=='__main__':main()
