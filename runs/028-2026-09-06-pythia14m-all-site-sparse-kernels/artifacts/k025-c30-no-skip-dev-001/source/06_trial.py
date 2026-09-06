"""Immutable full-model development trial against stock and previous winner."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN, ROOT, R25, R26, R27, manifest, module, runtime, read_json, write_json, record, verify_record, dense
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt',required=True)
    p.add_argument('--condition',required=True)
    p.add_argument('--candidate',default='k020',choices=['k020','k021','k022','k023','k024','k025'])
    p.add_argument('--inputs',type=int,default=8)
    p.add_argument('--passes',type=int,default=3)
    p.add_argument('--full-validation',action='store_true')
    p.add_argument('--no-prefix',action='store_true')
    p.add_argument('--round-p',action='store_true')
    p.add_argument('--projection-no-skip',action='store_true')
    a=p.parse_args()
    if a.projection_no_skip and a.candidate!='k025':p.error('Projection toggle is currently a K025 attribution control only')
    if not a.attempt.replace('-','').isalnum() or not 1<=a.inputs<=64 or a.passes<1:p.error('Invalid trial identity/count')
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();result={'status':'running','arguments':vars(a)}
    def emit(stage,**fields):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        runtime(torch)
        source=manifest();cfg=read_json(RUN/'config.json')
        checkpoint=next(r for r in source['checkpoints'] if r['id']==a.condition)
        if a.condition not in cfg['development_conditions'] and not a.full_validation:
            raise ValueError('Unregistered development endpoint')
        for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
        dev=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
        inputs=[torch.tensor(row.copy(),device='cuda',dtype=torch.long)[None] for row in dev[:a.inputs]]
        original=module('run028_previous_trial',R27/'adapter.py')
        selected=module('run028_selected_trial',RUN/f'candidates/{a.candidate}/candidate.py')
        paths=[RUN/'06_trial.py',RUN/'common.py',RUN/'config.json']+list((RUN/f'candidates/{a.candidate}').glob('*.py'))+list((RUN/f'candidates/{a.candidate}').glob('*.cu'))
        if a.candidate!='k020':paths+=list((RUN/'candidates/k020').glob('*.py'))+list((RUN/'candidates/k020').glob('*.cu'))
        if a.candidate in {'k022','k023','k024','k025'}:paths+=list((RUN/'candidates/k021').glob('*.py'))+list((RUN/'candidates/k021').glob('*.cu'))
        for path in paths:
            target=dest/'source'/path.relative_to(RUN)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        models={};runners={}
        with torch.inference_mode():
            for mode in ['native','previous','candidate']:
                emit('loading',mode=mode)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False
                if mode!='native':original.install(net,'sparse')
                if mode=='candidate':
                    kwargs={'skip':not a.projection_no_skip} if a.candidate=='k025' else {}
                    result['implementation_coverage']=selected.install(net,shortcut=not a.no_prefix,round_p=a.round_p,**kwargs)
                models[mode]=net
                runners[mode]=dense.DenseRunner(lambda ids,model=net:model(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                runners[mode].prepare()
            write_json(dest/'manifest.json',{'checkpoint':checkpoint,'inputs':source['inputs'],'arguments':vars(a),
                'sources':[record(path) for path in paths],
                'previous_sources':[record(R27/name) for name in ['adapter.py','kernel.cu','run027_common.py']]
                    +[record(R26/f'autoresearch/candidates/{k}/{name}') for k in ['k018','k019'] for name in ['candidate.py','kernel.cu']],
                'helper_sources':[record(R25/name) for name in ['run025_common.py','measurement.py','config.json','autoresearch/dense_probe/probe.py']]
                    +[record(path) for path in (ROOT/'src').rglob('*.py')],
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'topology':topology_metadata(models['native']),
                'coverage':'development timing; full validation only if explicitly selected'})
            for runner in runners.values():
                for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize()
            emit('timing',inputs=len(inputs),passes=a.passes)
            samples=dense.paired_probe(runners,inputs,passes=a.passes,seed=2804)
            timing=dense.timing_summary(samples,reference='native')
            write_json(dest/'timing.json',{'samples':samples,'native_reference':timing,'previous_reference':dense.timing_summary(samples,reference='previous')})
            emit('quality',timing=timing)
            quality_inputs=inputs
            if a.full_validation:
                val=np.memmap(verify_record(source['inputs']['validation']),dtype=np.int32,mode='r')
                if divmod(len(val),2048)!=(338,1444):raise ValueError('Coverage changed')
                quality_inputs=(torch.tensor(val[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(338))
            quality=dense.compare_inputs(runners,quality_inputs,read_json(R25/'config.json')['calibration'],progress=lambda **x:emit('quality',**x))
            write_json(dest/'quality.json',quality)
            result.update(status='complete',timing=timing,qualified=quality['pass'],loss=quality['loss'],loss_delta=quality['loss_delta'],
                          validation_blocks=338 if a.full_validation else 0,development_quality_blocks=0 if a.full_validation else a.inputs,
                          canonical_logical_products=checkpoint['canonical_logical_products'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete',qualified=quality['pass'],loss=quality['loss'],timing=timing)
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write_json(dest/'result.json',result)


if __name__=='__main__':main()
