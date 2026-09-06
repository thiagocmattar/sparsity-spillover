"""Paired six-mode development attribution; full validation, no final selection."""
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
    p.add_argument('--candidate',default='k029',choices=['k027','k029','k030','k031','k032','k033','k034'])
    p.add_argument('--inputs',type=int,default=8)
    p.add_argument('--passes',type=int,default=7)
    p.add_argument('--execution',choices=['native','graph'],default='native')
    p.add_argument('--full-validation',action='store_true',default=True)
    a=p.parse_args()
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
        paths=[RUN/'36_attribution.py',RUN/'common.py',RUN/'config.json']+[f for f in (RUN/f'candidates/{a.candidate}').iterdir() if f.is_file()]
        for dependency in ['k020','k021']:
            paths+=list((RUN/f'candidates/{dependency}').glob('*.py'))+list((RUN/f'candidates/{dependency}').glob('*.cu'))
        if a.candidate in {'k032','k033','k034'}:paths+=[f for f in (RUN/'candidates/k031').iterdir() if f.is_file()]
        if a.candidate in {'k033','k034'}:paths+=[f for f in (RUN/'candidates/k032').iterdir() if f.is_file()]
        if a.candidate=='k034':paths+=[f for f in (RUN/'candidates/k033').iterdir() if f.is_file()]
        if a.execution=='graph':paths+=[RUN/'45_graph_forward.py']
        for path in paths:
            target=dest/'source'/path.relative_to(RUN)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        models={};runners={}
        with torch.inference_mode():
            for mode in ['native','previous','sparse','attention_dense','no_prefix','all_dense']:
                emit('loading',mode=mode)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False
                if mode!='native':original.install(net,'sparse')
                if mode not in {'native','previous'}:
                    coverage=selected.install(net,shortcut=mode!='no_prefix',skip=mode not in {'attention_dense','all_dense'},projection_skip=mode!='all_dense')
                    result.setdefault('implementation_coverage',{})[mode]=coverage
                models[mode]=net
                if a.execution=='graph':
                    scaffold=module('run028_graph_forward',RUN/'45_graph_forward.py')
                    forward=lambda ids,model=net:scaffold.forward(model,ids)
                else:forward=lambda ids,model=net:model(input_ids=ids,use_cache=False).logits
                runners[mode]=dense.DenseRunner(forward,inputs[0].clone(),a.execution)
                runners[mode].prepare()
            if a.execution=='graph':
                runners['eager_stock']=dense.DenseRunner(lambda ids:models['native'](input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                runners['eager_stock'].prepare()
            write_json(dest/'manifest.json',{'checkpoint':checkpoint,'inputs':source['inputs'],'arguments':vars(a),
                'sources':[record(path) for path in paths],
                'previous_sources':[record(R27/name) for name in ['adapter.py','kernel.cu','run027_common.py']]
                    +[record(R26/f'autoresearch/candidates/{k}/{name}') for k in ['k018','k019'] for name in ['candidate.py','kernel.cu']],
                'helper_sources':[record(R25/name) for name in ['run025_common.py','measurement.py','config.json','autoresearch/dense_probe/probe.py']]
                    +[record(path) for path in (ROOT/'src').rglob('*.py')],
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'topology':topology_metadata(models['native']),
                'runner_setup_seconds':{mode:runner.setup_seconds for mode,runner in runners.items()},
                'upstream_dependencies':{'inventory':record(RUN/'runtime/vendor/inventory.json'),'sources':read_json(RUN/'runtime/vendor/inventory.json')['sources']},
                'coverage':'development timing; full338 validation; matched execution for every comparator'})
            for runner in runners.values():
                for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize()
            emit('timing',inputs=len(inputs),passes=a.passes)
            samples=dense.paired_probe(runners,inputs,passes=a.passes,seed=2804)
            timing=dense.timing_summary(samples,reference='native')
            write_json(dest/'timing.json',{'samples':samples,'native_reference':timing,'previous_reference':dense.timing_summary(samples,reference='previous'),
                'attention_dense_reference':dense.timing_summary(samples,reference='attention_dense'),
                'no_prefix_reference':dense.timing_summary(samples,reference='no_prefix'),
                'all_dense_reference':dense.timing_summary(samples,reference='all_dense')})
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
