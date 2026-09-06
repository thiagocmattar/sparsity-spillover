"""K043 hybrid short-row h/z versus frozen K036; development only."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN,ROOT,R25,R27,module,manifest,runtime,read_json,write_json,record,verify_record,dense
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--candidate',choices=['k043'],default='k043');a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity')
    cfg=read_json(RUN/'config.json')
    if a.condition not in cfg['development_conditions']:p.error('Registered development endpoints only')
    frozen=read_json(RUN/'final-policy-001.json')
    for row in frozen['sources']+frozen['dependencies']['files']:verify_record(row)
    runtime(torch);source=manifest();checkpoint=next(r for r in source['checkpoints'] if r['id']==a.condition)
    for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();result={'status':'running','arguments':vars(a)}
    def emit(stage,**fields):
        row={'stage':stage,'elapsed_seconds':time.monotonic()-started,**fields}
        write_json(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    try:
        paths=[ROOT/r['path'] for r in frozen['sources']]+[RUN/'87_k043_comparison.py']+list((RUN/f'candidates/{a.candidate}').iterdir())+list((RUN/'candidates/k042').iterdir())
        paths=sorted(set(p for p in paths if p.is_file()))
        for i,path in enumerate(paths):
            target=dest/'source'/f'{i:03d}-{path.name}';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        original=module('run028_k037_original',R27/'adapter.py')
        current=module('run028_k037_current',RUN/'candidates/k036/candidate.py')
        selected=module('run028_next_selected',RUN/f'candidates/{a.candidate}/candidate.py')
        scaffold=module('run028_k037_graph',RUN/'45_graph_forward.py')
        development=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
        inputs=[torch.tensor(row.copy(),device='cuda',dtype=torch.long)[None] for row in development[:32]]
        models={};runners={}
        with torch.inference_mode():
            for name in ['native','k036','selected','no_skip','attention_dense']:
                emit('loading',mode=name)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False
                if name!='native':original.install(net,'sparse')
                if name=='k036':current.install(net,shortcut=False,skip=True,projection_skip=True)
                elif name in {'selected','no_skip','attention_dense'}:
                    selected.install(net,shortcut=False,skip=name=='selected',projection_skip=name!='no_skip')
                models[name]=net
                if name=='native':
                    runners['native']=dense.DenseRunner(lambda ids,model=net:model(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                    runners['native'].prepare()
                runners[name+'_graph']=dense.DenseRunner(lambda ids,model=net:scaffold.forward(model,ids),inputs[0].clone(),'graph')
                runners[name+'_graph'].prepare()
            write_json(dest/'manifest.json',{'arguments':vars(a),'checkpoint':checkpoint,'inputs':source['inputs'],
                'sources':[record(p) for p in paths],'frozen_comparator_policy':record(RUN/'final-policy-001.json'),
                'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                'comparison':f'{a.candidate} versus K036, uniform no-prefix, matched graph; eager-native numerical anchor',
                'timing':'32 registered training inputs x7 paired passes; full338 validation; no interior timing for tuning'})
            for runner in runners.values():
                for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize();emit('timing')
            samples=dense.paired_probe(runners,inputs,passes=7,seed=2804)
            summaries={ref:dense.timing_summary(samples,reference=ref) for ref in runners}
            write_json(dest/'timing.json',{'samples':samples,'summary_by_reference':summaries})
            validation=np.memmap(verify_record(source['inputs']['validation']),dtype=np.int32,mode='r')
            if divmod(len(validation),2048)!=(338,1444):raise ValueError('Coverage changed')
            emit('validation')
            stream=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(338))
            quality=dense.compare_inputs(runners,stream,read_json(R25/'config.json')['calibration'],progress=lambda **f:emit('validation',**f))
            quality.update(blocks=338,documents=500,input_tokens=692224,excluded_tail_tokens=1444)
            write_json(dest/'quality.json',quality)
            result.update(status='complete',qualified=quality['pass'],loss=quality['loss'],timing=summaries)
            emit('complete',qualified=quality['pass'],loss=quality['loss'],timing=summaries['native_graph'])
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write_json(dest/'result.json',result)


if __name__=='__main__':main()
