"""Frozen nine-mode, all35/three-process study; failures are retained."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN,ROOT,R25,R27,module,manifest,runtime,read_json,write_json,record,verify_record,dense
from sparsity_research.pythia import load_checkpoint_pythia,topology_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',required=True);p.add_argument('--replicate',type=int,required=True)
    p.add_argument('--attempt',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--policy',default='final-policy-002.json')
    a=p.parse_args()
    if not a.attempt.replace('-','').isalnum() or not 1<=a.replicate<=3:p.error('Invalid identity')
    if '/' in a.policy or '\\' in a.policy or not a.policy.endswith('.json'):p.error('Local policy basename required')
    policy_path=RUN/a.policy;policy=read_json(policy_path)
    for row in policy['sources']:verify_record(row)
    for row in policy['dependencies']['files']:verify_record(row)
    runtime(torch);source=manifest();cfg=read_json(RUN/'config.json')
    checkpoint=next(r for r in source['checkpoints'] if r['id']==a.condition)
    for row in checkpoint['files']+checkpoint['provenance']:verify_record(row)
    dest=RUN/'artifacts'/a.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();result={'status':'running','arguments':vars(a),'condition':a.condition,
        'family':checkpoint['family'],'dose':checkpoint['dose'],'historical':checkpoint['historical'],
        'canonical_logical_products':checkpoint['canonical_logical_products'],'policy':record(policy_path)}
    stage_started={}
    def emit(stage,**fields):
        now=time.monotonic();stage_started.setdefault(stage,now)
        row={'stage':stage,'elapsed_seconds':now-started,**fields}
        if stage=='validation' and fields.get('blocks',0)>0:
            elapsed=now-stage_started[stage]
            if elapsed>0:
                row.update(input_tokens_per_second=fields['blocks']*2048/elapsed,
                           remaining_seconds=max(0,quality_blocks-fields['blocks'])*elapsed/fields['blocks'])
        write_json(dest/'status.json',row)
        with (dest/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    try:
        validation=np.memmap(verify_record(source['inputs']['validation']),dtype=np.int32,mode='r')
        if divmod(len(validation),2048)!=(338,1444):raise ValueError('Validation coverage changed')
        indices=np.random.default_rng(cfg['timing_seed']).choice(338,cfg['timing_inputs'],replace=False)
        if a.smoke:indices=indices[:4]
        inputs=[torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in indices]
        original=module('run028_study_original',R27/'adapter.py')
        selected=module('run028_study_candidate',RUN/f"candidates/{policy['candidate']}/candidate.py")
        scaffold=module('run028_study_graph',RUN/'45_graph_forward.py')
        diagnostic=module('run028_study_diagnostic',RUN/'115_hybrid_diagnostics.py')
        unfused=module('run028_study_unfused',RUN/'candidates/k049/candidate.py')
        models={};runners={};coverage={}
        snapshots=[]
        for index,path in enumerate([policy_path]+[ROOT/r['path'] for r in policy['sources']]):
            target=dest/'source'/f'{index:03d}-{path.name}'
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            snapshots.append({'original':record(path),'snapshot':str(target.relative_to(dest))})
        write_json(dest/'source-map.json',snapshots)
        with torch.inference_mode():
            for mode in ['native','previous','unfused','sparse','no_skip','attention_dense']:
                emit('loading',mode=mode)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False
                if mode!='native':original.install(net,'sparse')
                if mode=='unfused':coverage[mode]=unfused.install(net,shortcut=False,skip=True,projection_skip=True)
                if mode not in {'native','previous','unfused'}:
                    coverage[mode]=selected.install(net,shortcut=policy['shortcut'],skip=mode=='sparse',projection_skip=mode!='no_skip')
                models[mode]=net
                if mode in {'native','previous','sparse'}:
                    runners[mode]=dense.DenseRunner(lambda ids,model=net:model(input_ids=ids,use_cache=False).logits,inputs[0].clone(),'native')
                    runners[mode].prepare()
                key=mode+'_graph'
                runners[key]=dense.DenseRunner(lambda ids,model=net:scaffold.forward(model,ids),inputs[0].clone(),'graph')
                runners[key].prepare()
            write_json(dest/'manifest.json',{'checkpoint':checkpoint,'inputs':source['inputs'],'arguments':vars(a),
                'policy':policy,'sources':[record(policy_path)],
                'torch':torch.__version__,'torch_git':torch.version.git_version,'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(),'topology':topology_metadata(models['native']),
                'implementation_coverage':coverage,'runner_setup_seconds':{k:v.setup_seconds for k,v in runners.items()},
                'timer':'synchronized full resident-input forward and50304 logits; identical staging excluded',
                'comparisons':'eager vs eager; graph vs graph; sparse-path ablations share norm fusion; unfused K049 isolates fusion; never cross-mode as sparsity gain'})
            for runner in runners.values():
                for ids in inputs:runner.stage(ids);runner()
            torch.cuda.synchronize();passes=2 if a.smoke else cfg['timing_passes']
            emit('timing',inputs=len(inputs),passes=passes)
            samples=dense.paired_probe(runners,inputs,passes=passes,seed=cfg['timing_seed']+a.replicate-1)
            summaries={reference:dense.timing_summary(samples,reference=reference) for reference in runners}
            write_json(dest/'timing.json',{'indices':indices.tolist(),'samples':samples,'summary_by_reference':summaries})
            result['timing']=summaries
            quality_blocks=8 if a.smoke else cfg['validation_blocks']
            emit('validation',target_blocks=quality_blocks)
            quality_inputs=(torch.tensor(validation[i*2048:(i+1)*2048].copy(),device='cuda',dtype=torch.long)[None] for i in range(quality_blocks))
            quality=dense.compare_inputs(runners,quality_inputs,read_json(R25/'config.json')['calibration'],progress=lambda **f:emit('validation',**f))
            quality.update(blocks=quality_blocks,documents=500 if not a.smoke else None,
                           input_tokens=quality_blocks*2048,excluded_tail_tokens=1444 if not a.smoke else None)
            write_json(dest/'quality.json',quality)
            result.update(qualified=quality['pass'],loss=quality['loss'],loss_delta=quality['loss_delta'],validation_blocks=quality_blocks)
            if a.replicate==1:
                emit('diagnostics',target_blocks=quality_blocks,loss=quality['loss'])
                diagnostic.collect(models['sparse'],models['native'],validation,dest/'diagnostics.json',emit,
                                   checkpoint['canonical_logical_products']['architecture_maximum'],blocks=quality_blocks)
            result.update(status='complete',peak_allocated_bytes=torch.cuda.max_memory_allocated())
            emit('complete',qualified=quality['pass'],loss=quality['loss'],timing=summaries['native'])
    except BaseException as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc());emit('failed',error=str(exc));raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started;write_json(dest/'result.json',result)


if __name__=='__main__':main()
