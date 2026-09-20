"""Diagnostic CUDA-kernel breakdown; not a replacement for paired timing."""
import argparse
from collections import defaultdict
import gc
import shutil
import time
import traceback
import numpy as np
import torch
import transformers
from common import RUN, ROOT, R27, module, manifest, runtime, record, verify_record, write_json, read_json
from sparsity_research.pythia import load_checkpoint_pythia


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',required=True);args=parser.parse_args()
    if not args.attempt.replace('-','').isalnum():parser.error('Simple attempt identity')
    dest=RUN/'artifacts'/args.attempt;dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();summaries={}
    try:
        runtime(torch);source=manifest();policy=read_json(RUN/'final-policy-001.json')
        for entry in policy['sources']+policy['dependencies']['files']:verify_record(entry)
        checkpoint=next(row for row in source['checkpoints'] if row['id']=='c30')
        for entry in checkpoint['files']+checkpoint['provenance']:verify_record(entry)
        candidates=['k036','k038','k039','k041','k042'];modes=['native','previous']+candidates
        paths=sorted(set([ROOT/entry['path'] for entry in policy['sources']]+[RUN/'84_kernel_profile.py']+
            [path for name in candidates for path in (RUN/f'candidates/{name}').iterdir() if path.is_file()]))
        snapshots=[]
        for index,path in enumerate(paths):
            target=dest/'source'/f'{index:03d}-{path.name}';target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target);snapshots.append({'snapshot':str(target.relative_to(dest)),'original':record(path)})
        write_json(dest/'manifest.json',{'arguments':vars(args),'checkpoint':checkpoint,'inputs':source['inputs'],
            'sources':snapshots,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
            'input':'training block0','replays':10,'execution':'same graph scaffold; full logits; BF16 B1 T2048',
            'limits':'CUPTI instrumentation may perturb timings; summed kernel durations are not end-to-end latency or a speedup estimate'})
        data=np.memmap(verify_record(source['inputs']['development']),dtype=np.int32,mode='r').reshape(-1,2048)
        ids=torch.tensor(data[0].copy(),device='cuda',dtype=torch.long)[None]
        original=module('run028_profile_original',R27/'adapter.py')
        scaffold=module('run028_profile_scaffold',RUN/'45_graph_forward.py')
        from common import dense
        with torch.inference_mode():
            for name in modes:
                quality_folder=RUN/'artifacts'/('final-c30-p1-001' if name in {'native','previous','k036'} else f'{name}-c30-graph-001')
                quality=read_json(quality_folder/'result.json')
                quality_mode={'native':'native_graph','previous':'previous_graph','k036':'sparse_graph'}.get(name,'selected_graph')
                if quality['status']!='complete' or not quality['qualified'][quality_mode]:raise ValueError('Profile requires prior full-validation qualification')
                if name not in {'native','previous','k036'}:
                    for entry in read_json(quality_folder/'manifest.json')['sources']:verify_record(entry)
                net=load_checkpoint_pythia(transformers.AutoModelForCausalLM,ROOT/checkpoint['checkpoint'],torch=torch).to('cuda',dtype=torch.bfloat16).eval()
                net.set_attn_implementation('sdpa');net.config.use_cache=False
                if name!='native':original.install(net,'sparse')
                if name in candidates:
                    candidate=module(f'run028_profile_{name}',RUN/f'candidates/{name}/candidate.py')
                    candidate.install(net,shortcut=False,skip=True,projection_skip=True)
                runner=dense.DenseRunner(lambda value,model=net:scaffold.forward(model,value),ids.clone(),'graph')
                runner.prepare();runner.stage(ids)
                for _ in range(5):runner()
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
                    for _ in range(10):runner()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(str(dest/f'{name}-trace.json'))
                events=[{'name':event.name,'device_us':float(event.device_time_total)} for event in profile.events() if 'CUDA' in str(event.device_type)]
                if not events or not sum(event['device_us'] for event in events)>0:raise ValueError('No CUDA kernel durations captured')
                groups=defaultdict(lambda:{'calls':0,'total_us':0.})
                for event in events:
                    groups[event['name']]['calls']+=1;groups[event['name']]['total_us']+=event['device_us']
                rows=sorted([{'name':key,**value,'us_per_replay':value['total_us']/10} for key,value in groups.items()],key=lambda row:-row['total_us'])
                summaries[name]={'gpu_events':len(events),'sum_kernel_us_per_replay':sum(event['device_us'] for event in events)/10,'by_kernel':rows,
                    'qualification_source':record(quality_folder/'result.json'),
                    'qualification_manifest':record(quality_folder/'manifest.json')}
                write_json(dest/f'{name}-events.json',events);write_json(dest/'summary.json',summaries)
                print({'mode':name,'gpu_events':len(events),'sum_kernel_us_per_replay':summaries[name]['sum_kernel_us_per_replay']},flush=True)
                del profile,runner,net;gc.collect();torch.cuda.empty_cache()
        write_json(dest/'result.json',{'status':'complete','modes':modes,'elapsed_seconds':time.monotonic()-started})
    except BaseException as exc:
        write_json(dest/'result.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
