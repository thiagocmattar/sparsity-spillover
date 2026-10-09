"""Exploratory matched projection screen; retain every candidate, including failures."""
import argparse
import time
import torch
from common import RUN, config, read, verify, record, runtime, support, write, shortlist, emit, manifest_identity, check_runtime
from kernel import make, dense
from measurement import Graph, compare, paired_timing, summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--development',action='store_true')
    args=p.parse_args();cfg=config();runtime(cfg['runtime_seed']);identity=manifest_identity(args.development)
    check_runtime(args.development)
    root=RUN/('prelaunch/development-capture' if args.development else 'artifacts/capture')
    manifest=read(root/'manifest.json');row=next(x for x in manifest['files'] if x['path'].endswith(args.model+'.pt'))
    data=torch.load(verify(row),map_location='cpu',weights_only=True)
    dest=RUN/('prelaunch/development-screen' if args.development else 'artifacts/screen')/args.model
    dest.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    linear=torch.nn.Linear(128,50304,bias=False,device='cuda',dtype=torch.float32)
    linear.weight.data.copy_(data['weight']);spec=data['gate'];pairs=[]
    with torch.inference_mode():
        for pre in data['pre']:
            pre=pre.cuda();live=pre.reshape(-1,128)[::16,::16]>spec['kappa']
            x=(pre.reshape(128,16,8,16)*live[:,None,:,None]).reshape(2048,128).half()
            pairs.append([x,support(pre,spec['kappa'])])
        graphs={};heads={};qualification={};excluded={};work_counts={}
        native=dense(linear)
        graphs['native']=Graph(lambda x,s:torch.nn.functional.linear(x,native.packed_weight),pairs[0])
        for variant in cfg['variants']:
            for skip in (False,True):
                mode=variant['name']+('/sparse' if skip else '/no_skip')
                head=make(linear,spec,variant,skip);head(data['pre'][0].cuda())
                def operation(x,s,h=head):h.x_half=x;h.support=s;return h.project()
                graph=Graph(operation,pairs[0]);checks=[];issued=0;bypassed=0
                for x,s in pairs:
                    graph.stage([x,s]);check=compare(torch.nn.functional.linear(x,native.packed_weight),graph(),cfg['numerical_bounds'])
                    checks.append(check)
                    # Separate untimed Count specialization on the actual timed operands.
                    head.x_half=x;head.support=s;head.project(count=True)
                    counts=head.stats.sum((0,1)).tolist();issued+=counts[0];bypassed+=counts[1]
                qualification[mode]=checks
                work_counts[mode]={'issued_mma':issued,'skipped_mma':bypassed,'potential_mma':issued+bypassed,
                                   'input_count':len(pairs),'coverage':'timed real head operands; separate untimed Count specialization'}
                if not all(c['pass'] for c in checks):
                    excluded[mode]='Failed unchanged numerical bounds on timed operands';del graph,head
                else:heads[mode]=head;graphs[mode]=graph
                emit(dest,'qualify',mode=mode,pass_count=sum(c['pass'] for c in checks),total=len(checks),elapsed_seconds=time.monotonic()-started)
        with (dest/'samples.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            import json
            def sink(r):stream.write(json.dumps(r)+'\n');stream.flush()
            samples=paired_timing(graphs,pairs,passes=2 if args.development else cfg['timing_passes'],seed=cfg['timing_seed'],sink=sink)
        timings=summarize(samples);nested={}
        for variant in cfg['variants']:
            name=variant['name']
            if all(name+'/'+m in timings for m in ('no_skip','sparse')):
                nested[name]={m:timings[name+'/'+m] for m in ('no_skip','sparse')}
        chosen=shortlist(nested,cfg['shortlist_limit']) if len(nested)>1 else []
        # Synthetic zero-MMA path is a diagnostic, not a runtime lower bound.
        zero=pairs[0][1].new_zeros(pairs[0][1].shape)
        legacy=heads.get('legacy_n256/sparse');fixed=None
        if legacy is not None:
            floor=Graph(lambda x,s:(setattr(legacy,'x_half',x),setattr(legacy,'support',s),legacy.project())[-1],[pairs[0][0],zero])
            fixed=summarize(paired_timing({'zero_mma':floor},[[pairs[0][0],zero]],passes=20,seed=cfg['timing_seed'],sink=lambda r:None))
        result={'status':'complete','identity':identity,'input_snapshot':row,'timings':timings,
                'qualification':qualification,'excluded':excluded,'shortlist':chosen,'zero_mma_diagnostic':fixed,'work_counts':work_counts,
                'interpretation':'exploratory selection on the same fixed inputs; fresh final processes required; no quality or practical frontier claim',
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'elapsed_seconds':time.monotonic()-started}
        write(dest/'result.json',result);emit(dest,'complete',shortlist=chosen,elapsed_seconds=result['elapsed_seconds'])


if __name__=='__main__':main()
