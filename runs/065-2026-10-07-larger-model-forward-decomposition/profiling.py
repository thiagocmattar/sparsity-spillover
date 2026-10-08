"""Retain raw CUPTI timelines and map kernel launches to annotated eager stages."""
import collections
import json
import torch
from accounting import intervals_union


def capture(fn,path,repeats):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeats):fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    trace=json.loads(path.read_text())['traceEvents']
    stages=[e for e in trace if e.get('cat')=='user_annotation' and e.get('name','').startswith('stage::')]
    launches={}
    for e in trace:
        correlation=e.get('args',{}).get('correlation')
        if correlation is None or 'cuda' not in e.get('cat','').lower() or e.get('cat')=='kernel':continue
        owners=[s for s in stages if s.get('tid')==e.get('tid') and s['ts']<=e['ts'] and e['ts']+e.get('dur',0)<=s['ts']+s['dur']]
        if owners:launches[correlation]=min(owners,key=lambda s:s['dur'])['name'].removeprefix('stage::')
    kernels=sorted([e for e in trace if e.get('cat')=='kernel'],key=lambda e:e['ts'])
    rows=[{'name':e['name'],'ts_us':e['ts'],'duration_us':e['dur'],'stream':e.get('args',{}).get('stream'),
           'correlation':e.get('args',{}).get('correlation'),'stage':launches.get(e.get('args',{}).get('correlation')),
           'graph_node':e.get('args',{}).get('graph node id')} for e in kernels]
    groups=collections.defaultdict(list)
    for row in rows:groups[row['name']].append(row['duration_us'])
    return {'repeats':repeats,'kernels':rows,'kernel_summary':[{'name':name,'count':len(values),'total_us':sum(values),'mean_us':sum(values)/len(values)} for name,values in groups.items()],
            'gpu_active_union_us':intervals_union([(r['ts_us'],r['ts_us']+r['duration_us']) for r in rows]),
            'gpu_span_us':rows[-1]['ts_us']+rows[-1]['duration_us']-rows[0]['ts_us'] if rows else None,
            'annotated_cpu_stages':[{'stage':s['name'].removeprefix('stage::'),'duration_us':s['dur']} for s in stages]}


def map_graph(graph,eager):
    expected=eager['kernels'];actual=graph['kernels'];count=len(expected)
    valid=bool(count and len(actual)==count*graph['repeats'] and all(r['stage'] for r in expected))
    valid=valid and all(row['name']==expected[i%count]['name'] for i,row in enumerate(actual))
    if not valid:return {'mapped':False,'reason':'Kernel count, exact launch signature or eager stage attribution mismatch','eager_kernels':count,'graph_kernels':len(actual)}
    rows=[{**row,'stage':expected[i%count]['stage'],'replay':i//count} for i,row in enumerate(actual)]
    return {'mapped':True,'kernels':rows,'kernels_per_replay':count,'mapping':'Exact full ordered kernel-name equality between annotated eager and uninstrumented graph, every replay'}
