"""Shape-supported native GEMM attribution at both sizes; profiles are diagnostic."""
from collections import defaultdict
from io_utils import RUN, read, record, write


def signature(event):
    args=event['args']
    return event['name'],tuple(args['grid']),tuple(args['block'])


def native_mapping(events,hidden):
    cpu={e['args']['External id']:e for e in events
         if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    shapes={(hidden,3*hidden):'a',(hidden,4*hidden):'m',
            (4*hidden,hidden):'h',(hidden,hidden):'z',(hidden,50304):'head'}
    mapping=defaultdict(set)
    for event in events:
        if event.get('cat')!='kernel':continue
        owner=cpu.get(event['args'].get('External id'),{})
        if owner.get('name') not in ('aten::addmm','aten::mm'):continue
        dims=owner['args']['Input Dims']
        matrix,weight=(dims[1],dims[2]) if owner['name']=='aten::addmm' else (dims[0],dims[1])
        assert matrix[0]==2048 and matrix[1]==weight[0]
        mapping[signature(event)].add(shapes[tuple(weight)])
    assert set().union(*mapping.values())==set(shapes.values())
    assert all(len(value)==1 or value <= {'h','z'} for value in mapping.values()),mapping
    return {key:('h_z_gemms' if value <= {'h','z'} else next(iter(value)))
            for key,value in mapping.items()}


def custom_head_mapping(events):
    """Match launches inside the explicit output-projection CPU annotation."""
    spans=[e for e in events if e.get('name')=='run042_dense_head' and 'dur' in e]
    launches={e['args']['correlation']:e for e in events
              if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
    mapping={}
    for event in events:
        if event.get('cat')!='kernel':continue
        launch=launches.get(event['args'].get('correlation'))
        if launch and any(span['tid']==launch['tid'] and
                          span['ts']<=launch['ts']<=span['ts']+span['dur'] for span in spans):
            mapping[signature(event)]='head'
    if spans:assert mapping,'Annotated head found, but no correlated GPU launch'
    return mapping


def summarize(events,mapping,custom,inputs=4,layers=6):
    rows=defaultdict(lambda:{'calls':0,'duration_us':0.})
    for event in events:
        if event.get('cat')!='kernel':continue
        name=event['name'];group=mapping.get(signature(event),'other')
        if 'flash' in name and ('kernel<' in name or 'kernel(' in name):group='attention'
        if custom:
            if 'joint<' in name:group='fused_h_z'
            elif 'parallel_rows<' in name:group='h_z_parallel_inspection_and_scalar'
            elif 'scalar_only<' in name:group='h_z_scalar_path'
            elif 'inspect_rows(' in name:group='h_z_inspection'
            elif 'norm_pair(' in name:group='paired_norm'
            elif 'rope_gate(' in name:group='rope_and_gates'
            elif group=='other' and name.startswith('_ZN7cutlass'):group='custom_a_m'
        rows[group]['calls']+=1;rows[group]['duration_us']+=event['dur']
    total=sum(r['duration_us'] for r in rows.values())
    for row in rows.values():
        row['ms_per_input']=row['duration_us']/inputs/1000
        row['gpu_kernel_time_fraction']=row['duration_us']/total
    if not custom:
        assert all(rows[op]['calls']==layers*inputs for op in ('a','m'))
        assert rows['h_z_gemms']['calls']==2*layers*inputs
        assert rows['head']['calls']==inputs
    return dict(rows)


def main():
    rows=[];sources=[]
    for folder in sorted((RUN/'artifacts/attempts').glob('*')):
        path=folder/'profile-native-eager.json'
        if not path.exists():continue
        result=read(folder/'result.json')
        if result['status']!='complete' or not result.get('qualified'):continue
        hidden=128 if result['condition'].startswith('m14-') else 512
        mapping=native_mapping(read(path)['traceEvents'],hidden)
        sources.append(record(path));groups={}
        eager=folder/'profile-candidate-eager.json'
        custom_mapping={**mapping,**custom_head_mapping(read(eager)['traceEvents'])}
        sources.append(record(eager))
        for mode in ('native','candidate'):
            path=folder/f'profile-{mode}-graph.json'
            groups[mode]=summarize(read(path)['traceEvents'],custom_mapping if mode=='candidate' else mapping,mode=='candidate')
            sources.append(record(path))
        rows.append({'attempt':folder.name,'condition':result['condition'],
                     'candidate':result['candidate'],'hidden':hidden,'groups':groups})
    assert rows,'No completed profiles'
    write(RUN/'results/profile-attribution-v2.json',{'rows':rows,'sources':sources,'script':record(__file__),
      'coverage':'Four declared timing inputs, separate profiling passes; first fresh process for each profiled condition.',
      'method':'Native addmm/mm operand dimensions map exact GPU kernel/grid/thread-block signatures; ambiguous h/z signatures are pooled.',
      'limits':'Instrumented summed GPU durations, not full-forward latency. Parallel h/z includes fresh row inspection, scalar projections, gates and residual additions; the separate fused h/z entry includes its fallback and completion checks; native h/z here are GEMMs only. Conditional replacement timings are nonadditive.'})

if __name__=='__main__':main()
