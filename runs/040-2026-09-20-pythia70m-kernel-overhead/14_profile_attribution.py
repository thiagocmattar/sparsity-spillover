"""Attribute retained native GEMM traces by recorded operand shapes, not guesses."""
from collections import defaultdict
from io_utils import RUN, read, record, write


def key(event):
    args=event['args']
    return event['name'],tuple(args['grid']),tuple(args['block'])


def native_mapping(events):
    cpu={e['args']['External id']:e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
    mapping=defaultdict(set)
    shapes={(512,1536):'a',(512,2048):'m',(2048,512):'h',(512,512):'z',(512,50304):'head'}
    for event in events:
        if event.get('cat')!='kernel':continue
        owner=cpu.get(event['args'].get('External id'),{})
        if owner.get('name') not in ('aten::addmm','aten::mm'):continue
        dims=owner['args']['Input Dims']
        matrix,weight=(dims[1],dims[2]) if owner['name']=='aten::addmm' else (dims[0],dims[1])
        assert matrix[0]==2048 and matrix[1]==weight[0]
        operation=shapes[tuple(weight)]
        mapping[key(event)].add(operation)
    assert set().union(*mapping.values())==set(shapes.values())
    assert all(len(value)==1 or value=={'h','z'} for value in mapping.values())
    return {signature:('h_z_gemms' if value <= {'h','z'} else next(iter(value)))
            for signature,value in mapping.items()}


def summarize(events,mapping,custom):
    rows=defaultdict(lambda:{'calls':0,'duration_us':0.})
    for e in events:
        if e.get('cat')!='kernel':continue
        name=e['name']
        group=mapping.get(key(e),'other')
        if custom:
            if 'joint<' in name:group='fused_h_z'
            elif 'run035_flash::kernel<' in name:group='attention'
            elif name.startswith('_ZN7cutlass'):group='a_m'
            elif 'norm_pair(' in name:group='paired_norm'
            elif 'rope_gate(' in name:group='rope_and_gates'
        elif 'pytorch_flash::flash_fwd_kernel<' in name:group='attention'
        rows[group]['calls']+=1;rows[group]['duration_us']+=e['dur']
    for row in rows.values():row['ms_per_input']=row['duration_us']/4/1000
    if not custom:
        assert all(rows[op]['calls']==24 for op in ('a','m'))
        assert rows['h_z_gemms']['calls']==48
        assert rows['head']['calls']==4
    else:
        assert rows['fused_h_z']['calls']==24 and rows['a_m']['calls']==48
    return dict(rows)


def main():
    rows=[];sources=[]
    for folder in sorted((RUN/'artifacts/attempts').glob('scientific-*-full-r*-001')):
        result=read(folder/'result.json')
        assert result['qualified'] and result['status']=='complete'
        native_eager=folder/'profile-native-eager.json'
        mapping=native_mapping(read(native_eager)['traceEvents'])
        row={'condition':result['condition'],'replicate':result['arguments']['replicate'],'groups':{}}
        sources.append(record(native_eager))
        for mode in ('native','candidate'):
            path=folder/f'profile-{mode}-graph.json'
            row['groups'][mode]=summarize(read(path)['traceEvents'],mapping,mode=='candidate')
            sources.append(record(path))
        rows.append(row)
    assert len(rows)==6
    write(RUN/'results/profile-attribution.json',{'rows':rows,'sources':sources,'script':record(__file__),
        'coverage':'Four declared timing inputs per profile; both checkpoints, three processes each.',
        'method':'Native CPU addmm/mm operand dimensions label exact kernel-name/grid/block signatures in corresponding graph traces.',
        'limits':'Instrumented GPU kernel durations, not reported full-model latency. Native h and z share a kernel/grid signature and are pooled as GEMMs. Fused h/z additionally includes gates and residual addition. Use paired replacement timings for conditional latency effects.'})


if __name__=='__main__':main()
