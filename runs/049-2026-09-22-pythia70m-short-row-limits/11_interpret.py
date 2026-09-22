"""Post-hoc mechanism summary of the predeclared comparison; no model selection."""
from io_utils import RUN, read, write, record
import re


def compiler_resources():
    path=RUN/'artifacts/operators.log'
    result={}; limit=None; component=None; stack=None
    for line in path.read_text().splitlines():
        if '[1/2]' in line and '/nvcc ' in line:
            match=re.search(r'-DSHORT_LIMIT=(\d+)',line)
            limit=int(match[1]) if match else None
        if 'Compiling entry function' in line:
            component='h_z_fallback' if '_Z5jointILb1ELb0EE' in line else ('h_z_prepass' if '_Z13parallel_rows' in line else None)
        match=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',line)
        if match:stack=dict(zip(('stack_bytes','spill_store_bytes','spill_load_bytes'),map(int,match.groups())))
        match=re.search(r'Used (\d+) registers',line)
        if match and limit is not None and component is not None:
            result.setdefault(str(limit),{})[component]={'registers':int(match[1]),**stack}
    assert set(result)=={'8','16','32','64'} and all(len(v)==2 for v in result.values())
    return {'source':record(path),'by_capacity':result}


def mechanism(condition, mode):
    attempts = [p for p in (RUN/'artifacts/attempts').glob(f'final-{condition}-r1-*/result.json') if read(p)['status']=='complete']
    assert len(attempts)==1
    folder=attempts[0].parent
    path=folder/f'diagnostics-{mode}.json'
    diagnostics=read(path)
    limit={'opt073':8,'native_hz':None,'limit16':16,'limit32':32,'limit64':64}[mode]
    sites={}
    for site,width in [('h',2048),('z',512)]:
        histograms=[v for k,v in diagnostics['active_features_per_row'].items() if k.startswith(site+'.')]
        assert len(histograms)==6 and all(len(v)==width+1 for v in histograms)
        histogram=[sum(v[i] for v in histograms) for i in range(width+1)]
        rows=sum(histogram); nonzeros=sum(i*n for i,n in enumerate(histogram))
        assert rows==338*2048*6
        sites[site]={'rows':rows,'elements':rows*width,'nonzero_elements':nonzeros,
                     'exact_zero_fraction':1-nonzeros/(rows*width),'mean_nonzeros_per_row':nonzeros/rows,
                     'row_fraction_at_most':{str(n):sum(histogram[:n+1])/rows for n in (8,16,32,64)}}
    groups={}
    for capacity in (8,16,32,64):
        rows=[v for k,v in diagnostics['joint_occupancy_by_layer_limit'].items() if k.endswith(f':limit_{capacity}')]
        assert len(rows)==6
        pooled={k:sum(v[k] for v in rows) for k in rows[0]}
        pooled['paired_short_row_fraction']=pooled['paired_short_rows']/pooled['rows']
        pooled['all_short_group_fraction']=pooled['all_paired_short_groups']/pooled['groups']
        groups[str(capacity)]=pooled
    work=diagnostics['hybrid_counts_by_layer']
    counters={key:sum(row[i] for row in work.values()) for i,key in enumerate(diagnostics['hybrid_fields'])} if work else None
    profiles=read(folder/'profile-summary.json')
    graph=next(p for p in profiles['profiles'] if p['mode']==mode and p['execution']=='graph')
    components={}
    for label,pattern in [('h_z_fallback','joint<'),('h_z_prepass','parallel_rows<')]:
        matched={k:v for k,v in graph['kernels'].items() if pattern in k}
        components[label]={'instrumented_ms_per_input':sum(v['duration_us'] for v in matched.values())/graph['inputs']/1000,
                           'kernels':matched}
    if counters is not None:assert components['h_z_fallback']['kernels']
    return {'mode':mode,'short_limit':limit,'sites':sites,'occupancy_by_limit':groups,
            'actual_h_z_work':counters,'profile_components':components,
            'sources':[record(path),record(folder/'profile-summary.json')]}


def main():
    summary=read(RUN/'results/summary.json')
    output={'summary':record(RUN/'results/summary.json'),'compiler_resources':compiler_resources(),'conditions':[],
            'cautions':['Profile durations are instrumented and separate from reported latency.',
                        'Native h/z instruction counts are unmeasured, not zero.',
                        'Counterfactual occupancy at another capacity is not an executed work counter.',
                        'Three processes repeat the same full validation corpus; they are timing replicates, not independent datasets.']}
    for condition in summary['conditions']:
        cid=condition['id']; failures={}
        for mode in condition['modes']:
            rows=[]
            for path in (RUN/'artifacts/attempts').glob(f'final-{cid}-*/result.json'):
                if read(path)['status']!='complete':continue
                q=read(path.with_name('quality.json'));key=mode+'_graph'
                rows.append({'attempt':path.parent.name,'failed_blocks':[g['input_index'] for g in q['gates'][key] if not g['pass']],
                             'pooled_loss_delta':q['loss_delta'][key]})
            failures[mode]=rows
        output['conditions'].append({'id':cid,'mechanisms':[mechanism(cid,mode) for mode in condition['modes'] if mode!='native'],
                                     'numerical_checks':failures})
    write(RUN/'results/mechanism.json',output)
    print({'status':'written','conditions':len(output['conditions']),'diagnostic_records':15})


if __name__=='__main__':main()
