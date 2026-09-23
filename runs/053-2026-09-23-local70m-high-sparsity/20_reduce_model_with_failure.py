"""Summarize qualified and failed full-model cells without promoting failures."""
import argparse
import math
from collections import defaultdict
from local_support import RUN, load, sha, write

def geometric(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(math.fsum(map(math.log,values))/len(values))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag',default='model-001')
    args = parser.parse_args(); cfg = load(RUN/'model-config.json')
    sources=[]; rows=[]; profiles={}; diagnostics={}
    for cid in cfg['conditions']:
        folder = RUN/'artifacts/attempts'/(args.tag+'-reference-'+cid)
        for name in ('result.json','quality.json','memory.json','timing.json','profile-summary.json'):
            path=folder/name
            if not path.exists():
                assert name=='profile-summary.json' and cid=='c25'
                continue
            sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
        result=load(folder/'result.json'); q=load(folder/'quality.json'); timing=load(folder/'timing.json')
        assert result['status'] in ('complete','failed')
        if result['status']=='failed': assert result['error']=='Numerical qualification failed; retained failed measurements'
        assert all(len(x)==338 for x in q['gates'].values())
        assert (q['split'],q['blocks'],q['documents'],q['excluded_tail_tokens']) == ('validation',338,500,1444)
        assert result['policy_sha256']==sha(RUN/'provenance/local-policy.json')
        assert len(timing['indices'])==len(set(timing['indices']))==64
        assert len(timing['samples'])==6*64*7
        modes=defaultdict(list)
        for sample in timing['samples']: modes[sample['mode']].append(sample)
        values={}
        for mode,samples in modes.items():
            assert len(samples)==448
            assert len({(r['repeat'],r['input_index']) for r in samples})==448
            assert {r['input_index'] for r in samples}==set(range(64))
            values[mode]={'host_ms':geometric([r['host_ms'] for r in samples]),
                          'cuda_ms':geometric([r['cuda_ms'] for r in samples])}
            assert math.isclose(values[mode]['host_ms'],result['timing_ms'][mode],rel_tol=1e-12)
        rows.append({'condition':cid,'kappa':result['checkpoint']['dose'],'loss':result['loss'],
            'qualification':q['pass'],'failed_inputs':{m:[v['input_index'] for v in rows if not v['pass']] for m,rows in q['gates'].items()},
            'loss_delta':q['loss_delta'],'latency':values,
            'peak_allocated_GiB':result['peak_allocated_bytes']/1024**3,
            'peak_reserved_GiB':result['peak_reserved_bytes']/1024**3,
            'minimum_free_after_capture_GiB':min(r['free_bytes'] for r in result['memory'])/1024**3,
            'max_logit_errors':{mode:max(x['max_abs'] for x in gates) for mode,gates in q['gates'].items()},
            'max_relative_l2':{mode:max(x['relative_l2'] for x in gates) for mode,gates in q['gates'].items()}})
        profiles[cid]=[]
        for profile in (load(folder/'profile-summary.json')['profiles'] if (folder/'profile-summary.json').exists() else []):
            if profile['execution']!='graph': continue
            kernels=sorted(profile['kernels'].items(),key=lambda pair:pair[1]['duration_us'],reverse=True)
            profiles[cid].append({'mode':profile['mode'],'inputs':profile['inputs'],
                'total_instrumented_kernel_us_per_input':sum(v['duration_us'] for _,v in kernels)/profile['inputs'],
                'top_kernels':[{'name':name,'calls':v['calls'],'us_per_input':v['duration_us']/profile['inputs']} for name,v in kernels[:10]]})
        if cid!='c00':
            if cid=='c25':
                folder=RUN/'artifacts/attempts/diagnosis-001-c25'
                assert load(folder/'result.json')['status']=='complete'
                for name in ('result.json','numerical-trace.json','profile-summary.json'):
                    path=folder/name;sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
            for name in ('candidate-activation-statistics.json','candidate-weight-statistics.json','candidate-work.json'):
                path=folder/name
            if not path.exists():
                assert name=='profile-summary.json' and cid=='c25'
                continue
            sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path)})
            a=load(folder/'candidate-activation-statistics.json'); w=load(folder/'candidate-work.json')['rows']
            assert a['coverage']=={'documents':500,'blocks':338,'input_tokens':692224,'excluded_tail_tokens':1444}
            assert len(a['per_site_layer'])==12 and len(w)==338*8
            assert len({(v['block'],v['site']) for v in w})==338*8
            for value in a['per_site_layer']:
                assert value['total']==338*2048*(2048 if value['name'].startswith('h.') else 512)
                assert value['nonfinite']==0 and value['finite']==value['total']
            for site,entries in a['structure'].items():
                for key,hist in entries.items():
                    group=int(key.removeprefix('union').removesuffix('_nnz_hist'))
                    assert sum(hist)==338*2048//group
            counts=defaultdict(lambda:[0,0])
            for value in w:
                assert 0<=value['executed_reduction_tiles']<=value['potential_reduction_tiles']
                counts[value['site']][0]+=value['executed_reduction_tiles']
                counts[value['site']][1]+=value['potential_reduction_tiles']
            diagnostics[cid]={'pooled_activations':a['pooled_by_site'],
                'work':{site:{'executed':x,'potential':y,'fraction':x/y} for site,(x,y) in counts.items()}}
    by={r['condition']:r for r in rows}; baseline=by['c00']['latency']['native_graph']['host_ms']
    for row in rows:
        t=row['latency'];candidate=t['candidate_graph']['host_ms']
        row['candidate_vs_native_Base_ratio']=candidate/baseline
        row['candidate_vs_same_checkpoint_dense_ratio']=candidate/t['dense_policy_graph']['host_ms']
        row['candidate_vs_same_checkpoint_prior_ratio']=candidate/t['prior_policy_graph']['host_ms']
        row['skip_saving_ms']=t['candidate_no_skip_graph']['host_ms']-candidate
    report={'scope':'One local process per checkpoint; full-validation numerical qualification, descriptive timings; the kappa .1 candidate fails numerical qualification.',
        'rows':rows,'diagnostics':diagnostics,'profiles':profiles,'sources':sources,
        'candidate_qualified_all':all(r['qualification']['candidate_graph'] for r in rows),
        'delta70_ms_unqualified_if_any_endpoint_fails':{mode:by['c24']['latency'][mode]['host_ms']-by['c25']['latency'][mode]['host_ms'] for mode in by['c24']['latency']},
        'limitations':['No process replication or matched local 14M delta.','Base and threshold conditions run in separate sequential processes.',
                       'Local laptop/WSL and RTX5090 timings are separate cohorts.','Profiles are instrumented; not benchmark latency.']}
    write(RUN/'results'/(args.tag+'-summary.json'),report)
    table=['# Local full-model integration','',report['scope'],'',
           '| Checkpoint | Native | opt073 | Best tested dense | Prior policy | New candidate | Skip off | Candidate / native Base | Candidate qualified |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        table.append('| '+row['condition']+' | '+' | '.join(f'{row["latency"][m+"_graph"]["host_ms"]:.4f}' for m in cfg['modes'])+f' | {row["candidate_vs_native_Base_ratio"]:.4f} | {row["qualification"]["candidate_graph"]} |')
    table+=['','Geometric-mean synchronized host milliseconds. c00=Base; c24/c25/c26=T2/Ph kappa .05/.1/.5.',
            'The new candidate fails at kappa .1 on input78. Its timing is retained but is not a qualified speedup. No statistical or cross-size claim follows.',
            'Source: `20_reduce_model_with_failure.py`; summary JSON contains source hashes, loss differences, diagnostics and profile summaries.']
    (RUN/'results'/(args.tag+'-table.md')).write_text('\n'.join(table)+'\n',encoding='utf-8')
    print('\n'.join(table))

if __name__=='__main__': main()
