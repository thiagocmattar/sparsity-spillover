"""Reconcile retained timing pairs, attribution and row-NNZ diagnostics."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('run029_reduction', RUN/'10_reduce.py')
reduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduction)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
gm = reduction.geometric
RENAMES = {'A1-H+L1':'A1-H-L1','A1-H+OL1':'A1-H-OL1',
           'A4+OL1@4':'A4-OL1','A7+OL1@7':'A7-OL1'}


def main():
    source = RUN/'results/matched-retrospective-001.json'
    raw = read(source)
    cfg = read(RUN/'config.json')
    checked = {}
    for record in raw['sources']:
        path = RUN/record['path']
        assert path.stat().st_size == record['bytes'], path
        assert sha(path) == record['sha256'], path
        checked[record['path']] = record['sha256']
    points = [p for p in raw['points'] if p['phase']=='final' and p['family']!='A4+OL1@h']
    assert len(points)==150
    raw_pair_count = 0
    for point in points:
        for rep in point['replicates']:
            folder = RUN/'artifacts/attempts'/rep['attempt']
            result = read(folder/'result.json')
            assert (result['candidate'],result['condition']) == (point['candidate'],point['condition'])
            if rep['status'] != 'complete':
                continue
            quality, timing = read(folder/'quality.json'), read(folder/'timing.json')
            assert quality['blocks']==338 and quality['prediction_tokens']==691886
            assert quality['excluded_tail_tokens']==1444
            for gates in quality['gates'].values():
                assert [r['input_index'] for r in gates] == list(range(338))
            assert all(quality['pass'].values()) == rep['qualified']
            ratios = reduction.ratios(timing['samples'],cfg)
            raw_pair_count += len(ratios)
            assert math.isclose(gm(ratios),rep['speedup'],abs_tol=1e-12)
            for mode, field in [('native_graph','native_ms'),('candidate_graph','candidate_ms')]:
                median = statistics.median(s['host_ms'] for s in timing['samples'] if s['mode']==mode)
                assert math.isclose(median,rep[field],abs_tol=1e-12)
        if 'speedup' in point:
            assert math.isclose(gm([r['speedup'] for r in point['replicates']]),point['speedup'],abs_tol=1e-12)
            for field in ('native_ms','candidate_ms'):
                assert math.isclose(gm([r[field] for r in point['replicates']]),point[field],abs_tol=1e-12)
    selected = [p for p in points if p['candidate']=='k050']
    assert len(selected)==30 and all(p['qualified'] for p in selected)
    lookup = {(p['candidate'],p['condition']):p for p in points}
    inputs = {p['id']:p for p in read(RUN/'provenance/inputs.json')['checkpoints']}
    rows, diagnostics = [], []
    candidates = ('k049','k050','k050-no-skip','k050-attention-dense','p0')
    for p in selected:
        condition = p['condition']
        core = {c:lookup[c,condition] for c in candidates if lookup[c,condition]['qualified']}
        for c in ('k050','k050-no-skip','k050-attention-dense'):
            assert c in core
        bf16_loss = p['replicates'][0]['loss']['native_graph']
        assert all(math.isclose(r['loss']['native_graph'],bf16_loss,abs_tol=1e-12)
                   for q in core.values() for r in q['replicates'])
        record = inputs[condition]
        weights = next(f for f in record['original_files'] if f['path'].endswith('model.safetensors'))
        rows.append({'condition':condition,'family':RENAMES.get(p['family'],p['family']),
                     'dose':record['dose'],'canonical_FP16_S_model':p['R_model'],
                     'canonical_FP16_loss':record['canonical_logical_products']['coverage']['loss'],
                     'BF16_validation_loss':bf16_loss,'weight_file_sha256':weights['sha256'],
                     'native_relative_speedups':{c:q['speedup'] for c,q in core.items()},
                     'candidate_latency_ms':{c:q['candidate_ms'] for c,q in core.items()},
                     'paired_native_latency_ms':{c:q['native_ms'] for c,q in core.items()},
                     'sparse_path_factor':p['speedup']/core['k050-no-skip']['speedup'],
                     'attention_dense_over_K050':core['k050-attention-dense']['speedup']/p['speedup'],
                     'attempts':{c:[r['attempt'] for r in q['replicates']] for c,q in core.items()}})
        diagnostic_rep = next(r for r in p['replicates'] if r['replicate']==1)
        folder = RUN/'artifacts/attempts'/diagnostic_rep['attempt']
        diag = read(folder/'diagnostics.json')
        assert diag['status']=='complete' and diag['coverage']['blocks']==338
        per_layer = {r['name']:r for r in diag['per_site_layer']}
        site_rows = {}
        for site in ('h','z'):
            layer_rows=[]
            for layer in range(6):
                key=f'{site}.layer_{layer}'
                histogram=diag['active_features_per_row'][key]
                n=sum(histogram)
                assert n==338*2048
                nonzeros=sum(i*count for i,count in enumerate(histogram))
                assert nonzeros==per_layer[key]['total']-per_layer[key]['exact_zero_count']
                layer_rows.append({'layer':layer,'rows':n,'all_zero_rows':histogram[0],
                                   'all_zero_row_fraction':histogram[0]/n,
                                   'mean_nonzeros_per_row':nonzeros/n})
            site_rows[site]={'all_zero_row_fraction':sum(r['all_zero_rows'] for r in layer_rows)/sum(r['rows'] for r in layer_rows),
                             'mean_nonzeros_per_row':sum(r['mean_nonzeros_per_row'] for r in layer_rows)/6,
                             'layers':layer_rows}
        lower=diag['bf16_scalar_opportunity_lower_bound']
        assert lower['model_products']==p['canonical_counts']['model_product_count']
        assert lower['zero_products']==sum(r['zero_product_count'] for r in lower['per_operation'].values())
        diagnostics.append({'condition':condition,'family':rows[-1]['family'],'dose':record['dose'],
                            'source':str((folder/'diagnostics.json').relative_to(ROOT)).replace('\\','/'),
                            'source_sha256':sha(folder/'diagnostics.json'),
                            'precision':diag['precision'],'h_z_rows':site_rows,
                            'BF16_scalar_lower_bound':lower,'attention_counts_by_layer':diag['attention_counts_by_layer'],
                            'attention_counter_fields':diag['attention_counter_fields']})
    summary={}
    for candidate in candidates:
        values=[p['speedup'] for p in points if p['candidate']==candidate and p['qualified']]
        summary[candidate]={'qualified':len(values),'total':30,'geomean_native_relative':gm(values),
                            'minimum':min(values),'maximum':max(values)}
    sensitivity={}
    subsets={'all':rows,'without_search_checkpoint':[r for r in rows if r['condition']!='c30']}
    for family in sorted({r['family'] for r in rows}):
        subsets['without_'+family]=[r for r in rows if r['family']!=family]
        same=[r for r in rows if r['family']==family]
        if len(same)>=3:subsets['within_'+family]=same
    for name,subset in subsets.items():
        sensitivity[name]={target:reduction.ols([r['canonical_FP16_S_model'] for r in subset],
                          [r['native_relative_speedups']['k050'] if target=='native_relative' else r['sparse_path_factor'] for r in subset])
                          for target in ('native_relative','sparse_path')}
    result={'sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                       (source,RUN/'config.json',RUN/'provenance/inputs.json',RUN/'10_reduce.py')},
            'verified_original_artifacts':checked,'recomputed_timing_pairs':raw_pair_count,
            'cohort':'c01-c30; five historical A4+OL1@h controls c31-c35 remain separate.',
            'precision':'FP16 canonical count/loss; BF16 full-validation timed checkpoints and untimed row diagnostics.',
            'latency_estimand':'Geometric mean across three processes of each process median over 448 host-time samples. Native latency is paired separately with each candidate.',
            'speedup_estimand':raw['metric'],
            'attribution_estimand':'Ratio of separately measured native-normalized candidate speedups. Not a direct within-process ablation or confidence interval.',
            'rows':rows,'candidate_summary':summary,'diagnostics':diagnostics,'association_sensitivity':sensitivity,
            'sparse_path_geomean':gm([r['sparse_path_factor'] for r in rows]),
            'sparse_path_helped':sum(r['sparse_path_factor']>1 for r in rows),
            'attention_dense_faster':sum(r['attention_dense_over_K050']>1 for r in rows),
            'original_search_progress':raw['progress'],
            'functional_limit':'Row histograms provide token-level nonzero-count distributions, not token identities, input-dependent coordinate selection, residual-branch output magnitudes or context sensitivity.',
            'BF16_count_limit':'Retained BF16 scalar opportunity is a lower bound excluding probability underflow and weight zeros; it is not an equivalent replacement for the canonical FP16 counter.'}
    (HERE/'runtime-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print('Verified artifacts:',len(checked),'recomputed timing pairs:',raw_pair_count)
    print('Sparse-path GM:',result['sparse_path_geomean'],'helped:',result['sparse_path_helped'],'attention-dense faster:',result['attention_dense_faster'])
    for name in ('all','without_search_checkpoint','without_A7-OL1'):
        print(name,{k:v['r_squared'] for k,v in sensitivity[name].items()})
    for r in diagnostics:
        if r['condition'] in ('c20','c30'):
            print(r['condition'],{s:round(d['all_zero_row_fraction']*100,4) for s,d in r['h_z_rows'].items()})


if __name__=='__main__':
    main()
