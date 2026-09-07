"""Strict matched reduction; failures retained, no endpoint or baseline selection."""
import math
from collections import Counter, defaultdict
from io_utils import RUN, read, write, record, verify


def geometric(values):
    if not values or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Positive finite observations required')
    return math.exp(sum(math.log(v) for v in values)/len(values))


def ratios(samples,cfg,field='host_ms'):
    modes={'native_graph':{},'candidate_graph':{}}
    expected={(p,i) for p in range(cfg['timing_passes']) for i in range(cfg['timing_inputs'])}
    for row in samples:
        if row['mode'] not in modes:raise ValueError('Unexpected timer mode')
        key=(row['repeat'],row['input_index'])
        if key in modes[row['mode']]:raise ValueError('Duplicate timing sample')
        if row['output_shape']!=[1,2048,50304]:raise ValueError('Not full-model full-logit timing')
        value=row[field]
        if value is None or not math.isfinite(value) or value<=0:raise ValueError('Invalid latency')
        modes[row['mode']][key]=value
    if any(set(v)!=expected for v in modes.values()):raise ValueError('Incomplete paired timing coverage')
    return [modes['native_graph'][k]/modes['candidate_graph'][k] for k in sorted(expected)]


def progress(points,catalog,anchor):
    selected=[p for p in points if p['phase']=='history' and p['condition']==anchor and p['qualified'] and p['candidate']!='p0']
    by_iteration=defaultdict(list)
    mapping={r['id']:r['paper_iteration'] for r in catalog if r['candidate']!='p0' and r['status']=='eligible'}
    for p in selected:by_iteration[mapping[p['candidate']]].append(p)
    best=1.;winner='dense';curve=[{'iteration':0,'speedup':1.,'incumbent':'dense'}]
    for i in range(1,max(mapping.values())+1):
        for p in by_iteration[i]:
            if p['speedup']>best:best=p['speedup'];winner=p['candidate']
        curve.append({'iteration':i,'speedup':best,'incumbent':winner})
    return curve


def ols(x,y):
    if len(x)!=len(y) or len(x)<3:raise ValueError('At least three paired points required')
    xm=sum(x)/len(x);ym=sum(y)/len(y)
    xx=sum((a-xm)**2 for a in x)
    if xx==0:raise ValueError('No predictor variation')
    slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/xx;intercept=ym-slope*xm
    fitted=[intercept+slope*a for a in x]
    ss=sum((b-ym)**2 for b in y);res=sum((b-c)**2 for b,c in zip(y,fitted))
    return {'intercept':intercept,'slope':slope,'r_squared':1-res/ss if ss else None,
            'n':len(x),'fitted':fitted,'residuals':[b-c for b,c in zip(y,fitted)],
            'weighting':'Unweighted checkpoint-level OLS; estimated intercept; R_model expressed as fraction.'}


def main():
    cfg=read(RUN/'config.json');inputs=read(RUN/'provenance/inputs.json')
    checkpoints={r['id']:r for r in inputs['checkpoints']}
    catalog=read(RUN/'provenance/candidates.json')['configurations']
    plan=read(RUN/'artifacts/scientific/plan.json')
    for row in plan['sources']:verify(row)
    completed=read(RUN/'artifacts/scientific/completed.json')
    if len(completed)!=len(plan['jobs']) or {r['job']['key'] for r in completed}!={r['key'] for r in plan['jobs']}:
        raise ValueError('Scientific matrix incomplete')
    groups=defaultdict(list);sources=[];uuids=set();diagnostics=[]
    for row in completed:
        path=verify(row['result']);sources.append(row['result']);result=read(path);job=row['job']
        if (result['candidate'],result['condition'])!=(job['candidate'],job['condition']):raise ValueError('Result identity mismatch')
        item={'replicate':job['replicate'],'status':result['status'],'qualified':False,
              'attempt':path.parent.name,'elapsed_seconds':result['elapsed_seconds'],'error':result.get('error')}
        if 'runtime' in result:uuids.add(result['runtime']['device_uuid'])
        if job['final'] and job['candidate']=='k050' and job['replicate']==1:
            diagnostic_path=path.parent/'diagnostics.json'
            if not diagnostic_path.exists():raise ValueError(f'Missing agreed K050 diagnostics: {job["condition"]}')
            diagnostic=read(diagnostic_path);sources.append(record(diagnostic_path))
            if diagnostic['status']!='complete' or diagnostic['coverage']['blocks']!=338:
                raise ValueError('Incomplete diagnostic coverage')
            lower=diagnostic['bf16_scalar_opportunity_lower_bound']
            canonical=checkpoints[job['condition']]['canonical_logical_products']['measured']
            if lower['model_products']!=canonical['model_product_count']:
                raise ValueError('Diagnostic model-product denominator differs')
            if lower['zero_products']!=sum(r['zero_product_count'] for r in lower['per_operation'].values()):
                raise ValueError('Unpooled diagnostic numerator')
            for operation,count in lower['per_operation'].items():
                if count['product_count']!=canonical['per_operation'][operation]['product_count']:
                    raise ValueError('Per-operation diagnostic denominator differs')
                if any(type(v)!=int or v<0 for v in count.values()):raise ValueError('Nonnegative integer counters required')
            diagnostics.append({'condition':job['condition'],'source':record(diagnostic_path),
                  'canonical_R_model':canonical['R_model'],'bf16_scalar_opportunity_lower_bound':lower,
                  'attention_counts_by_layer':diagnostic['attention_counts_by_layer'],
                  'hybrid_counts_by_layer':diagnostic['hybrid_counts_by_layer'],
                  'interpretation':'BF16 runtime scalar opportunities and MMA/SIMT execution counters are not canonical FP16 R_model or removed FLOPs.'})
        if result['status']=='complete':
            quality=read(path.parent/'quality.json');timing=read(path.parent/'timing.json')
            sources.extend([record(path.parent/'quality.json'),record(path.parent/'timing.json')])
            if quality['blocks']!=338 or quality['prediction_tokens']!=338*2047 or quality['excluded_tail_tokens']!=1444:
                raise ValueError('Qualification coverage mismatch')
            for gates in quality['gates'].values():
                if [r['input_index'] for r in gates]!=list(range(338)):raise ValueError('Incomplete numerical gates')
            qualified=all(quality['pass'].values())
            if qualified!=result['qualified']:raise ValueError('Qualification serialization differs')
            if not quality['pass']['native_graph']:raise ValueError('Shared denominator did not qualify')
            host=ratios(timing['samples'],cfg);cuda=ratios(timing['samples'],cfg,'cuda_ms')
            speedup=geometric(host)
            if abs(speedup-result['timing']['candidate_graph']['paired_geomean_speedup'])>1e-10:
                raise ValueError('Stored summary differs from raw pairs')
            item.update(qualified=qualified,speedup=speedup,cuda_speedup=geometric(cuda),loss=quality['loss'],
                        loss_delta=quality['loss_delta'],native_ms=result['timing']['native_graph']['median_host_ms'],
                        candidate_ms=result['timing']['candidate_graph']['median_host_ms'])
        groups[('final' if job['final'] else 'history',job['candidate'],job['condition'])].append(item)
    if len(uuids)!=1 or 'unavailable' in uuids:raise ValueError('Matrix must use one identified physical GPU')
    if {d['condition'] for d in diagnostics}!=set(checkpoints):raise ValueError('All35 diagnostic records required')
    points=[]
    for (phase,candidate,condition),replicates in sorted(groups.items()):
        if sorted(r['replicate'] for r in replicates)!=[1,2,3]:raise ValueError('Three independent process replicates required')
        completed_reps=[r for r in replicates if r['status']=='complete']
        row={'phase':phase,'candidate':candidate,'condition':condition,'replicates':replicates,
             'qualified':all(r['qualified'] for r in replicates),'statuses':dict(Counter(r['status'] for r in replicates)),
             'R_model':checkpoints[condition]['canonical_logical_products']['measured']['R_model'],
             'canonical_counts':checkpoints[condition]['canonical_logical_products']['measured'],
             'family':checkpoints[condition]['family']}
        if len(completed_reps)==3:
            row.update(speedup=geometric([r['speedup'] for r in replicates]),
                       cuda_speedup=geometric([r['cuda_speedup'] for r in replicates]),
                       process_min=min(r['speedup'] for r in replicates),process_max=max(r['speedup'] for r in replicates),
                       native_ms=geometric([r['native_ms'] for r in replicates]),candidate_ms=geometric([r['candidate_ms'] for r in replicates]))
        points.append(row)
    selected=[p for p in points if p['phase']=='final' and p['candidate']=='k050' and p['qualified']]
    fit=ols([p['R_model'] for p in selected],[p['speedup'] for p in selected]) if len(selected)>=3 else None
    value={'points':points,'progress':progress(points,catalog,cfg['progress_anchor']),'k050_regression':fit,'diagnostics':diagnostics,
           'physical_gpu_uuid':next(iter(uuids)),'qualification_summary':dict(Counter((p['phase']+'-'+str(p['qualified'])) for p in points)),
           'metric':'Geometric mean of native_graph/candidate_graph host-latency pairs, equally pooling all448 pairs in each of three processes.',
           'sources':sources,'plan':record(RUN/'artifacts/scientific/plan.json'),'script':record(__file__)}
    write(RUN/'results/matched-retrospective-001.json',value)
    print({'points':len(points),'summary':value['qualification_summary'],'fit':fit,'final_progress':value['progress'][-1]})


if __name__=='__main__':main()
