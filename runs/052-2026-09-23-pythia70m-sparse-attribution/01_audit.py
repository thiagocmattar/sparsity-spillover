"""Rank retained bottlenecks without running a model or selecting on validation."""
from support import RUN, read, write, sha


def main():
    repo=RUN.parents[1]
    previous=RUN.parent/'050-2026-09-22-pythia70m-kernel-families'
    grid=RUN.parent/'051-2026-09-22-pythia70m-frozen-kernel-grid'
    paths=[previous/'results/mechanism-summary.json', previous/'artifacts/screen-refine-001/summary.json',
           grid/'results/complete-table.json', repo/'analyses/034-2026-09-21-70m-latency-quality-frontier/data/combined-figure.json']
    mechanism,screen,table,old14=map(read,paths)
    rows={r['id']:r for r in table['rows']}
    r14={r['kappa']:r for r in old14['cohorts']['14M'] if r.get('family')=='HZ-OL1-h'}
    result={'interpretation':'Historical diagnosis; separate sessions, not a new speedup claim or validation-based dispatch selection.',
            'sources':[{'path':p.relative_to(repo).as_posix(),'sha256':sha(p)} for p in paths],
            'historical_delta_us':{'14M':1000*(r14[.05]['latency_ms']-r14[.1]['latency_ms']),
                                  '70M':1000*(rows['c24']['latency_ms']['sparse_c_graph']-rows['c25']['latency_ms']['sparse_c_graph'])},
            'per_site_layer':{}, 'profiles':{}}
    for site in ('h','z'):
        for layer in range(6):
            key=f'{site}.{layer}';values={}
            for cid in ('c24','c25'):
                d=mechanism['results'][cid]['sparse_c']['layers'][key]
                trained=screen[f'{cid}:confirmation:{key}']
                dense=min((n for n in trained if n.startswith('dense_') and trained[n]['qualified']),key=lambda n:trained[n]['host_ms'])
                gathered=min((n for n in trained if n.startswith('c_') and trained[n]['qualified']),key=lambda n:trained[n]['host_ms'])
                values[cid]={'mean_nnz':d['mean_nnz_per_row'],'rows_at_most8':d['fraction_rows_nnz_at_most']['8'],
                             'dense_control':dense,'best_prior_gathered':gathered,
                             'prior_gathered_over_dense':trained[gathered]['host_ms']/trained[dense]['host_ms']}
            result['per_site_layer'][key]=values
    for cid in ('c24','c25'):
        p=grid/f'artifacts/final-001-diagnostics-{cid}/profile-summary-sparse_c.json'
        trace=next(t for t in read(p)['profiles'] if t['execution']=='graph')
        result['profiles'][cid]={k:v['duration_us']/trace['inputs'] for k,v in trace['kernels'].items()
                                 if k in ('union_index','gathered','dense_gate','gate','combine')}
        result['sources'].append({'path':p.relative_to(repo).as_posix(),'sha256':sha(p)})
    result['priorities']=[
        'z: prior global-pack gathered paths approach dense at .1 but lose at .05; test no-global-pack execution and retain dense fallback.',
        'h layers1--5: test whether CTA-local metadata removes the recurring global scan without losing too much cross-output reuse.',
        'h layer0: include in the same component screen, but preserve dense unless both training splits and kappas improve.',
        'Producer fusion or a different family only after these measurements show conversion or consumer cost is the limiting factor.']
    write(RUN/'results/prior-audit.json',result)
    print({'historical_delta_us':result['historical_delta_us'],'profile_us_per_forward':result['profiles']})


if __name__=='__main__':main()
