"""Verify complete paired evidence before accepting any speedup claim."""
import math
import statistics
from io_utils import RUN, read, write, record, verify


def geomean(values):
    if not values or any(v<=0 or not math.isfinite(v) for v in values):
        raise ValueError('Positive finite latency samples required')
    return math.exp(statistics.mean(math.log(v) for v in values))


def reduce_pair(attempts):
    cfg=read(RUN/'config.json');groups={};cuda={};processes=[];sources=[];diagnostics={};indices=None
    comparisons={name:[] for name in ['tile_over_scalar_native','tile_over_scalar_no_skip','tile_over_scalar_sparse',
        'scalar_sparse_over_native','tile_sparse_over_native','scalar_sparse_over_no_skip','tile_sparse_over_no_skip']}
    original={r['id']:r for r in read(RUN/'provenance/inputs.json')['checkpoints']}
    uuid=set();replicates=set()
    for attempt in attempts:
        result=read(attempt/'result.json');timing=read(attempt/'timing.json')
        if result['status']!='complete' or result['config']!=record(RUN/'config.json'):
            raise ValueError('Incomplete process or changed scientific config')
        if any(c!=original[c['id']] for c in result['checkpoints'].values()):
            raise ValueError('Source checkpoint changed')
        if any(row!=record(RUN/row['path']) for row in result['runtime']['source_files']):
            raise ValueError('Runtime scientific source changed')
        if attempt.name.endswith('-002'):
            audits=[]
            for path in (RUN/'artifacts/bootstrap-retry-002').glob('*.json'):
                audit=read(path)
                if attempt.name in audit['argv'] and audit['argv'][0].endswith('02_benchmark.py'):
                    audits.append((path,audit))
            if len(audits)!=1:
                raise ValueError('One module-selection audit per retry process required')
            path,audit=audits[0]
            if audit['status']!='verified' or 'blocks=338' not in audit['diagnostic_signature']:
                raise ValueError('Run061 diagnostic interface not verified')
            if {r['path'] for r in audit['source_files']}!={'kernel.py','diagnostics.py','infrastructure/retry-002/sitecustomize.py'}:
                raise ValueError('Unexpected retry modules')
            for row in audit['source_files']:
                verify(row)
            sources.append(record(path))
        uuid.add(result['runtime']['gpu_uuid']);replicates.add(result['arguments']['replicate'])
        if indices is not None and timing['indices']!=indices:
            raise ValueError('Timing inputs changed')
        indices=timing['indices']
        if len(indices)!=64 or len(set(indices))!=64 or any(not 0<=i<338 for i in indices):
            raise ValueError('Invalid fixed timing inputs')
        if not result['qualified'] or not all(all(q['pass'].values()) and q['blocks']==338 and q['prediction_tokens']==691886
                                              for q in result['quality'].values()):
            raise ValueError('Complete own-eager numerical qualification required')
        local={}
        for sample in timing['samples']:
            key=(sample['repeat'],sample['input_index'])
            target=local.setdefault(sample['mode'],{})
            if key in target:
                raise ValueError('Duplicate paired observation')
            target[key]=sample['host_ms'];groups.setdefault(sample['mode'],[]).append(sample['host_ms'])
            cuda.setdefault(sample['mode'],[]).append(sample['cuda_ms'])
        expected={(r,i) for r in range(7) for i in range(64)}
        modes={f+'_'+m for f in ['scalar','tile'] for m in cfg['modes']}
        if set(local)!=modes or any(set(v)!=expected for v in local.values()):
            raise ValueError('Timing coverage is not fully paired')
        ratios={}
        for name in comparisons:
            if name.startswith('tile_over_scalar_'):
                mode=name.removeprefix('tile_over_scalar_');a,b='scalar_'+mode,'tile_'+mode
            else:
                family,reference=name.split('_sparse_over_');a,b=family+'_'+reference,family+'_sparse'
            values=[local[a][key]/local[b][key] for key in sorted(expected)]
            comparisons[name].extend(values);ratios[name]=geomean(values)
        processes.append({'attempt':attempt.name,'ratios':ratios,'loss':result['loss'],
                          'peak_allocated_bytes':result['peak_allocated_bytes'],
                          'peak_reserved_bytes':result['peak_reserved_bytes'],'runtime':result['runtime']})
        for name in ['result.json','timing.json','quality.json','manifest.json','samples.jsonl']:
            sources.append(record(attempt/name))
        for family in ['scalar','tile']:
            path=attempt/f'diagnostics-{family}.json'
            if path.exists():
                if family in diagnostics:
                    raise ValueError('Duplicate diagnostic')
                row=read(path)
                if row['status']!='complete' or row['coverage']!=dict(blocks=338,documents=500,input_tokens=692224,excluded_tail_tokens=1444):
                    raise ValueError('Incomplete diagnostic')
                sources.append(record(path));diagnostics[family]=record(path)
    if len(processes)!=3 or replicates!={1,2,3} or len(uuid)!=1 or 'unavailable' in uuid or set(diagnostics)!={'scalar','tile'}:
        raise ValueError('Missing unique process, same physical GPU or final diagnostic')
    return {'kappa':result['checkpoints']['scalar']['dose'],'gpu_uuid':next(iter(uuid)),'qualified':True,
            'paired_samples_per_comparison':len(next(iter(comparisons.values()))),
            'speedup':{name:geomean(values) for name,values in comparisons.items()},
            'host_geomean_ms':{m:geomean(v) for m,v in groups.items()},
            'host_median_ms':{m:statistics.median(v) for m,v in groups.items()},
            'cuda_geomean_ms':{m:geomean(v) for m,v in cuda.items()},
            'processes':processes,'diagnostics':diagnostics,'source_files':sources}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='001',choices=['001','002'])
    args=parser.parse_args()
    for row in read(RUN/f'transfer/inventory-{args.tag}.json')['files']:
        verify(row)
    summary=read(RUN/f'artifacts/scientific/summary-{args.tag}.json')
    if len(summary)!=6:
        raise ValueError('Exactly six declared full processes required')
    pairs=[]
    for pair in ['0','0p1']:
        attempts=[RUN/'artifacts/attempts'/r['attempt'] for r in summary if r['attempt'].startswith(f'scientific-k{pair}-')]
        pairs.append(reduce_pair(attempts))
    if len({p['gpu_uuid'] for p in pairs})!=1:
        raise ValueError('Both thresholds must use one physical GPU')
    output={'status':'verified','pairs':pairs,'source_script':record(__file__),
            'scope':'RTX5090 BF16 B1/T2048 full50304 logits; uncached; equal staging/setup excluded; six fresh processes',
            'limits':'one seed/GPU; tile models have higher original loss; skipping gain is not measured DRAM bandwidth or decoding speed'}
    write(RUN/'results/results.json',output)
    lines=['# Scalar versus corner tile: full-model latency','',output['scope'],'',
           '| Gate | kappa | Native ms | No h/z skip ms | Sparse ms | Native/sparse | No-skip/sparse | Tokens/s | BF16 loss |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pair in pairs:
        for f in ['scalar','tile']:
            t=pair['host_geomean_ms'];s=pair['speedup']
            loss=statistics.mean(p['loss'][f]['native'] for p in pair['processes'])
            lines.append(f"| {f} | {pair['kappa']:g} | {t[f+'_native']:.6f} | {t[f+'_no_skip']:.6f} | {t[f+'_sparse']:.6f} | {s[f+'_sparse_over_native']:.4f}x | {s[f+'_sparse_over_no_skip']:.4f}x | {2048000/t[f+'_sparse']:.0f} | {loss:.6f} |")
    lines+=['','| kappa | Tile/scalar sparse speedup | Three process ratios |','|---:|---:|---|']
    for p in pairs:
        ratios=', '.join(f"{r['ratios']['tile_over_scalar_sparse']:.6f}" for r in p['processes'])
        lines.append(f"| {p['kappa']:g} | {p['speedup']['tile_over_scalar_sparse']:.6f}x | {ratios} |")
    lines+=['','Speedup >1 means the named implementation is faster. Ratios pool same-input/pass pairs, not selected medians.',
            'The no-skip control disables h/z tile bypass and short-row substitution together; all other K050 settings are matched.',
            output['limits']]
    (RUN/'results/tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
