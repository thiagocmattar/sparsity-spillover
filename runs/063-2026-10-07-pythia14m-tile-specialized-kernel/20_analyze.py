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
    comparisons={name:[] for name in ['specialized_over_generic_sparse','specialized_over_generic_no_skip','generic_skip','specialized_skip']}
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
        modes={f+'_'+m for f in ['tile'] for m in cfg['modes']}
        if set(local)!=modes or any(set(v)!=expected for v in local.values()):
            raise ValueError('Timing coverage is not fully paired')
        ratios={}
        definitions={'specialized_over_generic_sparse':('tile_generic_sparse','tile_tile_sparse'),
                     'specialized_over_generic_no_skip':('tile_generic_no_skip','tile_tile_no_skip'),
                     'generic_skip':('tile_generic_no_skip','tile_generic_sparse'),
                     'specialized_skip':('tile_tile_no_skip','tile_tile_sparse')}
        for name,(a,b) in definitions.items():
            values=[local[a][key]/local[b][key] for key in sorted(expected)]
            comparisons[name].extend(values);ratios[name]=geomean(values)
        processes.append({'attempt':attempt.name,'ratios':ratios,'loss':result['loss'],
                          'peak_allocated_bytes':result['peak_allocated_bytes'],
                          'peak_reserved_bytes':result['peak_reserved_bytes'],'runtime':result['runtime']})
        for name in ['result.json','timing.json','quality.json','manifest.json','samples.jsonl']:
            sources.append(record(attempt/name))
        for family in ['generic_sparse','tile_sparse']:
            path=attempt/f'diagnostics-{family}.json'
            if path.exists():
                if family in diagnostics:
                    raise ValueError('Duplicate diagnostic')
                row=read(path)
                if row['status']!='complete' or row['coverage']!=dict(blocks=338,documents=500,input_tokens=692224,excluded_tail_tokens=1444):
                    raise ValueError('Incomplete diagnostic')
                sources.append(record(path));diagnostics[family]=record(path)
    inspections=[attempt/'inspection.json' for attempt in attempts if (attempt/'inspection.json').exists()]
    if len(inspections)!=1 or read(inspections[0])['status']!='complete':raise ValueError('Missing isolated support diagnostic')
    sources.append(record(inspections[0]))
    if len(processes)!=3 or replicates!={1,2,3} or len(uuid)!=1 or 'unavailable' in uuid or set(diagnostics)!={'generic_sparse','tile_sparse'}:
        raise ValueError('Missing unique process, same physical GPU or final diagnostic')
    return {'kappa':result['checkpoints']['tile']['dose'],'gpu_uuid':next(iter(uuid)),'qualified':True,
            'paired_samples_per_comparison':len(next(iter(comparisons.values()))),
            'speedup':{name:geomean(values) for name,values in comparisons.items()},
            'host_geomean_ms':{m:geomean(v) for m,v in groups.items()},
            'host_median_ms':{m:statistics.median(v) for m,v in groups.items()},
            'cuda_geomean_ms':{m:geomean(v) for m,v in cuda.items()},
            'processes':processes,'diagnostics':diagnostics,'inspection':record(inspections[0]),'source_files':sources}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--tag',default='001',)
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
    lines=['# Generic versus corner-specialized consumer','',output['scope'],'',
           '| kappa | Generic sparse ms | Specialized sparse ms | Speedup | Latency reduction | Process speedups | BF16 eager loss |',
           '|---:|---:|---:|---:|---:|---|---:|']
    for p in pairs:
        t=p['host_geomean_ms'];s=p['speedup']['specialized_over_generic_sparse']
        ratios=', '.join(f"{r['ratios']['specialized_over_generic_sparse']:.6f}" for r in p['processes'])
        loss=statistics.mean(r['loss']['tile']['native'] for r in p['processes'])
        lines.append(f"| {p['kappa']:g} | {t['tile_generic_sparse']:.6f} | {t['tile_tile_sparse']:.6f} | {s:.6f}x | {(1-1/s)*100:.3f}% | {ratios} | {loss:.6f} |")
    lines+=['','All five graphs qualified against their own eager checkpoint on all 338 validation blocks.',
            'Same weights, corner gate and input coverage. Three processes are runtime repetitions, not independent training seeds.',
            'Both no-skip controls and CUDA-event samples are retained in results.json.']
    (RUN/'results/tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Verified two thresholds and six full processes')

if __name__=='__main__':main()
