"""Reduce direct latency pairs; qualification gates all accepted conclusions."""
import hashlib
import math
from pathlib import Path
import statistics
from io_utils import RUN,read,write,record,verify


def geomean(values):
    if not values or any(value<=0 or not math.isfinite(value) for value in values):
        raise ValueError('Positive finite paired latencies required')
    return math.exp(statistics.mean(math.log(value) for value in values))


def reduce_pair(attempts):
    cfg=read(RUN/'config.json');by_mode={};cuda_by_mode={};ratios=[];native_ratios=[];processes=[];sources=[];diagnostics={};fixed_indices=None
    original_checkpoints={row['id']:row for row in read(RUN/'provenance/inputs.json')['checkpoints']}
    for attempt in attempts:
        result=read(attempt/'result.json');timing=read(attempt/'timing.json')
        if result['status']!='complete':raise RuntimeError('Incomplete benchmark process')
        if result['config']!=record(RUN/'config.json'):raise RuntimeError('Scientific config identity changed')
        if any(row!=original_checkpoints[row['id']] for row in result['checkpoints'].values()):
            raise RuntimeError('Original checkpoint identity changed')
        indices=timing['indices']
        if len(indices)!=64 or len(set(indices))!=64 or any(not 0<=i<338 for i in indices):
            raise RuntimeError('Invalid fixed validation timing inputs')
        if fixed_indices is not None and indices!=fixed_indices:raise RuntimeError('Timing inputs differ across processes')
        fixed_indices=indices
        if result['qualified']!=all(all(q['pass'].values()) for q in result['quality'].values()):
            raise RuntimeError('Numerical qualification serialization disagrees')
        if not all(q['blocks']==338 and q['prediction_tokens']==691886 for q in result['quality'].values()):
            raise RuntimeError('Incomplete validation')
        grouped={}
        for sample in timing['samples']:
            key=(sample['repeat'],sample['input_index'])
            target=grouped.setdefault(sample['mode'],{})
            if key in target:raise ValueError('Duplicate timing pair')
            target[key]=sample['host_ms'];by_mode.setdefault(sample['mode'],[]).append(sample['host_ms'])
            cuda_by_mode.setdefault(sample['mode'],[]).append(sample['cuda_ms'])
        expected={(repeat,i) for repeat in range(cfg['timing_passes']) for i in range(cfg['timing_inputs'])}
        if set(grouped)!={'ph_native','ph_candidate','p2_native','p2_candidate'} or any(set(rows)!=expected for rows in grouped.values()):
            raise ValueError('Unmatched timing coverage')
        local=[grouped['ph_candidate'][key]/grouped['p2_candidate'][key] for key in sorted(expected)]
        ratios.extend(local)
        native_ratios.extend(grouped['ph_native'][key]/grouped['p2_native'][key] for key in sorted(expected))
        processes.append({'attempt':attempt.name,'qualified':result['qualified'],
            'speedup_p2_over_ph':geomean(local),'loss':result['loss'],
            'native_eager_qualification':{p:q['pass']['native'] for p,q in result['quality'].items()},
            'qualification':{p:q['pass'] for p,q in result['quality'].items()},
            'maximum_logit_errors':{p:{mode:{'max_absolute':max((row['max_abs'] for row in rows if row['max_abs'] is not None),default=None),
                'max_relative_l2':max((row['relative_l2'] for row in rows if row['relative_l2'] is not None),default=None)} for mode,rows in q['gates'].items()} for p,q in result['quality'].items()},
            'peak_allocated_bytes':result['peak_allocated_bytes'],'runtime':result['runtime'],
            'elapsed_seconds':result['elapsed_seconds']})
        for name in ['result.json','timing.json','quality.json','manifest.json']:sources.append(record(attempt/name))
        for pressure in ['ph','p2']:
            path=attempt/f'diagnostics-{pressure}.json'
            if path.exists():
                if pressure in diagnostics:raise RuntimeError('Duplicate final diagnostic')
                row=read(path);sources.append(record(path))
                if row['status']!='complete' or row['coverage']!=dict(blocks=338,documents=500,input_tokens=692224,excluded_tail_tokens=1444):
                    raise RuntimeError('Incomplete kernel diagnostic coverage')
                diagnostics[pressure]={'source':record(path),'coverage':row['coverage'],
                    'per_site_layer':row['per_site_layer'],'pooled_by_site':row['pooled_by_site'],
                    'hybrid_counts_by_layer':row['hybrid_counts_by_layer'],
                    'hybrid_counter_fields':row['hybrid_counter_fields'],'hybrid_limits':row['hybrid_limits'],
                    'bf16_scalar_opportunity_lower_bound':row['bf16_scalar_opportunity_lower_bound']}
    if len(processes)!=cfg['process_replicates'] or set(diagnostics)!={'ph','p2'}:raise RuntimeError('Missing process or final diagnostic')
    if {read(attempt/'result.json')['arguments']['replicate'] for attempt in attempts}!={1,2,3}:
        raise RuntimeError('Missing unique fresh-process replicates')
    speedup=geomean(ratios);qualified=all(row['qualified'] for row in processes)
    medians={mode:statistics.median(values) for mode,values in by_mode.items()}
    return {'kappa':float(read(attempts[0]/'result.json')['checkpoints']['ph']['dose']),
        'qualified':qualified,'accepted_speedup_p2_over_ph':speedup if qualified else None,
        'unqualified_or_qualified_timing_ratio':speedup,
        'native_graph_speedup_p2_over_ph':geomean(native_ratios),
        'latency_reduction_percent':100*(1-1/speedup) if qualified else None,
        'median_host_ms':medians,'geomean_host_ms':{mode:geomean(values) for mode,values in by_mode.items()},
        'median_cuda_ms':{mode:statistics.median(values) for mode,values in cuda_by_mode.items()},
        'geomean_cuda_ms':{mode:geomean(values) for mode,values in cuda_by_mode.items()},
        'input_tokens_per_second_from_median':{mode:2048/value*1000 for mode,value in medians.items()},
        'process_speedup_range':[min(row['speedup_p2_over_ph'] for row in processes),max(row['speedup_p2_over_ph'] for row in processes)],
        'paired_samples':len(ratios),'processes':processes,'diagnostics':diagnostics,'source_files':sources}


def main():
    inventory=read(RUN/'transfer/inventory-001.json')
    for row in inventory['files']:verify(row)
    summaries=read(RUN/'artifacts/scientific/cohort-summary.json')
    if len(summaries)!=6:raise RuntimeError('Incomplete six-process comparison')
    pairs=[]
    for pair in ['0','0p1']:
        attempts=[RUN/'artifacts/attempts'/row['attempt'] for row in summaries if row['attempt'].startswith(f'scientific-k{pair}-')]
        pairs.append(reduce_pair(attempts))
    identities={process['runtime']['gpu_uuid'] for pair in pairs for process in pair['processes']}
    if len(identities)!=1 or 'unavailable' in identities:raise RuntimeError('Physical GPU identity is not matched')
    output={'status':'complete','coverage':{'documents':500,'blocks':338,'input_tokens':692224,
        'prediction_tokens':691886,'excluded_tail_tokens':1444,'training_seed':1234,
        'processes_per_pair':3,'timing_inputs':64,'timing_passes':7},'pairs':pairs,
        'source_script':record(Path(__file__)),'transfer_receipt':record(RUN/'transfer/receipt-001.json'),
        'estimator':'geometric mean of direct same-input/pass Ph/P2 synchronized-host latency ratios, pooled over three fresh processes',
        'scope':'RTX5090 BF16 B1 T2048 uncached complete50304 logits; K050 frozen; equal input staging and setup excluded'}
    write(RUN/'results/results.json',output)
    lines=['# P2 versus Ph: measured inference latency','','RTX5090, BF16, B1/T2048, full logits; K050. Lower milliseconds is faster.',
        '','| kappa | Ph geomean ms | P2 geomean ms | Paired P2/Ph speedup | P2 latency reduction | Process range | Qualified |',
        '|---:|---:|---:|---:|---:|---:|---|']
    for row in pairs:
        med=row['geomean_host_ms'];speed=row['accepted_speedup_p2_over_ph'];delta=row['latency_reduction_percent']
        lo,hi=row['process_speedup_range']
        speed_text=f'{speed:.6f}x' if speed is not None else 'unqualified'
        delta_text=f'{delta:+.3f}%' if delta is not None else 'unqualified'
        lines.append(f"| {row['kappa']:g} | {med['ph_candidate']:.6f} | {med['p2_candidate']:.6f} | {speed_text} | {delta_text} | {lo:.6f}--{hi:.6f} | {row['qualified']} |")
    lines+=['','Speedup is the geometric mean of paired latency ratios, not the ratio of selected medians.',
        'Each threshold has1344 direct pairs in three fresh processes, with complete338-block numerical qualification.',
        'Process repetitions are systems repetitions, not new training seeds. A failed numerical gate prevents an accepted kernel speedup claim.']
    lines+=['','| kappa | Execution | Host median ms | CUDA median ms | Input tokens/s from host median |',
        '|---:|---|---:|---:|---:|']
    for row in pairs:
        for mode in ['ph_native','p2_native','ph_candidate','p2_candidate']:
            lines.append(f"| {row['kappa']:g} | {mode} | {row['median_host_ms'][mode]:.6f} | {row['median_cuda_ms'][mode]:.6f} | {row['input_tokens_per_second_from_median'][mode]:.1f} |")
    lines+=['','Native denotes the unmodified graph scaffold, qualified against its own eager checkpoint.',
        'Throughput is the inverse batch-1 latency for this uncached full-sequence workload, not autoregressive decoding throughput.']
    (RUN/'results/tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
