"""Rebuild all measured effects from raw samples; retain descriptive process spans."""
import math
from controls import MODES, settings, contrasts
from io_utils import RUN, read, write, record


def geomean(values):
    if not values or any(not math.isfinite(v) or v<=0 for v in values):
        raise ValueError('Positive finite samples required')
    return math.exp(math.fsum(map(math.log,values))/len(values))


def main():
    summary=read(RUN/'artifacts/scientific/summary-001.json')
    assert len(summary)==15
    assert {(r['mode'],r['replicate']) for r in summary}=={(m,r) for m in MODES for r in [1,2,3]}
    samples={m:[] for m in MODES};processes={m:[] for m in MODES}
    sources=[];counters={};indices=None;devices=set();losses=set();errors=[]
    for item in summary:
        folder=RUN/'artifacts/attempts'/item['attempt']
        result,timing,quality=[read(folder/name) for name in ['result.json','timing.json','quality.json']]
        assert result['status']=='complete' and result['qualified'] and result['validation_blocks']==338
        assert result['implementation_coverage']['port_settings']==settings(item['mode'])
        assert all(quality['pass'].values()) and quality['excluded_tail_tokens']==1444
        assert quality['prediction_tokens']==691886
        assert not result['arguments']['smoke']
        devices.add(result['runtime']['device_uuid']);losses.update(quality['loss'].values())
        for gates in quality['gates'].values():
            assert len(gates)==338 and all(g['pass'] for g in gates)
            errors.extend(g['max_abs'] for g in gates)
        if indices is None:indices=timing['indices']
        assert timing['indices']==indices and len(set(indices))==64
        means={}
        for label in ['candidate_graph','native_graph']:
            rows=[r for r in timing['samples'] if r['mode']==label]
            assert len(rows)==448
            assert {(r['input_index'],r['repeat']) for r in rows}=={(i,p) for i in range(64) for p in range(7)}
            assert all(r['output_shape']==[1,2048,50304] for r in rows)
            means[label]=geomean([r['host_ms'] for r in rows])
            if label=='candidate_graph':samples[item['mode']].extend(r['host_ms'] for r in rows)
        processes[item['mode']].append({'replicate':item['replicate'],**means})
        for name in ['result.json','timing.json','quality.json']:sources.append(record(folder/name))
        if item['replicate']==1:
            diag=read(folder/'diagnostics.json')
            assert diag['coverage']=={'blocks':338,'documents':500,'input_tokens':692224,'excluded_tail_tokens':1444}
            counters[item['mode']]=diag;sources.append(record(folder/'diagnostics.json'))
    assert len(devices)==1
    latencies={m:geomean(v) for m,v in samples.items()}
    ranges={m:[min(p['candidate_graph'] for p in v),max(p['candidate_graph'] for p in v)] for m,v in processes.items()}
    effects=contrasts(latencies)
    for mode,row in effects.items():
        ref,other=('port-am-dense','port-am') if mode=='port_sparse_effect' else ('frozen',mode)
        row['saved_time_span_us']=[1000*(ranges[ref][0]-ranges[other][1]),1000*(ranges[ref][1]-ranges[other][0])]
    invariant=True
    for mode,diag in counters.items():
        a=diag['bf16_scalar_opportunity_lower_bound'];b=counters['frozen']['bf16_scalar_opportunity_lower_bound']
        invariant &= a['zero_products']==b['zero_products'] and a['model_products']==b['model_products']
    report={'status':'qualified','latencies_ms':latencies,'processes':processes,'process_ranges_ms':ranges,
            'effects':effects,'diagnostics':counters,'sources':sources,'raw_samples_verified':13440,
            'maximum_absolute_logit_error':max(errors),'loss_values':sorted(losses),
            'logical_counts_match_frozen':bool(invariant),'device_uuid':next(iter(devices)),
            'interpretation':'Single checkpoint; descriptive process-extrema spans, not confidence intervals. Port-dense has padded geometry, not A0.'}
    write(RUN/'results/am-load-avoidance.json',report)
    lines=['# a/m direct-port result','','14M T7/Pall, kappa=0.5. All timings measured in the same session.','',
           '| Mode | Full-model latency (ms) | Process range (ms) |','|---|---:|---|']
    for mode in MODES:lines.append(f'| {mode} | {latencies[mode]:.6f} | {ranges[mode][0]:.6f} to {ranges[mode][1]:.6f} |')
    lines+=['','| Comparison | Saved time (us) | Speedup | Difference span (us) |','|---|---:|---:|---|']
    for mode,row in effects.items():
        lo,hi=row['saved_time_span_us']
        lines.append(f"| {mode} | {1000*row['saved_ms']:+.3f} | {row['speedup']:.5f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines+=['',f'Maximum recorded absolute logit error: {max(errors)}.',f'Logical counts unchanged: {invariant}.',
            'Positive saved time means faster than frozen, except port_sparse_effect uses the port-dense control.','']
    (RUN/'results/verdict.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
