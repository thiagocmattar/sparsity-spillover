"""Audit all fifteen processes and report conditional, joint and interaction effects."""
import math
from controls import MODES, contrasts, difference_spans, mask
from io_utils import RUN, read, write, record


def geomean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite latencies required')
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def main():
    cfg = read(RUN/'config.json')
    summary = read(RUN/'artifacts/scientific/summary-001.json')
    expected = {(mode,rep) for mode in MODES for rep in range(1,cfg['process_replicates']+1)}
    assert len(summary)==len(expected) and {(r['mode'],r['replicate']) for r in summary}==expected
    samples, processes, diagnostics = {m:[] for m in MODES}, {m:[] for m in MODES}, {}
    sources, uuids, indices = [], set(), None
    for item in summary:
        folder = RUN/'artifacts/attempts'/item['attempt']
        result,timing,quality,diag = [read(folder/name) for name in ['result.json','timing.json','quality.json','diagnostics.json']]
        assert result['status']=='complete' and result['qualified'] and result['operand_bitwise']
        assert not result['arguments']['smoke'] and result['validation_blocks']==338
        assert result['condition']=='c30' and result['candidate']==item['mode']
        assert result['implementation_coverage']['mechanism_mask']==mask(item['mode'])
        assert quality['blocks']==338 and quality['documents']==500 and quality['prediction_tokens']==691886
        assert quality['excluded_tail_tokens']==1444 and all(quality['pass'].values())
        assert len(quality['exact_graph_checks'])==338 and all(r['bitwise'] for r in quality['exact_graph_checks'])
        assert diag['status']=='complete' and diag['coverage']['blocks']==338 and diag['operand_bitwise']
        assert len(diag['exact_operand_checks'])==338*6
        assert all(all(all(s.values()) for s in r['sites'].values()) for r in diag['exact_operand_checks'])
        assert result['runtime']['gpu']==cfg['gpu']
        uuids.add(result['runtime']['device_uuid'])
        if indices is None: indices=timing['indices']
        assert timing['indices']==indices and len(set(indices))==64
        row={'replicate':item['replicate']}
        for label in ['candidate_graph','native_graph']:
            selected=[s for s in timing['samples'] if s['mode']==label]
            assert len(selected)==448
            assert {(s['input_index'],s['repeat']) for s in selected}=={(i,r) for i in range(64) for r in range(7)}
            assert all(s['output_shape']==[1,2048,50304] for s in selected)
            row[label]=geomean([s['host_ms'] for s in selected])
            row[label+'_cuda_ms']=geomean([s['cuda_ms'] for s in selected])
            if label=='candidate_graph':samples[item['mode']].extend(s['host_ms'] for s in selected)
        processes[item['mode']].append(row)
        if item['replicate']==1:
            assert diag['counts_collected']
            diagnostics[item['mode']]=diag
        for name in ['result.json','timing.json','quality.json','diagnostics.json']:
            sources.append(record(folder/name))
    assert len(uuids)==1 and 'unavailable' not in uuids
    counts=diagnostics['frozen']['bf16_scalar_opportunity_lower_bound']
    for mode,diag in diagnostics.items():
        other=diag['bf16_scalar_opportunity_lower_bound']
        assert (counts['zero_products'],counts['model_products'])==(other['zero_products'],other['model_products'])
        assert diag['active_features_per_row']==diagnostics['frozen']['active_features_per_row']
        for name,row in diag['mechanisms_by_layer'].items():
            if name.startswith('joint.'):continue
            assert row['mma_issued']+row['mma_omitted_whole_short_group']+row['mma_omitted_empty_tile']==row['mma_potential']
            if not mask(mode)['short']:assert row['scalar_products']==row['short_completed_groups']==0
            if not mask(mode)['tile']:assert row['mma_omitted_empty_tile']==0
    assert diagnostics['t11']['mechanisms_by_layer']==diagnostics['frozen']['mechanisms_by_layer']
    times={m:geomean(values) for m,values in samples.items()}
    ranges={m:[min(r['candidate_graph'] for r in rows),max(r['candidate_graph'] for r in rows)] for m,rows in processes.items()}
    overlap=max(ranges['t11'][0],ranges['frozen'][0])<=min(ranges['t11'][1],ranges['frozen'][1])
    report={'status':'qualified' if overlap else 'fidelity-review-required',
        'interpretation':'Conditional mechanism effects, not additive allocations or a decomposition of memory/compute time',
        'latencies_ms':times,'processes':processes,'process_ranges_ms':ranges,
        'effects':contrasts(times),'difference_spans_ms':difference_spans(ranges),
        'span_definition':'Process-extrema spans, not confidence intervals',
        't11_over_frozen_latency_ratio':times['t11']/times['frozen'],
        't11_frozen_process_ranges_overlap':overlap,'gpu_uuid':next(iter(uuids)),
        'fidelity_rule':'Non-overlap requires review; overlap does not prove equivalence.',
        'diagnostics':diagnostics,'sources':sources,'scientific_processes':len(summary),
        'raw_timing_records':sum(len(v) for v in samples.values())*2}
    write(RUN/'results/mechanism-latency.json',report)
    print(f'Reduced {len(summary)} processes; status={report["status"]}; t11/frozen={report["t11_over_frozen_latency_ratio"]:.6f}')


if __name__=='__main__':main()
