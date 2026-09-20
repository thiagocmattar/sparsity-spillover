"""Verify this run's complete qualification, timing coverage and gain attribution."""
from collections import defaultdict
from statistics import median
import math
from io_utils import RUN, read, record, verify, write


def main():
    cfg=read(RUN/'config.json')
    selection=read(RUN/'provenance/final-selection.json')
    selected=selection['candidate']
    for row in read(verify(selection['manifest']))['files']:verify(row)
    for evidence in selection['development_evidence']:
        result=read(verify(evidence['result']))
        assert result['evaluation_split']=='training-development'
        assert result['arguments']['development'] and result['validation_blocks']==16
        verify(evidence['timing'])
    groups=defaultdict(list)
    for path in sorted((RUN/'artifacts/attempts').glob('*/result.json')):
        row=read(path)
        if row['arguments'].get('development') or row['arguments'].get('smoke'):continue
        groups[(row['condition'],row['candidate'])].append((path,row))
    expected=[(c,'full') for c in cfg['conditions']]
    expected += [(c,selected) for c in ('c00','c21','c16')]
    modes=('selected-no-skip','selected-native-hz','selected-native-attention',
           'selected-native-am','selected-native-norm','selected-native-rope','selected-native-head')
    expected += [(c,k) for c in ('c00','c21') for k in modes]
    expected += [('m14-c20',k) for k in ('hz-skips-off','native-hz')]
    expected += [('m14-c01','native-hz')]
    summaries={};sources=[];uuids=set();timing_indices=set();diagnostics=[]
    for key in expected:
        items=groups[key]
        assert len(items)==3 and {r['arguments']['replicate'] for _,r in items}=={1,2,3},key
        losses=[];latencies=[];native=[];errors=[]
        for path,row in items:
            assert row['status']=='complete' and row['qualified'],path
            assert row['evaluation_split']=='validation' and row['validation_blocks']==338
            assert row['config']==record(RUN/'config.json')
            uuids.add(row['runtime']['device_uuid'])
            assert row['runtime']['gpu']==cfg['gpu']
            timing=read(path.with_name('timing.json'))
            assert len(timing['indices'])==len(set(timing['indices']))==64
            timing_indices.add(tuple(timing['indices']))
            samples=defaultdict(list)
            for sample in timing['samples']:
                assert sample['output_shape']==[1,2048,50304]
                assert all(math.isfinite(sample[k]) and sample[k]>0 for k in ('host_ms','cuda_ms'))
                samples[sample['mode']].append(sample)
            assert {'candidate_graph','native_graph'} <= samples.keys()
            for mode,values in samples.items():
                assert len(values)==448
                assert {(s['repeat'],s['input_index']) for s in values}=={(p,i) for p in range(7) for i in range(64)}
                assert median(s['host_ms'] for s in values)==row['timing'][mode]['median_host_ms']
            quality=read(path.with_name('quality.json'))
            assert quality['blocks']==338 and quality['documents']==500
            assert quality['prediction_tokens']==338*2047 and quality['input_tokens']==338*2048
            assert quality['excluded_tail_tokens']==1444 and all(quality['pass'].values())
            for gates in quality['gates'].values():
                assert len(gates)==338 and {g['input_index'] for g in gates}==set(range(338))
                assert all(g['pass'] and g['finite'] and g['elementwise_gate'] and g['relative_l2']<=.02 for g in gates)
            assert abs(quality['loss_delta']['candidate_graph'])<=.001
            if key[1]==selected:
                assert row['optimization_candidate']==read(verify(selection['manifest']))
                assert row['arguments']['final']
                if row['arguments']['replicate']==1:
                    diagnostic_path=path.with_name('diagnostics.json');d=read(diagnostic_path)
                    assert d['status']=='complete' and d['coverage']['blocks']==338
                    assert d['coverage']['documents']==500 and d['coverage']['excluded_tail_tokens']==1444
                    assert len(d['weight_statistics'])>0
                    totals=[sum(v[i] for v in d['hybrid_counts_by_layer'].values()) for i in range(6)]
                    diagnostics.append({'condition':key[0],'source':record(diagnostic_path),
                        'fields':d['hybrid_fields'],'pooled_actual_work':totals,
                        'layouts':d['actual_output_projection_layouts'],
                        'native_instruction_counts':d['native_instruction_counts']})
            latencies.append(row['timing']['candidate_graph']['median_host_ms'])
            native.append(row['timing']['native_graph']['median_host_ms'])
            losses.append(quality['loss']['candidate_graph'])
            errors.append(abs(quality['loss_delta']['candidate_graph']))
            sources.extend(record(p) for p in (path,path.with_name('quality.json'),path.with_name('timing.json')))
        summaries[key]={'condition':key[0],'candidate':key[1],
            'candidate_ms':median(latencies),'candidate_range_ms':[min(latencies),max(latencies)],
            'native_ms':median(native),'native_range_ms':[min(native),max(native)],
            'loss':median(losses),'maximum_loss_delta':max(errors)}
    assert len(uuids)==1 and 'unavailable' not in uuids and len(timing_indices)==1
    conditional=[]
    for condition in ('c00','c21'):
        reference=summaries[(condition,selected)]['candidate_ms']
        for mode in modes:
            alternative=summaries[(condition,mode)]['candidate_ms']
            conditional.append({'condition':condition,'replacement':mode,'selected_ms':reference,
                'replacement_ms':alternative,'replacement_minus_selected_ms':alternative-reference})
    b70=summaries[('c00',selected)];s70=summaries[('c21',selected)]
    b14=summaries[('m14-c01','full')]
    controls=[summaries[(c,'full')] for c in ('m14-c20','m14-c35')]
    speed70=b70['native_ms']/s70['candidate_ms']
    speed14=max(b14['native_ms']/c['candidate_ms'] for c in controls)
    conservative70=b70['native_range_ms'][0]/s70['candidate_range_ms'][1]
    optimistic14=max(b14['native_range_ms'][1]/c['candidate_range_ms'][0] for c in controls)
    write(RUN/'results/final-verification.json',{
        'status':'complete','selected':selected,'groups_verified':len(expected),'processes_verified':3*len(expected),
        'device_uuids':sorted(uuids),'timing_inputs':64,'timing_passes':7,'fresh_processes_per_group':3,
        'validation_blocks_per_process':338,'validation_documents':500,'excluded_tail_tokens':1444,
        'selected_70m_native_base_speedup':speed70,'best_fresh14_native_base_speedup':speed14,
        'target_met':speed70>speed14 and speed70>1.42650079,
        'conservative70_exceeds_both14_process_ranges':conservative70>optimistic14,
        'comparison_note':'Process extrema are descriptive ranges, not confidence intervals. Different trained checkpoints have different quality.',
        'groups':list(summaries.values()),'conditional_replacements':conditional,'diagnostics':diagnostics,
        'attribution_limit':'Each replacement is conditional on all other selected components. Effects are nonadditive; disabling skips does not turn the custom fallback into a native baseline.',
        'sources':sources,'selection':record(RUN/'provenance/final-selection.json'),'script':record(__file__)})
    print({'selected':selected,'target_met':speed70>speed14 and speed70>1.42650079,
           'speedup70':speed70,'speedup14':speed14,'verified_processes':3*len(expected)},flush=True)


if __name__=='__main__':
    main()
