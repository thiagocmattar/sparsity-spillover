"""Check recovered raw samples and source identities without overwriting Pod results."""
import math
import statistics
from controls import MODES, settings
from io_utils import RUN, read, write, verify, record


def main():
    report=read(RUN/'results/am-load-avoidance.json')
    transfer=read(RUN/'artifacts/verification.json')
    assert transfer['all_transferred_bytes_verified'] and transfer['complete_scientific_matrix']
    assert report['status']=='qualified'
    for source in report['sources']:verify(source)
    samples={m:[] for m in MODES};loss_errors=[];max_errors=[];uuids=set()
    identities=None;native_means=[];total=0
    for mode in MODES:
        for replicate in [1,2,3]:
            folder=RUN/'artifacts/attempts'/f'scientific-{mode}-r{replicate}-001'
            result,quality,timing=[read(folder/name) for name in ['result.json','quality.json','timing.json']]
            assert result['qualified'] and result['status']=='complete'
            assert result['implementation_coverage']['port_settings']==settings(mode)
            assert quality['blocks']==338 and quality['documents']==500 and quality['excluded_tail_tokens']==1444
            assert quality['prediction_tokens']==691886 and all(quality['pass'].values())
            uuids.add(result['runtime']['device_uuid'])
            if identities is None:identities=timing['indices']
            assert timing['indices']==identities and len(set(identities))==64
            for label in ['candidate_graph','native_graph']:
                values=[s for s in timing['samples'] if s['mode']==label]
                assert len(values)==448
                # Samples index positions in the retained 64-block selection;
                # timing.indices maps those positions to validation-block IDs.
                assert {(s['input_index'],s['repeat']) for s in values}=={(i,p) for i in range(64) for p in range(7)}
                assert all(s['output_shape']==[1,2048,50304] and math.isfinite(s['host_ms']) and s['host_ms']>0 for s in values)
                assert all(math.isfinite(s['cuda_ms']) and s['cuda_ms']>0 and s['staging_ms_excluded']>=0 for s in values)
                total+=len(values)
                if label=='candidate_graph':samples[mode].extend(s['host_ms'] for s in values)
                else:native_means.append(statistics.geometric_mean(s['host_ms'] for s in values))
                loss_errors.append(abs(quality['loss'][label]-quality['loss']['native']))
                checks=quality['gates'][label]
                assert len(checks)==338 and all(s['pass'] for s in checks)
                max_errors.extend(s['max_abs'] for s in checks)
    means={m:statistics.geometric_mean(v) for m,v in samples.items()}
    assert total==13440 and len(uuids)==1 and next(iter(uuids))==report['device_uuid']
    assert all(math.isclose(means[m],report['latencies_ms'][m],rel_tol=0,abs_tol=1e-12) for m in MODES)
    for mode,effect in report['effects'].items():
        ref,other=('port-am-dense','port-am') if mode=='port_sparse_effect' else ('frozen',mode)
        assert math.isclose(effect['saved_ms'],means[ref]-means[other],rel_tol=0,abs_tol=1e-12)
        assert math.isclose(effect['speedup'],means[ref]/means[other],rel_tol=0,abs_tol=1e-12)
    result={'status':'passed','report':record(RUN/'results/am-load-avoidance.json'),
            'source_files_verified':len(report['sources']),'processes':15,'raw_samples':total,
            'recomputed_latencies_ms':means,'maximum_logit_absolute_error':max(max_errors),
            'maximum_absolute_loss_difference':max(loss_errors),
            'native_graph_process_geomean_range_ms':[min(native_means),max(native_means)],
            'single_gpu_uuid':next(iter(uuids)),'reduction_absolute_tolerance_ms':1e-12}
    write(RUN/'artifacts/local-reduction-verification.json',result)
    print(result)


if __name__=='__main__':main()
