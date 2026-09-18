"""Join all 40 retained clipping points to verified fresh-process timings."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCES={}


def read(path):
    raw=path.read_bytes()
    SOURCES[path.relative_to(HERE).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)


def gm(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return math.exp(math.fsum(map(math.log,values))/len(values))


def pairs(samples):
    by={name:{} for name in ['native_graph','candidate_graph']}
    for sample in samples:
        key=sample['repeat'],sample['input_index']
        assert sample['mode'] in by and key not in by[sample['mode']]
        assert sample['output_shape']==[1,2048,50304]
        by[sample['mode']][key]=sample['host_ms']
    expected={(r,i) for r in range(7) for i in range(64)}
    assert all(set(v)==expected for v in by.values())
    n=[by['native_graph'][k] for k in sorted(expected)]
    c=[by['candidate_graph'][k] for k in sorted(expected)]
    return n,c,[a/b for a,b in zip(n,c)]


def main():
    cfg=read(HERE/'config.json');inputs=read(HERE/'provenance/inputs.json')
    verified=read(HERE/'artifacts/verification.json')
    assert verified['all_required_artifacts_verified']
    points=[];uuids=set();indices=set();kernel_sources=set()
    for condition in inputs['conditions']:
        cp=next(c for c in inputs['checkpoints'] if c['id']==condition['checkpoint_id'])
        native=[];candidate=[];ratios=[];reps=[]
        for replicate in range(1,4):
            folder=HERE/'artifacts/attempts'/f"scientific-{condition['id']}-r{replicate}-001"
            result,quality,timing=[read(folder/f) for f in ['result.json','quality.json','timing.json']]
            assert result['status']=='complete' and not result['arguments']['smoke']
            assert result['candidate']==condition['candidate'] and result['clipping_condition']==condition
            assert result['checkpoint']==cp and quality['blocks']==338 and quality['documents']==500
            assert quality['prediction_tokens']==691886 and quality['excluded_tail_tokens']==1444
            assert result['qualification']==quality['pass'] and result['qualified']==all(quality['pass'].values())
            assert quality['pass']['native_graph']
            assert all(math.isfinite(v) for v in quality['loss'].values())
            n,c,r=pairs(timing['samples'])
            assert math.isclose(gm(r),timing['summary']['candidate_graph']['paired_geomean_speedup'],rel_tol=1e-12)
            native+=n;candidate+=c;ratios+=r
            runtime=result['runtime'];assert runtime['gpu']==cfg['gpu']
            assert runtime['torch']=='2.11.0+cu128' and runtime['cuda']=='12.8'
            uuids.add(runtime['device_uuid']);indices.add(tuple(timing['indices']))
            kernel_sources.add(json.dumps(result['port_sources'],sort_keys=True))
            reps.append({'replicate':replicate,'qualified':result['qualified'],'native_gm_ms':gm(n),
                'candidate_gm_ms':gm(c),'speedup':gm(r),'loss':quality['loss'],'loss_delta':quality['loss_delta'],
                'peak_allocated_bytes':result['peak_allocated_bytes'],'elapsed_seconds':result['elapsed_seconds'],
                'attempt':folder.name})
        assert len(ratios)==1344
        old=condition['retained_point'];counts=old['counts']
        numerator=counts['block_zero_product_count'];denominator=counts['model_product_count']
        assert sum(c['zero_product_count'] for c in counts['per_operation'].values())==numerator
        assert math.isclose(numerator/denominator,old['R_model'],abs_tol=1e-15)
        diagnostic=read(HERE/'artifacts/attempts'/f"scientific-{condition['id']}-r1-001"/'diagnostics.json')
        assert diagnostic['status']=='complete' and diagnostic['coverage']['blocks']==338
        points.append({'condition':condition['id'],'scale':condition['scale'],'family':condition['family'],
            'p':condition['p'],'kernel':condition['candidate'],'checkpoint_key':cp['paper_checkpoint_key'],
            'checkpoint_content_sha256':old['checkpoint_content_sha256'],'original_checkpoint':cp['original_checkpoint'],
            'thresholds_by_site_layer':old['thresholds_by_site_layer'],'retained_fp16_loss':old['loss'],
            'retained_fp16_counts':counts,'sparsity_percent':100*numerator/denominator,
            'qualified':all(r['qualified'] for r in reps),'native_gm_ms':gm(native),'candidate_gm_ms':gm(candidate),
            'speedup':gm(ratios),'process_latency_min_ms':min(r['candidate_gm_ms'] for r in reps),
            'process_latency_max_ms':max(r['candidate_gm_ms'] for r in reps),
            'bf16_native_loss':math.fsum(r['loss']['native'] for r in reps)/3,
            'bf16_candidate_loss':math.fsum(r['loss']['candidate_graph'] for r in reps)/3,
            'bf16_scalar_opportunity_lower_bound':diagnostic['bf16_scalar_opportunity_lower_bound'],
            'replicates':reps})
    assert len(points)==40 and len(uuids)==len(indices)==len(kernel_sources)==1
    for point in points:
        baseline=next(p for p in points if p['scale']==point['scale'] and p['family']==point['family'] and p['p']==0)
        point['same_checkpoint_p0_speedup']=baseline['candidate_gm_ms']/point['candidate_gm_ms']
    data={'status':'complete_verified_reduction','protocol':cfg,'points':points,
        'device_uuid':next(iter(uuids)),'timing_indices':list(next(iter(indices))),
        'latency_definition':'geometric mean of 1344 synchronized host graph times per setting; recurring clipping is timed',
        'precision_note':'Retained FP16 clipping loss/counts and new BF16 native/candidate loss are separate estimands.',
        'kernel_note':'Byte-identical final K050 / qualified 70M port with external PyTorch clipping; p0 identity masks elided.',
        'sources':SOURCES}
    out=HERE/'results';out.mkdir(exist_ok=True)
    (out/'clipping-final-kernel.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    lines=['# Dense/ReLU post-hoc clipping: final-kernel latency','',
        'RTX5090, BF16, B1/T2048, full vocabulary; 1344 pairs/setting. FP16 loss and sparsity remain the retained sweep values. p0-relative speedup uses this same session and checkpoint. All clipping work is timed; process ranges are not confidence intervals.','',
        '| Size | Control | p | FP16 loss | S_model (%) | Native ms | Final + clipping ms | p0-relative | Qualified |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for p in points:
        lines.append(f"| {p['scale']} | {p['family']} | {p['p']:.1f} | {p['retained_fp16_loss']:.6f} | {p['sparsity_percent']:.6f} | {p['native_gm_ms']:.6f} | {p['candidate_gm_ms']:.6f} | {p['same_checkpoint_p0_speedup']:.4f}x | {p['qualified']} |")
    (HERE/'TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'points':40,'qualified':sum(p['qualified'] for p in points),'device_uuid':data['device_uuid']}))


if __name__=='__main__':main()
