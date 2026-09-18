"""Reduce complete, verified matched 70M timings without imputing failures."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCES={}

def read(path):
    raw=path.read_bytes();SOURCES[path.relative_to(HERE).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)

def gm(values):
    assert values and all(math.isfinite(x) and x>0 for x in values)
    return math.exp(math.fsum(map(math.log,values))/len(values))

def pairs(samples):
    by={name:{} for name in ['native_graph','candidate_graph']}
    for s in samples:
        k=s['repeat'],s['input_index'];assert s['mode'] in by and k not in by[s['mode']]
        assert s['output_shape']==[1,2048,50304]
        by[s['mode']][k]=s['host_ms']
    expected={(r,i) for r in range(7) for i in range(64)}
    assert all(set(v)==expected for v in by.values())
    n=[by['native_graph'][k] for k in sorted(expected)]
    c=[by['candidate_graph'][k] for k in sorted(expected)]
    return n,c,[a/b for a,b in zip(n,c)]

def main():
    cfg=read(HERE/'config.json');manifest=read(HERE/'provenance/inputs.json')
    verification=read(HERE/'artifacts/verification.json')
    assert verification['all_required_artifacts_verified']
    points=[];uuids=set();indices=set();identities=set()
    for checkpoint in manifest['checkpoints']:
        native=[];candidate=[];ratios=[];reps=[]
        for replicate in [1,2,3]:
            folder=HERE/'artifacts/attempts'/f"scientific-{checkpoint['id']}-r{replicate}-001"
            result,quality,timing=[read(folder/f) for f in ['result.json','quality.json','timing.json']]
            assert result['status']=='complete' and not result['arguments']['smoke']
            assert result['candidate']==cfg['final_candidates'][0] and result['condition']==checkpoint['id']
            assert quality['blocks']==338 and quality['documents']==500
            assert quality['prediction_tokens']==691886 and quality['excluded_tail_tokens']==1444
            assert result['qualification']==quality['pass'] and result['qualified']==all(quality['pass'].values())
            assert quality['pass']['native_graph'], 'Native graph must qualify independently'
            assert all(math.isfinite(x) for x in quality['loss'].values())
            n,c,r=pairs(timing['samples']);assert math.isclose(gm(r),timing['summary']['candidate_graph']['paired_geomean_speedup'],rel_tol=1e-12)
            native+=n;candidate+=c;ratios+=r
            runtime=result['runtime'];assert runtime['gpu']==cfg['gpu']
            assert runtime['cuda']==cfg['runtime']['cuda'] and runtime['torch']=='2.11.0+cu128'
            assert runtime['transformers']==cfg['runtime']['transformers'] and runtime['numpy']==cfg['runtime']['numpy']
            uuids.add(runtime['device_uuid']);indices.add(tuple(timing['indices']))
            identities.add(json.dumps(result['port_sources'],sort_keys=True))
            reps.append({'replicate':replicate,'qualified':result['qualified'],'native_gm_ms':gm(n),
                'candidate_gm_ms':gm(c),'speedup':gm(r),'loss':quality['loss'],'loss_delta':quality['loss_delta'],
                'elapsed_seconds':result['elapsed_seconds'],'peak_allocated_bytes':result['peak_allocated_bytes'],
                'runtime':runtime,'attempt':folder.name})
        assert len(ratios)==1344
        logical=checkpoint['canonical_logical_products'];counts=logical['measured']
        numerator=counts['block_zero_product_count'];denominator=counts['model_product_count']
        assert sum(x['zero_product_count'] for x in counts['per_operation'].values())==numerator
        assert counts['block_product_count']+counts['lm_head_product_count']==denominator
        assert math.isclose(numerator/denominator,counts['R_model'],abs_tol=1e-15)
        points.append({'condition':checkpoint['id'],'family':checkpoint['family'],'kappa':checkpoint['dose'],
            'sparsity_percent':100*numerator/denominator,'canonical_counts':counts,
            'qualified':all(r['qualified'] for r in reps),'native_gm_ms':gm(native),'candidate_gm_ms':gm(candidate),
            'speedup':gm(ratios),'process_speedup_min':min(r['speedup'] for r in reps),
            'process_speedup_max':max(r['speedup'] for r in reps),'replicates':reps,
            'checkpoint_files':checkpoint['original_files'],'source_condition':checkpoint['source_condition']})
    assert len(points)==22 and len(uuids)==len(indices)==len(identities)==1
    data={'status':'complete_verified_reduction','candidate':cfg['final_candidates'][0],
        'protocol':cfg,'device_uuid':next(iter(uuids)),'timing_indices':list(next(iter(indices))),
        'port_sources':json.loads(next(iter(identities))),
        'latency_definition':'geometric mean of all1344 raw synchronized host full-logit graph times',
        'speedup_definition':'geometric mean of all1344 paired native/candidate host-time ratios',
        'comparison_limits':'One 70M session; shape-ported K050, not unchanged14M K050 or equal optimization search effort. Original14M runs used other physical RTX5090 GPUs. Logical sparsity is not runtime gain.',
        'points':points,'sources':SOURCES}
    (HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/70m-final-kernel.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# 70M final kernel: retained checkpoints','',
        'S_model is canonical pooled FP16 logical-product sparsity. Native/candidate latency is BF16, B1/T2048, full50304 logits. Each point pools1344 timing pairs in three fresh processes. Process ranges are not confidence intervals. Failed qualification is retained and cannot support a runtime gain.','',
        '| Family | Kappa | S_model (%) | Native (ms) | Port (ms) | Speedup | Qualified |',
        '|---|---:|---:|---:|---:|---:|---|']
    for p in points:
        k='N/A' if p['kappa'] is None else f"{p['kappa']:g}"
        lines.append(f"| {p['family']} | {k} | {p['sparsity_percent']:.6f} | {p['native_gm_ms']:.6f} | {p['candidate_gm_ms']:.6f} | {p['speedup']:.4f}x | {p['qualified']} |")
    (HERE/'TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'points':len(points),'qualified':sum(p['qualified'] for p in points),'device_uuid':data['device_uuid']}))

if __name__=='__main__':main()
