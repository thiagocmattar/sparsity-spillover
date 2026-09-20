"""Audit the frozen105-process study, then reduce paired timing and counts."""
import math
import numpy as np
from common import RUN,ROOT,manifest,read_json,write_json,verify_record,record,sha256

MODES=['native','native_graph','previous','previous_graph','sparse','sparse_graph','no_skip_graph','attention_dense_graph']
COMPARISONS={'eager':('native','sparse'),'graph':('native_graph','sparse_graph'),
    'skip':('no_skip_graph','sparse_graph'),'attention':('attention_dense_graph','sparse_graph'),
    'previous_eager':('native','previous'),'previous_graph':('native_graph','previous_graph'),
    'new_over_previous_graph':('previous_graph','sparse_graph')}


def paired_reduction(timings,reference,target,*,inputs=64,passes=7):
    per_process=[];differences=[]
    for timing in timings:
        cells={}
        for sample in timing['samples']:
            if sample['mode'] not in {reference,target}:continue
            key=sample['input_index'],sample['repeat']
            if not 0<=key[0]<inputs or not 0<=key[1]<passes:raise ValueError('Invalid timing index')
            cell=cells.setdefault(key,{})
            value=sample['host_ms']
            if sample['mode'] in cell or not math.isfinite(value) or value<=0:raise ValueError('Invalid/duplicate time')
            cell[sample['mode']]=value
        if len(cells)!=inputs*passes or any(set(c)!={reference,target} for c in cells.values()):raise ValueError('Unpaired timing')
        per_process.append(math.exp(np.mean([math.log(c[reference]/c[target]) for c in cells.values()])))
        differences.append(float(np.mean([c[reference]-c[target] for c in cells.values()])))
    return {'ratio':math.exp(np.log(per_process).mean()),'process_min':min(per_process),
            'process_max':max(per_process),'process_ratios':per_process,
            'mean_saved_ms':float(np.mean(differences))}


def audit_diagnostics(d,architecture,*,blocks=338):
    sites={row['name']:row for row in d['per_site_layer']}
    expected={f'{s}.layer_{i}' for s in ['a','m','h','z','q_post','k_post','v'] for i in range(6)}
    if set(sites)!=expected or set(d['active_features_per_row'])!=expected:raise ValueError('Missing site/layer')
    for name,hist in d['active_features_per_row'].items():
        site=name.split('.')[0];width=512 if site=='h' else 32 if site in {'q_post','k_post','v'} else 128
        rows=blocks*2048*(4 if width==32 else 1);stat=sites[name]
        if len(hist)!=width+1 or sum(hist)!=rows or any(type(n)!=int or n<0 for n in hist):raise ValueError('Invalid occupancy histogram')
        if stat['total']!=rows*width or sum(i*n for i,n in enumerate(hist))+stat['exact_zero_count']!=stat['total']:raise ValueError('Histogram/zero count mismatch')
        if stat['nonfinite'] or stat['finite']!=stat['total']:raise ValueError('Nonfinite activation')
    lower=d['bf16_scalar_opportunity_lower_bound'];counts=lower['per_operation'];reduced={}
    for operation,perblock in architecture['per_block_operation_products'].items():
        c=counts[operation];den=perblock*architecture['layers']*blocks
        if c['product_count']!=den or not 0<=c['zero_product_count']<=den:raise ValueError('Invalid logical counts')
        reduced[operation]={'products':den,'zero_products_lower_bound':c['zero_product_count'],
                            'zero_fraction_lower_bound':c['zero_product_count']/den}
        if operation in {'qkv_projection','mlp_w1','mlp_w2','attention_output_projection'}:
            site,n={'qkv_projection':('a',384),'mlp_w1':('m',512),'mlp_w2':('h',128),'attention_output_projection':('z',128)}[operation]
            if sum(v['exact_zero_count'] for k,v in sites.items() if k.startswith(site+'.'))*n!=c['zero_product_count']:raise ValueError('Projection zero counts differ')
            total=c['issued_mmas']+c['skipped_mmas']
            if total*2048!=den or not 0<=c['skipped_mmas']<=total:raise ValueError('Invalid projection MMA counts')
            reduced[operation].update(mma_issued=c['issued_mmas'],mma_skipped=c['skipped_mmas'],mma_total=total,mma_skipped_fraction=c['skipped_mmas']/total)
    if set(d['attention_counts_by_layer'])!=set(map(str,range(6))):raise ValueError('Attention layer coverage')
    for layer,counts4 in d['attention_counts_by_layer'].items():
        if len(counts4)!=4 or any(type(n)!=int or n<0 for n in counts4):raise ValueError('Invalid attention counter')
        if counts4[0]+counts4[1]!=147456*blocks or counts4[2]+counts4[3]!=147456*blocks:raise ValueError('Attention counter conservation')
    for operation,offset in [('qk_scores',0),('probability_value',2)]:
        issued=sum(v[offset] for v in d['attention_counts_by_layer'].values())
        skipped=sum(v[offset+1] for v in d['attention_counts_by_layer'].values())
        reduced[operation].update(mma_issued=issued,mma_skipped=skipped,mma_total=issued+skipped,mma_skipped_fraction=skipped/(issued+skipped))
    if sum(c['zero_product_count'] for c in lower['per_operation'].values())!=lower['zero_products']:raise ValueError('Unpooled BF16 numerator')
    if lower['model_products']!=architecture['model_product_count']*blocks or lower['fraction']!=lower['zero_products']/lower['model_products']:raise ValueError('Wrong BF16 denominator')
    return reduced


def main():
    source=manifest();policy=read_json(RUN/'final-policy-001.json');policy_record=record(RUN/'final-policy-001.json')
    for s in policy['sources']:verify_record(s)
    matrix=read_json(RUN/'artifacts/final-matrix-001/result.json')
    if matrix['status']!='complete' or matrix['completed']!=105:raise ValueError('Matrix incomplete; do not publish a complete-cohort figure')
    rows=[];sources=[];cfg=read_json(RUN/'config.json')
    expected_indices=np.random.default_rng(cfg['timing_seed']).choice(338,64,replace=False).tolist()
    for checkpoint in source['checkpoints']:
        for s in checkpoint['files']+checkpoint['provenance']:verify_record(s)
        logical=checkpoint['canonical_logical_products']['measured'];total=logical['model_product_count'];zeros=logical['block_zero_product_count']
        if sum(c['zero_product_count'] for c in logical['per_operation'].values())!=zeros:raise ValueError('Canonical numerator mismatch')
        if sum(c['product_count'] for c in logical['per_operation'].values())+logical['lm_head_product_count']!=total:raise ValueError('Canonical denominator mismatch')
        timings=[];qualities=[];failures=[]
        for replicate in range(1,4):
            dest=RUN/'artifacts'/f"final-{checkpoint['id']}-p{replicate}-001"
            result=read_json(dest/'result.json')
            if result['status']!='complete':
                failures.append({'replicate':replicate,'result':result});continue
            identity=read_json(dest/'manifest.json');quality=read_json(dest/'quality.json');timing=read_json(dest/'timing.json')
            args=result['arguments']
            if args!=identity['arguments'] or (args['condition'],args['replicate'],args['attempt'],args['smoke'])!=(checkpoint['id'],replicate,dest.name,False):raise ValueError('Identity mismatch')
            if result['policy']!=policy_record or identity['policy']!=policy:raise ValueError('Policy changed')
            if identity['checkpoint']!=checkpoint or identity['inputs']!=source['inputs']:raise ValueError('Input provenance changed')
            if identity['gpu']!='NVIDIA GeForce RTX 5090' or (identity['torch'],identity['cuda'])!=('2.11.0+cu128','12.8'):raise ValueError('Runtime changed')
            maps=read_json(dest/'source-map.json')
            if [r['original'] for r in maps]!=[policy_record]+policy['sources']:raise ValueError('Source map mismatch')
            for mapping in maps:
                snapshot=dest/mapping['snapshot'];original=mapping['original']
                if snapshot.stat().st_size!=original['bytes'] or sha256(snapshot)!=original['sha256']:raise ValueError('Snapshot drift')
            if (quality['blocks'],quality['documents'],quality['input_tokens'],quality['prediction_tokens'],quality['excluded_tail_tokens'])!=(338,500,692224,691886,1444):raise ValueError('Incomplete validation')
            if set(quality['pass'])!=set(MODES) or set(result['qualified'])!=set(MODES):raise ValueError('Missing quality mode')
            for mode in set(MODES)-{'native'}:
                gates=quality['gates'][mode]
                if [g['input_index'] for g in gates]!=list(range(338)):raise ValueError('Missing block')
                valid=all(g['pass'] for g in gates) and abs(quality['loss_delta'][mode])<=.001
                if quality['pass'][mode]!=valid or result['qualified'][mode]!=valid:raise ValueError('Qualification mismatch')
            if timing['indices']!=expected_indices or len(timing['samples'])!=8*64*7:raise ValueError('Timing coverage changed')
            if {s['mode'] for s in timing['samples']}!=set(MODES) or any(s['output_shape']!=[1,2048,50304] for s in timing['samples']):raise ValueError('Timed graph changed')
            timings.append(timing);qualities.append(quality)
            sources.extend(record(dest/name) for name in ['result.json','manifest.json','quality.json','timing.json','source-map.json'])
        row={'condition':checkpoint['id'],'family':checkpoint['family'],'dose':checkpoint['dose'],
             'historical':checkpoint['historical'],'R_model_percent':100*zeros/total,
             'canonical_zero_products':zeros,'canonical_model_products':total,'failures':failures}
        if failures:
            row.update(complete=False);rows.append(row);continue
        dpath=RUN/'artifacts'/f"final-{checkpoint['id']}-p1-001/diagnostics.json";d=read_json(dpath)
        if d['coverage']!={'blocks':338,'documents':500,'input_tokens':692224,'excluded_tail_tokens':1444}:raise ValueError('Diagnostic coverage changed')
        operations=audit_diagnostics(d,checkpoint['canonical_logical_products']['architecture_maximum'])
        sources.append(record(dpath));mode_rows={}
        for mode in MODES:
            samples=[s['host_ms'] for t in timings for s in t['samples'] if s['mode']==mode]
            gates=[g for q in qualities for g in q['gates'].get(mode,[])]
            mode_rows[mode]={'qualified':all(q['pass'][mode] for q in qualities),
                'latency_median_ms':float(np.median(samples)),
                'process_median_ms':[float(np.median([s['host_ms'] for s in t['samples'] if s['mode']==mode])) for t in timings],
                'max_abs_logit_error':max([g['max_abs'] for g in gates],default=0),
                'max_relative_l2':max([g['relative_l2'] for g in gates],default=0),
                'max_loss_delta':max(abs(q['loss_delta'][mode]) for q in qualities),
                'failed_blocks':len({g['input_index'] for g in gates if not g['pass']}),
                'losses':[q['loss'][mode] for q in qualities]}
        comparisons={}
        for label,(reference,target) in COMPARISONS.items():
            comparisons[label]={**paired_reduction(timings,reference,target),
                                'qualified':mode_rows[reference]['qualified'] and mode_rows[target]['qualified']}
        total_saved=comparisons['graph']['mean_saved_ms'];skip_saved=comparisons['skip']['mean_saved_ms']
        row.update(complete=True,modes=mode_rows,comparisons=comparisons,operations=operations,
                   BF16_scalar_lower_bound_percent=100*d['bf16_scalar_opportunity_lower_bound']['fraction'],
                   skip_share_of_graph_latency_saving=skip_saved/total_saved if total_saved>0 else None)
        rows.append(row)
    if len(rows)!=35:raise ValueError('Missing checkpoint')
    summary={'rows':rows,'policy':policy_record,'sources':sources,'reducer':record(__file__),
        'coverage':{'checkpoints':35,'processes':105,'timing_inputs':64,'paired_passes':7,'validation_blocks_per_process':338,'validation_documents':500,'excluded_tail_tokens':1444},
        'uncertainty':'Points are geometric means of paired ratios, equally weighted across three processes; bars are min-max of process geometric means, not confidence intervals or training-seed uncertainty.',
        'counter_limits':'Canonical FP16 logical products and actual BF16 operands differ; scalar lower bound excludes weight zeros and probability underflow. Physical MMA counts include attention causal padding and do not measure runtime or memory traffic.'}
    write_json(RUN/'results/summary.json',summary)
    print({'variants':len(rows),'complete':sum(r['complete'] for r in rows),
           'qualified_new':sum(r['complete'] and r['modes']['sparse_graph']['qualified'] for r in rows)})


if __name__=='__main__':main()
