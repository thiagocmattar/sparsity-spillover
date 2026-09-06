"""Complete-cohort count/quality audit and paired three-process timing reduction."""
import csv
import math
import numpy as np
from run027_common import RUN,read_json,write_json,verify_record,record


def ratio_cube(timings,reference,target,*,n_inputs=64,passes=7):
    cube=np.full((len(timings),n_inputs,passes),np.nan)
    raw={reference:[],target:[]}
    for process,timing in enumerate(timings):
        cells={}
        for row in timing['samples']:
            if row['mode'] not in raw:continue
            key=(row['input_index'],row['repeat']); mode=row['mode']
            if not 0<=key[0]<n_inputs or not 0<=key[1]<passes:raise ValueError('Invalid timing cell')
            cell=cells.setdefault(key,{})
            if mode in cell or not math.isfinite(row['host_ms']) or row['host_ms']<=0:raise ValueError('Invalid/duplicate time')
            cell[mode]=row['host_ms'];raw[mode].append(row['host_ms'])
        for (i,j),cell in cells.items():
            if set(cell)!=set(raw):raise ValueError('Unpaired timing')
            cube[process,i,j]=math.log(cell[reference]/cell[target])
    if not np.isfinite(cube).all():raise ValueError('Missing timing samples')
    return cube,raw


def interval(cube,draws=10000,seed=2706):
    means=cube.mean(axis=2);rng=np.random.default_rng(seed)
    # Crossed process and shared-input clusters; retain all seven repeats.
    processes=rng.integers(means.shape[0],size=(draws,means.shape[0],1))
    inputs=rng.integers(means.shape[1],size=(draws,1,means.shape[1]))
    values=np.exp(means[processes,inputs].mean(axis=(1,2)))
    return np.quantile(values,[.025,.975]).tolist()


def main():
    cfg=read_json(RUN/'config.json'); manifest=read_json(RUN/'prelaunch/inputs.json')
    frozen=read_json(RUN/'prelaunch/frozen.json')
    rows=[];long_rows=[];sources=[]
    for condition in manifest['checkpoints']:
        count=condition['canonical_logical_products']['measured']
        total=count['model_product_count'];zero=count['block_zero_product_count']
        if sum(r['zero_product_count'] for r in count['per_operation'].values())!=zero:raise ValueError('Unpooled numerator')
        if sum(r['product_count'] for r in count['per_operation'].values())+count['lm_head_product_count']!=total:raise ValueError('Unpooled denominator')
        timings=[];qualities=[];native_losses=[]
        for replicate in range(1,4):
            dest=RUN/'artifacts'/f'{condition["id"]}-r{replicate}'
            result=read_json(dest/'result.json'); identity=read_json(dest/'manifest.json')
            if result['status']!='complete' or result['arguments']['smoke']:raise ValueError('Incomplete characterization')
            if result['condition']!=condition['id'] or result['validation_blocks']!=338:raise ValueError('Wrong cohort or coverage')
            if identity['checkpoint']['files']!=condition['files']:raise ValueError('Checkpoint identity mismatch')
            if identity['inputs']!=manifest['inputs']:raise ValueError('Validation input identity mismatch')
            if result['canonical_logical_products']!=condition['canonical_logical_products']:raise ValueError('Logical provenance changed')
            hashes={r['path']:r['sha256'] for r in identity['sources']+identity['source_helpers']}
            for source in frozen['files']:
                verify_record(source)
                if hashes.get(source['path'])!=source['sha256']:raise ValueError('Implementation changed during matrix')
            quality=read_json(dest/'quality.json')
            if (quality['documents'],quality['blocks'],quality['input_tokens'],quality['prediction_tokens'],quality['excluded_tail'])!=(500,338,692224,691886,1444):raise ValueError('Coverage changed')
            for mode in ['fusion_dense','sparse','no_skip']:
                if [g['input_index'] for g in quality['gates'][mode]]!=list(range(338)):raise ValueError('Missing numerical block')
            timing=read_json(dest/'timing.json')
            if any(s['output_shape']!=[1,2048,50304] for s in timing['samples']):raise ValueError('Incomplete timed logits')
            expected=np.random.default_rng(cfg['timing_seed']).choice(338,64,replace=False).tolist()
            if timing['indices']!=expected:raise ValueError('Timing cohort changed')
            timings.append(timing);qualities.append(quality);native_losses.append(quality['loss']['native'])
            sources.extend(record(dest/name) for name in ['result.json','manifest.json','quality.json','timing.json'])
        diagnostic=read_json(RUN/'artifacts'/f'{condition["id"]}-r1/diagnostics.json')
        if diagnostic['coverage']!={'documents':500,'blocks':338,'input_tokens':692224,'excluded_tail':1444}:raise ValueError('Incomplete activation coverage')
        sources.append(record(RUN/'artifacts'/f'{condition["id"]}-r1/diagnostics.json'))
        if diagnostic['eligible_products']!=338*2048*6*(512+128)*128:raise ValueError('Eligible denominator changed')
        row={'condition':condition['id'],'family':condition['family'],'dose':condition['dose'],
            'historical':condition['historical'],'R_model':zero/total,'R_model_percent':100*zero/total,
            'zero_products':zero,'model_products':total,
            'eligible_BF16_zero_fraction':diagnostic['eligible_zero_fraction'],
            'eligible_BF16_zero_products':diagnostic['eligible_zero_products'],
            'eligible_products':diagnostic['eligible_products'],
            'eligible_zero_share_of_model':diagnostic['eligible_zero_products']/total,
            'native_validation_loss':float(np.mean(native_losses)),'replicates':3,'pairs':1344}
        for mode in cfg['modes']:
            all_samples=[s['host_ms'] for t in timings for s in t['samples'] if s['mode']==mode]
            row[f'{mode}_latency_ms']=float(np.median(all_samples))
            row[f'{mode}_qualified']=all(q['pass'][mode] for q in qualities)
            errors=[g['max_abs'] for q in qualities for g in q['gates'].get(mode,[])]
            row[f'{mode}_max_abs_logit_difference']=None if None in errors else max(errors,default=0.)
            row[f'{mode}_max_loss_delta']=max(abs(q['loss_delta'][mode]) for q in qualities)
            relative=[g['relative_l2'] for q in qualities for g in q['gates'].get(mode,[])]
            row[f'{mode}_max_relative_l2']=None if None in relative else max(relative,default=0.)
            row[f'{mode}_failed_validation_blocks']=len({g['input_index'] for q in qualities for g in q['gates'].get(mode,[]) if not g['pass']})
            row[f'{mode}_failed_processes']=sum(not q['pass'][mode] for q in qualities)
            long_rows.append({'condition':condition['id'],'family':condition['family'],'dose':condition['dose'],
                'mode':mode,'latency_ms':row[f'{mode}_latency_ms'],'qualified':row[f'{mode}_qualified'],
                'R_model':zero/total,'zero_products':zero,'model_products':total})
        for reference,target,label in [('native','sparse','overall'),('fusion_dense','sparse','beyond_qkv_fusion'),
                                        ('no_skip','sparse','skip_only'),('native','fusion_dense','qkv_fusion')]:
            cube,_=ratio_cube(timings,reference,target)
            low,high=interval(cube,cfg['bootstrap_draws'],cfg['bootstrap_seed'])
            row[label+'_speedup']=float(np.exp(cube.mean()))
            row[label+'_ci_low']=low;row[label+'_ci_high']=high
            row[label+'_qualified']=row[reference+'_qualified'] and row[target+'_qualified']
        for mode_row in long_rows[-4:]:
            mode=mode_row['mode']
            mode_row['speedup_over_native']=1. if mode=='native' else float(np.exp(ratio_cube(timings,'native',mode)[0].mean()))
        # Paired arithmetic latency differences, not differences of speedup ratios.
        pair_times={mode:[] for mode in cfg['modes']}
        for timing in timings:
            cells={}
            for sample in timing['samples']:
                cells.setdefault((sample['input_index'],sample['repeat']),{})[sample['mode']]=sample['host_ms']
            for cell in cells.values():
                for mode in cfg['modes']:pair_times[mode].append(cell[mode])
        row['mean_total_saved_ms']=float(np.mean(np.array(pair_times['native'])-pair_times['sparse']))
        row['mean_skip_saved_ms']=float(np.mean(np.array(pair_times['no_skip'])-pair_times['sparse']))
        row['mean_non_skip_net_saved_ms']=row['mean_total_saved_ms']-row['mean_skip_saved_ms']
        row['skip_share_of_total_latency_saving']=row['mean_skip_saved_ms']/row['mean_total_saved_ms'] if row['mean_total_saved_ms']>0 else None
        rows.append(row)
    if len(rows)!=35:raise ValueError('Missing variant')
    (RUN/'results').mkdir(exist_ok=True)
    for name,data in [('summary.csv',rows),('all-variants.csv',long_rows)]:
        with (RUN/'results'/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write_json(RUN/'results/summary.json',{'rows':rows,'sources':sources,'reducer':record(__file__),
        'coverage':{'checkpoints':35,'processes_per_checkpoint':3,'validation_blocks':338,'timing_inputs':64,'passes':7},
        'ci':'10000 crossed process/input percentile bootstrap draws, seed2706; repeats retained in clusters; not independent checkpoints or training-seed uncertainty'})
    print({'variants':len(rows),'qualified_sparse':sum(r['sparse_qualified'] for r in rows),
           'faster_than_native':sum(r['overall_qualified'] and r['overall_speedup']>1 for r in rows)})


if __name__=='__main__':main()
