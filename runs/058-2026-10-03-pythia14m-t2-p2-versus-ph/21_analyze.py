"""Run058 paired Ph/P2 quality and count-pooled sparsity with source hashes."""
from pathlib import Path
from statistics import mean,median
import hashlib
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_config import RUN_DIR as RUN,load_config,condition_specs,write_json


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def collect():
    config=load_config()
    specs=condition_specs(config)
    verification=load(RUN/'artifacts/verification.json')
    if verification['status']!='verified' or verification['condition_count']!=len(specs):
        raise ValueError('Complete verified cohort required')
    rows=[];sources=[]
    for attempt in sorted((RUN/'artifacts/attempts').iterdir()):
        if not attempt.is_dir():
            continue
        metrics=load(attempt/'metrics.json')
        manifest=load(attempt/'manifest.json')
        condition=metrics['condition']
        if manifest['status']!='completed':
            raise ValueError('Terminal success required')
        act=load(attempt/'diagnostics/activation_statistics.json')
        moments={r['name']:r for r in act['pooled_by_site']}
        occupancy={r['name']:r for r in act['occupancy_pooled_by_site']}
        for e in act['sign_energy_rows']:
            for prefix in ('pre','post'):
                if sum(e[f'{prefix}_{s}'] for s in ('negative','zero','positive'))!=e['elements']:
                    raise ValueError('Sign counts do not partition the tensor')
            if e['post_negative'] or not math.isclose(e['pre_square_sum'],e['post_square_sum']+e['rejected_square_sum'],rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError('Unilateral sign/energy audit failed')
        logical=load(attempt/'diagnostics/logical_products.json')
        measured=logical['measured']
        if sum(o['zero_product_count'] for o in measured['per_operation'].values())!=measured['block_zero_product_count']:
            raise ValueError('Logical operation counts do not pool')
        if not math.isclose(measured['R_model'],measured['block_zero_product_count']/measured['model_product_count'],rel_tol=1e-12):
            raise ValueError('Logical fraction differs from integer ratio')
        sites={}
        for site in ('a','m','h','z'):
            m=moments[site];o=occupancy[site];hist=o['row_nonzero_histogram']
            if sum(hist)!=o['rows'] or sum(i*n for i,n in enumerate(hist))!=m['total']-m['exact_zero_count']:
                raise ValueError('Histogram differs from element counts')
            for shape,c in o['tiles'].items():
                if c['total']*math.prod(map(int,shape.split('x')))!=m['total'] or not 0<=c['empty']<=c['total']:
                    raise ValueError('Tiles do not partition complete coverage')
            sites[site]={
                'zero_percent':100*m['exact_zero_count']/m['total'],
                'tile_8x16_empty_percent':100*o['tiles']['8x16']['empty']/o['tiles']['8x16']['total'],
                'tile_16x16_empty_percent':100*o['tiles']['16x16']['empty']/o['tiles']['16x16']['total'],
                'rms':m['rms'],'mean':m['mean'],'near_zero_fractions':m['threshold_fractions'],
                'integer_counts':{'elements':m['total'],'zeros':m['exact_zero_count'],'tiles':o['tiles'],'row_nonzero_histogram':hist}}
        updates=[json.loads(line) for line in (attempt/'events.jsonl').read_text().splitlines() if json.loads(line).get('event')=='train']
        if len(updates)!=config['training']['max_steps'] or any(u['optimizer_step_skipped'] for u in updates):
            raise ValueError('Incomplete or skipped updates')
        gradients={}
        for label,window in (('all_updates',updates),('final_64_updates',updates[-64:])):
            gradients[label]={'updates':len(window),'mean_pressure_loss':mean(u['pressure_loss'] for u in window),
                'mean_gradient_cosine':mean(u['task_pressure_gradient_cosine'] for u in window),
                'conflict_updates':sum(u['gradient_conflict'] for u in window),
                'projection_updates':sum(u['projection_applied'] for u in window),
                'median_trust_scale':median(u['trust_scale'] for u in window),
                'median_final_pressure_task_ratio':median(u['pressure_to_task_ratio_final'] for u in window)}
        rows.append({'condition':condition['id'],'pressure':condition['pressure_recipe'],'kappa':condition['gate_threshold'],
            'pressure_sites':condition['pressure_sites'],'validation_loss':metrics['validation']['final']['loss'],
            'R_model_percent':100*measured['R_model'],'R_block_percent':100*measured['R_block'],
            'R_model_max_percent':logical['architecture_maximum']['R_model_max_percent'],
            'logical_counts':measured,'sites':sites,'gradient_summary':gradients,
            'initial_parameter_sha256':manifest['initial_parameter_sha256'],
            'schedule_sha256':metrics['training']['schedule_hash'],
            'training_seconds':metrics['timing']['total_seconds'],
            'median_step_seconds':metrics['training']['median_step_seconds'],
            'training_curve':[{'step':u['step'],'task_loss':u['task_loss']} for u in updates]})
        for name in ('manifest.json','metrics.json','config.yaml','events.jsonl','diagnostics/activation_statistics.json','diagnostics/logical_products.json','diagnostics/weight_statistics.json'):
            p=attempt/name
            sources.append({'path':p.relative_to(RUN).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    if len(rows)!=len(specs) or {r['condition'] for r in rows}!={s['id'] for s in specs}:
        raise ValueError('Missing or duplicate condition')
    if len({r['initial_parameter_sha256'] for r in rows})!=1 or len({r['schedule_sha256'] for r in rows})!=1:
        raise ValueError('Initialization/data order not matched')
    rows.sort(key=lambda r:(r['kappa'],('ph','p2').index(r['pressure'])))
    pairs=[]
    for kappa in config['conditions']['gate_thresholds']:
        ph,p2=(next(r for r in rows if r['pressure']==p and r['kappa']==kappa) for p in ('ph','p2'))
        pairs.append({'kappa':kappa,'ph':ph['condition'],'p2':p2['condition'],
            'loss_delta_p2_minus_ph':p2['validation_loss']-ph['validation_loss'],
            'R_model_delta_percentage_points':p2['R_model_percent']-ph['R_model_percent'],
            'site_deltas_percentage_points':{s:{key:p2['sites'][s][key]-ph['sites'][s][key] for key in ('zero_percent','tile_8x16_empty_percent','tile_16x16_empty_percent')} for s in ('a','m','h','z')}})
    return rows,pairs,sources


def table(rows,pairs):
    text='# T2/P2 versus T2/Ph: 14M, seed1234\n\n'
    text+='Complete500-document validation:338 complete2048-token blocks; excluded tail1444 tokens. All gates are unilateral at h/z only. Zeros and tiles are count-pooled percentages.\n\n'
    text+='| Pressure | kappa | Validation loss | R_model (%) | A zeros (%) | A empty8x16 (%) | M zeros (%) | M empty8x16 (%) | H zeros (%) | H empty8x16 (%) | Z zeros (%) | Z empty8x16 (%) |\n'
    text+='|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        values=[r['pressure'].upper(),f"{r['kappa']:g}",f"{r['validation_loss']:.6f}",f"{r['R_model_percent']:.4f}"]
        for site in ('a','m','h','z'):
            values.extend(f"{r['sites'][site][key]:.6f}" for key in ('zero_percent','tile_8x16_empty_percent'))
        text+='| '+' | '.join(values)+' |\n'
    text+='\nPaired differences are P2 minus Ph. Negative loss delta favors P2.\n\n'
    text+='| kappa | Loss difference | R_model difference (pp) | H zeros difference (pp) | H tiles difference (pp) | Z zeros difference (pp) | Z tiles difference (pp) |\n|---:|---:|---:|---:|---:|---:|---:|\n'
    for p in pairs:
        d=p['site_deltas_percentage_points']
        values=[f"{p['kappa']:g}",f"{p['loss_delta_p2_minus_ph']:+.6f}",f"{p['R_model_delta_percentage_points']:+.4f}"]
        values.extend(f"{d[s][key]:+.4f}" for s in ('h','z') for key in ('zero_percent','tile_8x16_empty_percent'))
        text+='| '+' | '.join(values)+' |\n'
    text+='\nEmpty tiles require all128 values to equal zero. R_model is logical-product opportunity, not runtime speedup. Equal lambda uses a mean over6 tensors in Ph versus12 in P2; this is a recipe comparison with one seed, not matched gradient intensity.\n'
    return text


def figures(rows):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.grid':True,'grid.alpha':.18})
    colors={'ph':'#17688a','p2':'#bb532d'}
    folder=RUN/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),layout='constrained')
    for ax,key,label in zip(axes,('validation_loss','R_model_percent'),('Validation cross-entropy','Logical opportunity R_model (%)')):
        for pressure in ('ph','p2'):
            rr=sorted((r for r in rows if r['pressure']==pressure),key=lambda r:r['kappa'])
            ax.plot([r['kappa'] for r in rr],[r[key] for r in rr],marker='o',color=colors[pressure],label='Ph: pressure h' if pressure=='ph' else 'P2: pressure h,z')
        ax.set(xlabel='Threshold kappa',ylabel=label)
        ax.set_xticks([0,.1]);ax.legend(frameon=False)
    fig.suptitle('14M T2: same h/z gates, change pressure targets',fontsize=11)
    fig.savefig(folder/'01-paired-quality-opportunity.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(8.2,5.8),layout='constrained')
    labels=[f"{r['pressure'].upper()}\nk={r['kappa']:g}" for r in rows]
    for ax,site in zip(axes.flat,('a','m','h','z')):
        for i,r in enumerate(rows):
            color=colors[r['pressure']]
            ax.bar(i-.18,r['sites'][site]['zero_percent'],width=.34,color=color,label='Zero elements' if i==0 else None)
            ax.bar(i+.18,r['sites'][site]['tile_8x16_empty_percent'],width=.34,facecolor='white',edgecolor=color,hatch='///',label='Empty 8x16 tiles' if i==0 else None)
        ax.set_xticks(range(len(rows)),labels);ax.set(title=f'Site {site.upper()}',ylabel='Count-pooled fraction (%)',ylim=(0,105))
        if site in ('a','m'):
            ax.text(.5,.5,'0% zero elements\n0% empty tiles\n(all four conditions)',
                transform=ax.transAxes,ha='center',va='center',fontsize=11,color='#555555')
    handles,legend_labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,-.025),
        ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Element sparsity and fully empty aligned tiles',fontsize=11)
    fig.savefig(folder/'02-site-zeros-tiles.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    rows,pairs,sources=collect()
    result={'rows':rows,'pairs':pairs,'sources':sources,'analysis_script_sha256':sha(Path(__file__)),
        'coverage':{'documents':500,'sequences':338,'input_tokens':692224,'excluded_tail_tokens':1444,'layers':6,'seed':1234},
        'tile_definition':'nonoverlapping token x feature tiles within each sequence, no padding, all elements exactly zero'}
    (RUN/'results').mkdir(exist_ok=True)
    write_json(RUN/'results/results.json',result)
    (RUN/'results/tables.md').write_text(table(rows,pairs),encoding='utf-8',newline='\n')
    figures(rows)
    conclusion=[]
    for p in pairs:
        direction='lowers' if p['loss_delta_p2_minus_ph']<0 else 'raises'
        conclusion.append(f"At kappa={p['kappa']:g}, P2 {direction} loss versus Ph by {p['loss_delta_p2_minus_ph']:+.6f}; logical opportunity changes by {p['R_model_delta_percentage_points']:+.4f} percentage points.")
    obs=RUN/'observations';obs.mkdir(exist_ok=True)
    (obs/'001-p2-versus-ph.md').write_text(
        '# T2/P2 versus T2/Ph\n\nQuestion: does adding z to OL1 pressure improve the same h/z gate recipe?\n\n'
        'Method: fresh matched random14M runs, seed1234,712 updates, kappa0/.1, h/z unilateral gates; Ph targets h, P2 targets h/z. Pool integer counts across all6 layers before dividing.\n\n'
        'Coverage: all500 MiniPile validation documents,338 complete2048-token blocks,692224 input tokens, excluded1444-token tail. All checkpoint and diagnostic source identities are linked in results/results.json.\n\n'
        'Result: '+' '.join(conclusion)+'\n\n'
        'Figure01: Ph blue and P2 orange; final full-validation cross-entropy (lower is better) and count-pooled logical-product opportunity (higher means more logical zeros). Lines connect evaluated thresholds; intermediate values were not measured.\n\n'
        'Figure02: each site shows zero-element percentages (solid bars) and fully empty8x16 tiles (hatched bars). Tiles stay within sequences, partition all values without padding, and require all128 entries to be exactly zero. Both use count-first pooling.\n\n'
        'Caveats: one seed, two thresholds, equal cutoff does not match sparsity. Pressure averages site/layer tensor means, so adding z changes the relative h contribution. Empty tiles and logical opportunity are not measured speedup. No tile-pressure intervention or manuscript change.\n\n'
        'Source: ../21_analyze.py; ../results/results.json and ../results/tables.md; ../figures/01-paired-quality-opportunity.pdf and ../figures/02-site-zeros-tiles.pdf.\n',
        encoding='utf-8',newline='\n')
    (obs/'INDEX.md').write_text('# Run058 observations\n\n- [001-p2-versus-ph.md](001-p2-versus-ph.md): matched quality, logical opportunity and sitewise zeros/empty tiles for T2/Ph versus T2/P2.\n',encoding='utf-8',newline='\n')
    print(json.dumps({'rows':len(rows),'pairs':pairs,'sources':len(sources)},indent=2))
