"""Paired one-seed results with source hashes and publication PDF figures."""
import hashlib
import json
import math
from statistics import mean, median
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_config import RUN_DIR as RUN,load_config,condition_specs,write_json

def load(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def by_name(rows):return {r['name']:r for r in rows}
def percent(n,d):return 100*n/d if d else 0.0

def audit_sign_energy(act):
    """Check the saved operator behavior independently of the training loss."""
    for e in act['sign_energy_rows']:
        n=e['elements']
        for prefix in ('pre','post'):
            if sum(e[f'{prefix}_{s}'] for s in ('negative','zero','positive'))!=n:
                raise ValueError('Sign counts do not partition the tensor')
        if not math.isclose(e['pre_square_sum'],e['post_square_sum']+e['rejected_square_sum'],rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError('Gate squared-energy balance failed')
        if e['operator']=='one_sided_threshold' and e['post_negative']:
            raise ValueError('Unilateral gate retained a negative value')
        if e['operator']=='symmetric_threshold' and e['rejected_large_negative_count']:
            raise ValueError('Bilateral gate removed a negative value above its cutoff')
        if e['operator']=='symmetric_threshold' and e['kappa']==0 and e['rejected_count']:
            raise ValueError('Bilateral zero-cutoff gate is not identity')

def collect():
    verification=load(RUN/'artifacts/verification.json')
    if verification['status']!='verified' or verification['condition_count']!=8:
        raise ValueError('Only a verified eight-condition cohort can be analyzed')
    result=[];sources=[]
    for attempt in sorted((RUN/'artifacts/attempts').iterdir()):
        if not attempt.is_dir():continue
        metrics=load(attempt/'metrics.json');condition=metrics['condition']
        act=load(attempt/'diagnostics/activation_statistics.json')
        audit_sign_energy(act)
        moments=by_name(act['pooled_by_site']);energy=by_name(act['sign_energy_pooled_by_site'])
        tiles=by_name(act['occupancy_pooled_by_site'])
        row={'condition':condition['id'],'pressure':condition['pressure_recipe'],
            'gate':'bilateral' if condition['am_operator']=='symmetric_threshold' else 'unilateral',
            'kappa':condition['gate_threshold'],'validation_loss':metrics['validation']['final']['loss'],
            'training_seconds':metrics['timing']['total_seconds'], 'median_step_seconds':metrics['training']['median_step_seconds'],
            'initial_parameter_sha256':load(attempt/'manifest.json')['initial_parameter_sha256'],
            'schedule_sha256':metrics['training']['schedule_hash'],'sites':{}}
        row['layernorm_affine_summary']=[{'layer':r['layer'],'site':r['site'],
            'scale_mean':sum(r['weight'])/len(r['weight']),
            'scale_rms':math.sqrt(sum(x*x for x in r['weight'])/len(r['weight'])),
            'bias_mean':sum(r['bias'])/len(r['bias']),
            'bias_rms':math.sqrt(sum(x*x for x in r['bias'])/len(r['bias']))}
            for r in act['layernorm_affine']]
        row['realized_cutoffs']=[{key:r[key] for key in ('name','operator','input_dtype','kappa','effective_kappa')}
                                for r in act['sign_energy_rows']]
        logical=load(attempt/'diagnostics/logical_products.json')
        measured=logical['measured']
        if sum(r['zero_product_count'] for r in measured['per_operation'].values())!=measured['block_zero_product_count']:
            raise ValueError('Logical numerator does not pool its operation counts')
        if not math.isclose(measured['R_model'],measured['block_zero_product_count']/measured['model_product_count'],rel_tol=1e-12):
            raise ValueError('Logical model opportunity differs from its integer ratio')
        row.update(R_model_percent=100*logical['measured']['R_model'],R_block_percent=100*logical['measured']['R_block'],
                   R_model_max_percent=logical['architecture_maximum']['R_model_max_percent'])
        for site in ('a','m','h','z'):
            e=energy[site];t=tiles[site]
            if e['elements']!=moments[site]['total'] or e['post_zero']!=moments[site]['exact_zero_count']:
                raise ValueError('Moment and sign element/zero counts differ')
            histogram=t['row_nonzero_histogram']
            if sum(histogram)!=t['rows'] or sum(i*n for i,n in enumerate(histogram))!=e['elements']-e['post_zero']:
                raise ValueError('Row histogram differs from pooled element counts')
            for shape,counters in t['tiles'].items():
                tile_elements=math.prod(int(v) for v in shape.split('x'))
                if counters['total']*tile_elements!=e['elements'] or not 0<=counters['empty']<=counters['total']:
                    raise ValueError('Aligned tiles do not partition the declared coverage')
            row['sites'][site]={'zero_percent':percent(moments[site]['exact_zero_count'],moments[site]['total']),
                'pre_negative_percent':percent(e['pre_negative'],e['elements']),
                'post_negative_percent':percent(e['post_negative'],e['elements']),
                'removed_energy_percent':percent(e['rejected_square_sum'],e['pre_square_sum']),
                'removed_large_negative_energy_percent':percent(e['rejected_large_negative_square_sum'],e['pre_square_sum']),
                'pre_rms':moments[site+'_pre']['rms'] if site in ('a','m') else None,
                'pre_mean':moments[site+'_pre']['mean'] if site in ('a','m') else None,
                'pre_near_zero_0p01_percent':100*moments[site+'_pre']['threshold_fractions']['0.01'] if site in ('a','m') else None,
                'post_rms':moments[site]['rms'],
                'tile_8x16_empty_percent':percent(t['tiles']['8x16']['empty'],t['tiles']['8x16']['total']),
                'tile_16x16_empty_percent':percent(t['tiles']['16x16']['empty'],t['tiles']['16x16']['total']),
                'rows_at_most_2_nonzeros_percent':percent(sum(t['row_nonzero_histogram'][:3]),t['rows']),
                'integer_counts':{'elements':moments[site]['total'],'zeros':moments[site]['exact_zero_count'],
                                  'pre_negative':e['pre_negative'],'post_negative':e['post_negative'],
                                  'rejected_large_negative':e['rejected_large_negative_count'],
                                  'tiles':t['tiles'],'row_nonzero_histogram':t['row_nonzero_histogram']}}
        events=[json.loads(line) for line in (attempt/'events.jsonl').read_text().splitlines()]
        updates=[r for r in events if r.get('event')=='train']
        row['gradient_summary']={}
        for window,ww in (('all_updates',updates),('final_64_updates',updates[-64:])):
            row['gradient_summary'][window]={'updates':len(ww),
                'first_step':ww[0]['step'],'last_step':ww[-1]['step'],
                'mean_pressure_loss':mean(r['pressure_loss'] for r in ww),
                'mean_clipped_task_pressure_gradient_cosine':mean(r['task_pressure_gradient_cosine'] for r in ww),
                'gradient_conflict_updates':sum(r['gradient_conflict'] for r in ww),
                'ol1_projection_updates':sum(r['projection_applied'] for r in ww),
                'median_trust_scale':median(r['trust_scale'] for r in ww),
                'median_ol1_pressure_to_task_ratio_final':median(r['pressure_to_task_ratio_final'] for r in ww)}
        row['training_curve']=[{'step':r['step'],'task_loss':r['task_loss'],'pressure_loss':r['pressure_loss']}
                                for r in updates]
        result.append(row)
        for name in ('manifest.json','metrics.json','config.yaml','events.jsonl',
                     'diagnostics/activation_statistics.json','diagnostics/logical_products.json','diagnostics/weight_statistics.json'):
            path=attempt/name;sources.append({'path':path.relative_to(RUN).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size})
    expected={r['id'] for r in condition_specs(load_config())}
    if len(result)!=8 or {r['condition'] for r in result}!=expected:raise ValueError('Wrong or duplicate cohort')
    if len({r['initial_parameter_sha256'] for r in result})!=1 or len({r['schedule_sha256'] for r in result})!=1:
        raise ValueError('Pairing is not matched')
    result.sort(key=lambda r:(('ph','pall').index(r['pressure']),r['kappa'],('unilateral','bilateral').index(r['gate'])))
    pairs=[]
    for pressure in ('ph','pall'):
        for kappa in (0.,.1):
            pair={r['gate']:r for r in result if r['pressure']==pressure and r['kappa']==kappa}
            u,b=pair['unilateral'],pair['bilateral']
            pairs.append({'pressure':pressure,'kappa':kappa,'unilateral':u['condition'],'bilateral':b['condition'],
                'loss_delta_bilateral_minus_unilateral':b['validation_loss']-u['validation_loss'],
                'R_model_delta_percentage_points':b['R_model_percent']-u['R_model_percent'],
                'a_zero_delta_percentage_points':b['sites']['a']['zero_percent']-u['sites']['a']['zero_percent'],
                'm_zero_delta_percentage_points':b['sites']['m']['zero_percent']-u['sites']['m']['zero_percent']})
    return result,pairs,sources

def table(rows,pairs):
    text='# T4 gate-sign results (14M, seed 1234)\n\n'
    text+='All 500 MiniPile validation documents; 338 complete 2,048-token blocks; excluded tail: 1,444 tokens.\n'
    text+='Bilateral means a/m only. h/z remain unilateral. Values are final reloaded-checkpoint diagnostics.\n\n'
    text+='| Pressure | a/m gate | kappa | Validation loss | R_model (%) | a zeros (%) | m zeros (%) | h zeros (%) | z zeros (%) |\n'
    text+='|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        values=[r['pressure'].upper(),r['gate'],f"{r['kappa']:g}",f"{r['validation_loss']:.6f}",f"{r['R_model_percent']:.4f}"]
        values += [f"{r['sites'][s]['zero_percent']:.4f}" for s in ('a','m','h','z')]
        text+='| '+' | '.join(values)+' |\n'
    text+='\nPaired deltas are bilateral minus unilateral; negative loss delta favors bilateral.\n\n'
    text+='| Pressure | kappa | Loss delta | R_model delta (pp) | a zeros delta (pp) | m zeros delta (pp) |\n'
    text+='|---|---:|---:|---:|---:|---:|\n'
    for p in pairs:
        text+=f"| {p['pressure'].upper()} | {p['kappa']:g} | {p['loss_delta_bilateral_minus_unilateral']:+.6f} | {p['R_model_delta_percentage_points']:+.4f} | {p['a_zero_delta_percentage_points']:+.4f} | {p['m_zero_delta_percentage_points']:+.4f} |\n"
    text+='\nR_model is a logical scalar-product opportunity, not measured speedup.\n\n'
    text+='| Pressure | Gate | kappa | Site | Pre negative (%) | Pre near zero <=.01 (%) | Removed large-negative energy (%) | Pre mean | Pre RMS | Post RMS | Empty 8x16 tiles (%) |\n'
    text+='|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        for site in ('a','m'):
            s=r['sites'][site]
            text+=f"| {r['pressure'].upper()} | {r['gate']} | {r['kappa']:g} | {site} | {s['pre_negative_percent']:.4f} | {s['pre_near_zero_0p01_percent']:.4f} | {s['removed_large_negative_energy_percent']:.4f} | {s['pre_mean']:.6f} | {s['pre_rms']:.6f} | {s['post_rms']:.6f} | {s['tile_8x16_empty_percent']:.4f} |\n"
    text+='\nBoundary summaries below use the final 64 updates (649--712), with equal weight per update. Cosine compares clipped task gradients to unclipped pressure gradients; OL1 projection is measured in the Adam-preconditioned direction space.\n\n'
    text+='| Pressure | Gate | kappa | Mean pressure loss | Mean gradient cosine | Conflict updates / 64 | OL1 projections / 64 | Median trust scale | Median final pressure/task direction ratio |\n'
    text+='|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:
        g=r['gradient_summary']['final_64_updates']
        text+=f"| {r['pressure'].upper()} | {r['gate']} | {r['kappa']:g} | {g['mean_pressure_loss']:.6f} | {g['mean_clipped_task_pressure_gradient_cosine']:+.6f} | {g['gradient_conflict_updates']} | {g['ol1_projection_updates']} | {g['median_trust_scale']:.6f} | {g['median_ol1_pressure_to_task_ratio_final']:.6f} |\n"
    return text

def figures(rows):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'figure.dpi':120,'axes.grid':True,'grid.alpha':.18})
    colors={'unilateral':'#bb532d','bilateral':'#17688a'}
    folder=RUN/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),layout='constrained',sharey=True)
    for ax,pressure in zip(axes,('ph','pall')):
        for gate in ('unilateral','bilateral'):
            rr=sorted([r for r in rows if r['pressure']==pressure and r['gate']==gate],key=lambda r:r['kappa'])
            ax.plot([r['kappa'] for r in rr],[r['validation_loss'] for r in rr],marker='o',color=colors[gate],label=gate)
        ax.set(title=pressure.upper(),xlabel='Shared threshold kappa',ylabel='Final validation cross-entropy')
        ax.set_xticks([0,.1]);ax.legend(frameon=False)
    fig.suptitle('Change only the a/m sign rule; h/z remain unilateral',fontsize=11)
    fig.savefig(folder/'01-paired-quality.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6),layout='constrained')
    keys=[(p,k) for p in ('ph','pall') for k in (0.,.1)]
    labels=[f'{p.upper()}\nk={k:g}' for p,k in keys]
    for ax,site in zip(axes,('a','m')):
        for gate,offset in (('unilateral',-.18),('bilateral',.18)):
            rr=[next(r for r in rows if r['pressure']==p and r['kappa']==k and r['gate']==gate) for p,k in keys]
            ax.bar([i+offset for i in range(4)],[r['sites'][site]['removed_large_negative_energy_percent'] for r in rr],
                width=.34,color=colors[gate],label=gate)
            if gate=='bilateral':
                for i,r in enumerate(rr):
                    ax.text(i+offset,r['sites'][site]['removed_large_negative_energy_percent']+2,'0%',
                            color=colors[gate],ha='center',fontsize=8)
        ax.set_xticks(range(4),labels);ax.set(title=f'Site {site}',ylabel='Pre-gate squared energy removed (%)',ylim=(0,100))
        ax.legend(frameon=False)
    fig.suptitle('Removed negative values at or above the threshold magnitude',fontsize=11)
    fig.savefig(folder/'02-negative-energy.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6),layout='constrained')
    for ax,pressure in zip(axes,('ph','pall')):
        for gate in ('unilateral','bilateral'):
            for k in (0.,.1):
                row=next(r for r in rows if r['pressure']==pressure and r['kappa']==k and r['gate']==gate)
                ax.plot([r['step'] for r in row['training_curve']],[r['task_loss'] for r in row['training_curve']],
                    color=colors[gate],linestyle='-' if k==0 else '--',label=f'{gate}, k={k:g}',lw=1)
        ax.set(title=pressure.upper(),xlabel='Optimizer update',ylabel='Global-batch task loss')
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(folder/'03-training-curves.pdf');plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.2,3.6),layout='constrained')
    for ax,site in zip(axes,('a','m')):
        upper=100 if site=='a' else max(.001,1.25*max(r['sites'][site]['tile_8x16_empty_percent'] for r in rows))
        for gate,offset in (('unilateral',-.18),('bilateral',.18)):
            rr=[next(r for r in rows if r['pressure']==p and r['kappa']==k and r['gate']==gate) for p,k in keys]
            ax.bar([i+offset for i in range(4)],[r['sites'][site]['tile_8x16_empty_percent'] for r in rr],
                   width=.34,color=colors[gate],label=gate)
            for i,r in enumerate(rr):
                value=r['sites'][site]['tile_8x16_empty_percent']
                label=f'{value:.4f}' if 0<value<.001 else f'{value:.3g}'
                ax.text(i+offset,value+.025*upper,label,ha='right' if 0<value<.001 else 'center',fontsize=7,color=colors[gate])
        ax.set_xticks(range(4),labels);ax.set(title=f'Site {site}'+(' (zoomed scale)' if site=='m' else ''),ylabel='Completely empty aligned 8 x 16 tiles (%)',ylim=(0,upper))
        ax.tick_params(axis='y',left=True,labelleft=True)
        ax.spines['left'].set_visible(True)
        ax.legend(frameon=False)
    fig.suptitle('Sequence-local token x feature tiles; pooled integer counts',fontsize=11)
    fig.savefig(folder/'04-am-tile-occupancy.pdf');plt.close(fig)

def observation(rows,pairs,sources):
    wins=sum(p['loss_delta_bilateral_minus_unilateral']<0 for p in pairs)
    conclusion=f'Bilateral a/m has lower loss in {wins}/4 matched pairs.'
    for pressure in ('ph','pall'):
        pp=[p for p in pairs if p['pressure']==pressure]
        conclusion+=f" {pressure.upper()}: {sum(p['loss_delta_bilateral_minus_unilateral']<0 for p in pp)}/2 pairs favor bilateral."
    conclusion += ' This supports the proposed direction within this one-seed joint-site experiment.' if wins==4 else ' Support depends on the pressure recipe; the simple claim that bilateral a/m improves quality across both recipes is not supported by this grid.'
    text='# O001: Bilateral versus unilateral after LayerNorm\n\n'
    text+='## Question and method\n\nDoes preserving large negative a/m values improve final quality under matched T4/Ph and T4/Pall pressure? Eight fresh 14M conditions, identical step-zero bytes, data schedule and 712-update recipe, kappa 0/.1; only the a/m sign rule changes within a pair. h/z stay unilateral.\n\n'
    text+='## Coverage and result\n\nAll 500 validation documents, 338 complete 2,048-token blocks, 692,224 input tokens, 691,886 prediction tokens; 1,444-token tail excluded. Pooled integer activation, tile and product counts precede division.\n\n'+conclusion+'\n\n'
    for p in pairs:text+=f"- {p['pressure'].upper()}, kappa={p['kappa']:g}: loss delta {p['loss_delta_bilateral_minus_unilateral']:+.6f}, R_model delta {p['R_model_delta_percentage_points']:+.4f} percentage points.\n"
    ph=[p for p in pairs if p['pressure']=='ph'];pall=[p for p in pairs if p['pressure']=='pall']
    if all(p['loss_delta_bilateral_minus_unilateral']<0 and p['R_model_delta_percentage_points']<0 for p in ph):
        text+='\nUnder Ph, bilateral improves quality at both cutoffs while reducing model-wide logical opportunity. This is a quality/sparsity tradeoff, rather than an improvement on both axes.\n'
    if all(p['loss_delta_bilateral_minus_unilateral']>0 and p['R_model_delta_percentage_points']<0 for p in pall):
        text+='\nUnder Pall, unilateral has both lower loss and higher logical opportunity at each tested cutoff. The bilateral points are dominated on these two measured axes in this cohort.\n'
    text+='\n## Distribution and structural diagnosis\n\n'
    for pressure in ('ph','pall'):
        uu=[r for r in rows if r['pressure']==pressure and r['gate']=='unilateral']
        ee=[r['sites'][s]['removed_large_negative_energy_percent'] for r in uu for s in ('a','m')]
        text+=f"Unilateral {pressure.upper()} removes {min(ee):.2f}--{max(ee):.2f}% of its own pre-gate squared energy from negative a/m values at or above the cutoff magnitude. "
    text+='Bilateral removes none of that large-negative energy, as confirmed by the saved count/energy audit. These are different learned distributions; the removed energy is not a causal estimate of useful information.\n\n'
    for pressure in ('ph','pall'):
        for gate in ('unilateral','bilateral'):
            rr=[r for r in rows if r['pressure']==pressure and r['gate']==gate]
            rms=[r['sites'][s]['pre_rms'] for r in rr for s in ('a','m')]
            neg=[r['sites'][s]['pre_negative_percent'] for r in rr for s in ('a','m')]
            text+=f"- {pressure.upper()}, {gate}: pre-gate a/m RMS {min(rms):.3f}--{max(rms):.3f}; negative mass {min(neg):.2f}--{max(neg):.2f}%.\n"
    text+='\nPh ends with a/m RMS near one and mean near zero, whereas the final distributions differ materially under Pall. Preserving large negative values is insufficient to predict quality under Pall. The RMS/affine and gradient summaries are descriptive diagnostics, not evidence that one specific pressure mechanism caused the loss change.\n\n'
    if all(r['sites'][s]['integer_counts']['tiles']['8x16']['empty']==0 for r in rows if r['pressure']=='ph' for s in ('a','m')):
        text+='All four Ph conditions have zero completely empty 8x16 tiles at a and m, despite high elementwise zero rates in the unilateral conditions.\n\n'
    uu=next(r for r in rows if r['pressure']=='pall' and r['gate']=='unilateral' and r['kappa']==.1)
    bb=next(r for r in rows if r['pressure']=='pall' and r['gate']=='bilateral' and r['kappa']==.1)
    text+=f"At Pall kappa=.1, a empty 8x16 tiles fall from {uu['sites']['a']['tile_8x16_empty_percent']:.4f}% (unilateral) to {bb['sites']['a']['tile_8x16_empty_percent']:.4f}% (bilateral); m falls from {uu['sites']['m']['tile_8x16_empty_percent']:.6f}% to {bb['sites']['m']['tile_8x16_empty_percent']:.6f}%. Tile pressure remains a separate deferred investigation.\n"
    text+='\n## Figures and legend\n\n'
    text+='1. `figures/01-paired-quality.pdf`: final validation loss; panels are pressure recipes; orange unilateral, blue bilateral. Only measured kappa endpoints are connected.\n'
    text+='2. `figures/02-negative-energy.pdf`: squared pre-gate energy removed from negative values with magnitude at least the realized gate cutoff, separately a/m. Each denominator is that condition\'s own pre-gate energy, not a common activation distribution.\n'
    text+='3. `figures/03-training-curves.pdf`: per-update global-batch task loss; solid kappa 0, dashed kappa .1; colors indicate the a/m rule.\n'
    text+='4. `figures/04-am-tile-occupancy.pdf`: percentage of completely empty aligned 8x16 tiles in a/m, within each sequence, without padding. Orange unilateral, blue bilateral. The m panel uses a zoomed percentage scale to show its very rare empty tiles; labels display the actual percentages.\n\n'
    text+='## Caveats and interpretation\n\nOne seed, two thresholds, a/m changed jointly. Equal cutoff does not match zero rates or rejected energy. Learned LayerNorm affine outputs need not stay centered at zero or within [-1,1]. Zero mean also does not imply zero median or exactly 50% negative mass. Gate-mask comparisons use each tensor\'s actual dtype, including the FP16 realization of kappa. Negative-energy rejection differences are descriptive and cannot by themselves prove mediation of the loss change. Pall applies L1 to post-gate a/m/h/z, so preserving negative a/m values also changes the values and gradients entering its pressure calculation; this comparison estimates the complete gate-rule change under the declared recipe. R_model and tile occupancy are logical/structural opportunities, not measured hardware gains. Pressure/gates use elementwise rules; tile pressure remains deferred. No automatic manuscript update.\n\n'
    text+='## Provenance\n\nSource script: `21_analyze.py`; source hashes and exact count exports: `results/results.json`. Complete numerical tables: `results/tables.md`. All final checkpoints and boundary diagnostics were retained and hash-verified locally before compute teardown.\n'
    return text

if __name__=='__main__':
    rows,pairs,sources=collect()
    results=RUN/'results';results.mkdir(exist_ok=True)
    write_json(results/'results.json',{'schema_version':1,'rows':rows,'pairs':pairs,'sources':sources,
        'analysis_script_sha256':sha(Path(__file__)),'estimand':'paired final full-validation loss; one seed; bilateral minus unilateral'})
    (results/'tables.md').write_text(table(rows,pairs),encoding='utf-8')
    figures(rows)
    observations=RUN/'observations';observations.mkdir(exist_ok=True)
    (observations/'001-gate-sign.md').write_text(observation(rows,pairs,sources),encoding='utf-8')
    (observations/'INDEX.md').write_text('# Observations\n\n- [O001: gate-sign comparison](001-gate-sign.md): paired 14M quality, removed negative energy, training trajectories, and final aligned a/m occupancy; source `21_analyze.py`.\n',encoding='utf-8')
    print(json.dumps({'pairs':pairs,'figures':4},indent=2))
