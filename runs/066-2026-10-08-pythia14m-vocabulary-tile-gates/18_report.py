"""Reduce verified Run066 evidence into loss, head sparsity and tile figures."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_config import RUN_DIR, load_config, condition_specs, write_json
from head_site import HEAD_NAME

COLORS = {'control':'#777777','ph':'#0072B2','ph_head':'#D55E00'}
LABELS = {'control':'No head gate / Ph','ph':'Head tile / Ph','ph_head':'Head tile / Ph + head'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def named(rows, name):
    return next(row for row in rows if row['name']==name)


def reduce_head(statistics):
    post=named(statistics['rows'],HEAD_NAME)
    pre=named(statistics['rows'],'head_in_pre.final')
    occupancy=named(statistics['occupancy_rows'],HEAD_NAME)
    energy=next((row for row in statistics['sign_energy_rows'] if row['name']==HEAD_NAME),None)
    result={'head_elements':post['total'],'head_pre_exact_zero_count':pre['exact_zero_count'],
        'head_exact_zero_count':post['exact_zero_count'],
        'head_exact_zero_fraction':post['exact_zero_count']/post['total'],
        'head_near_001_fraction':post['threshold_hits']['0.001']/post['total'],
        'head_near_01_fraction':post['threshold_hits']['0.01']/post['total'],
        'head_pre_rms':math.sqrt(pre['square_sum']/pre['finite']),
        'head_post_rms':math.sqrt(post['square_sum']/post['finite']),
        'head_removed_energy_fraction':(energy['rejected_square_sum']/energy['pre_square_sum']
            if energy['pre_square_sum'] else None) if energy else 0.,
        'head_rows':occupancy['rows'],'head_zero_rows':occupancy['row_nonzero_histogram'][0],
        'head_zero_row_fraction':occupancy['row_nonzero_histogram'][0]/occupancy['rows'],
        'head_native_dtypes':statistics['native_gate_operand_dtypes'][HEAD_NAME]}
    for label,counters in occupancy['tiles'].items():
        for metric in ('empty','total'):result[f'head_{label}_{metric}']=counters[metric]
        result[f'head_{label}_empty_fraction']=counters['empty']/counters['total']
    for site in ('h','z'):
        activation=named(statistics['pooled_by_site'],site)
        tiles=named(statistics['occupancy_pooled_by_site'],site)['tiles']['8x16']
        result[f'{site}_exact_zero_fraction']=activation['exact_zero_count']/activation['total']
        result[f'{site}_8x16_empty_fraction']=tiles['empty']/tiles['total']
    if post['total']!=88604672 or occupancy['rows']!=692224 or result['head_8x16_total']!=692224:
        raise ValueError('Incomplete full-validation head coverage')
    return result


def collect():
    config=load_config()
    verification=read(RUN_DIR/'artifacts/verification.json')
    if verification['status']!='verified' or verification['condition_count']!=4 or len(verification['controls'])!=2:
        raise ValueError('Verified full cohort and both controls required')
    rows=[]
    for source in read(RUN_DIR/'inputs/controls/inventory.json')['controls']:
        root=RUN_DIR/'artifacts/controls'/source['id'];metrics=read(root/'metrics.json')
        rows.append({'id':source['id'],'kind':'control','kappa':source['kappa'],
            'head_gate':False,'pressure_sites':['h'],'loss':metrics['validation']['loss'],
            'source':root.relative_to(RUN_DIR).as_posix(),'validation':metrics['validation'],
            **reduce_head(read(root/'activation_statistics.json'))})
    expected={c['id']:c for c in condition_specs(config)}
    for root in sorted((RUN_DIR/'artifacts/attempts').iterdir()):
        if not root.is_dir():continue
        metrics=read(root/'metrics.json');condition=metrics['condition']
        if condition['id'] not in expected:raise ValueError('Unexpected condition')
        rows.append({'id':condition['id'],'kind':condition['pressure_recipe'],'kappa':condition['gate_threshold'],
            'head_gate':True,'pressure_sites':condition['pressure_sites'],
            'loss':metrics['validation']['step_000712']['loss'],
            'source':root.relative_to(RUN_DIR).as_posix(),
            'validation':metrics['validation']['step_000712'],
            **reduce_head(read(root/'diagnostics/activation_statistics.json'))})
    if len(rows)!=6:raise ValueError('Six-model comparison required')
    rows.sort(key=lambda row:(row['kappa'],['control','ph','ph_head'].index(row['kind'])))
    controls={row['kappa']:row for row in rows if row['kind']=='control'}
    for row in rows:
        baseline=controls[row['kappa']]
        row['loss_delta_vs_control']=row['loss']-baseline['loss']
        row['perplexity']=math.exp(row['loss'])
        row['perplexity_change_percent']=100*math.expm1(row['loss_delta_vs_control'])
        row['head_8x16_delta_pp']=100*(row['head_8x16_empty_fraction']-baseline['head_8x16_empty_fraction'])
    return rows


def observation(path, question, caption, result):
    body=f'''# {question}

## Question

{question}

## Method and coverage

Four Run066 matched 14M trainings and two retained Run060 same-kappa controls,
re-evaluated without further training. Seed 1234, 712 updates, original parameter
bytes and data order. Each reported pass covers all 500 MiniPile validation
documents, 338 complete T=2048 blocks, 692,224 input positions, 691,886 predictions;
1,444 tail tokens excluded. Pool integer counts before fractions. The head port
is after affine final LayerNorm and before dense embed_out. Actual dtypes are
in the reduction JSON. All six models have h/z corner tiles; only four have a
head tile gate. Ph+head averages seven tensor means versus six for Ph.

## Caption and legend

{caption}

Gray denotes no head gate/Ph, blue head tile/Ph, orange head tile/Ph+head.
Circle/solid denotes kappa 0; triangle/dashed or hatched denotes kappa 0.1.
Colors retain this meaning throughout the figures.

## Result

{result}

## Caveats

One seed, one architecture and one training budget. No numerical acceptable-loss
bound was approved. The h coefficient changes from 1/6 to 1/7 in the pressure
variant. Native exact/near zeros, aligned empty tiles and logical opportunities
are distinct; no sparse vocabulary kernel or measured speedup is claimed.
Block-product eager/batch1 counters are separate from production SDPA/batch32
head statistics. No manuscript change or finding promotion is made.

## Sources

See `results/comparison.json` for per-row source directories and full integer
counters. Generated by `18_report.py`; PDF: `{path.name}`.
'''
    (RUN_DIR/'observations'/f'{path.stem}.md').write_text(body,encoding='utf-8')


def main():
    rows=collect()
    for folder in ('results','figures','observations'):(RUN_DIR/folder).mkdir(exist_ok=True)
    write_json(RUN_DIR/'results/comparison.json',{'rows':rows,'unit':'full validation native gate-output elements and sequence-local aligned tiles',
        'loss_unit':'mean causal cross-entropy in nats','speedup_measured':False})
    columns=['id','kappa','head_gate','loss','loss_delta_vs_control','perplexity_change_percent',
        'head_exact_zero_fraction','head_1x16_empty_fraction','head_8x16_empty_fraction',
        'head_16x16_empty_fraction','head_32x16_empty_fraction','head_removed_energy_fraction',
        'head_zero_row_fraction','head_pre_rms','head_post_rms','h_exact_zero_fraction','z_exact_zero_fraction',
        'h_8x16_empty_fraction','z_8x16_empty_fraction']
    with (RUN_DIR/'results/comparison.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for row in rows:
        ax.scatter(100*row['head_8x16_empty_fraction'],row['loss'],s=85,color=COLORS[row['kind']],
            marker='o' if row['kappa']==0 else '^',edgecolor='black',linewidth=.5,
            label=f"{LABELS[row['kind']]}, kappa={row['kappa']:g}")
        ax.annotate(f"{row['loss']:.3f}",(100*row['head_8x16_empty_fraction'],row['loss']),
            xytext=(5,5 if row['kappa']==0 else -12),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Empty head-input 8 x 16 tiles (%)',ylabel='Full validation loss (nats)',
        xlim=(-1,max(100*row['head_8x16_empty_fraction'] for row in rows)+4),
        ylim=(min(row['loss'] for row in rows)-.015,max(row['loss'] for row in rows)+.015),
        title='14M: vocabulary-input tiles and quality')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8);ax.grid(alpha=.2)
    path=RUN_DIR/'figures/01-head-tiles-and-loss.pdf';fig.savefig(path);plt.close(fig)
    result='; '.join(f"{LABELS[row['kind']]} kappa {row['kappa']:g}: loss {row['loss']:.6f}, empty 8x16 {100*row['head_8x16_empty_fraction']:.3f}%" for row in rows)+'.'
    observation(path,'Does structured head sparsity come with a quality penalty?',
        'Each point is one full-validation endpoint. Lower loss and more empty aligned head tiles are separate measured outcomes. The scatter axes focus on the observed endpoint ranges; the loss axis does not start at zero. Sparsity bars in Figure02 retain the full 0-100% range.',result)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=[f"{LABELS[row['kind']]}\nkappa={row['kappa']:g}" for row in rows]
    for i,row in enumerate(rows):
        hatch='' if row['kappa']==0 else '//'
        axes[0].bar(i,100*row['head_exact_zero_fraction'],color=COLORS[row['kind']],hatch=hatch,edgecolor='black',linewidth=.5)
        for j,shape in enumerate(('1x16','8x16','16x16','32x16')):
            axes[1].bar(j+(i-2.5)*.12,100*row[f'head_{shape}_empty_fraction'],width=.115,
                color=COLORS[row['kind']],hatch=hatch,edgecolor='black',linewidth=.4,label=labels[i] if j==0 else None)
    axes[0].set_xticks(range(6),labels,rotation=55,ha='right',fontsize=8)
    axes[1].set_xticks(range(4),['1 x 16','8 x 16','16 x 16','32 x 16'])
    axes[0].set(title='Native head exact zeros',ylabel='Activation elements (%)')
    axes[1].set(title='Aligned head tile occupancy',ylabel='Empty tiles (%)',xlabel='Token x feature tile shape')
    for ax in axes:ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2)
    axes[1].legend(fontsize=7,loc='best')
    path=RUN_DIR/'figures/02-head-zero-elements-and-tiles.pdf';fig.savefig(path);plt.close(fig)
    observation(path,'How much head sparsity forms complete tiles?',
        'Left: native post-gate exact-zero elements. Right: sequence-local fully empty 1x16/8x16/16x16/32x16 tiles. Bars pool integer counts over all 338 validation sequences. Hatch distinguishes kappa 0.1.',result)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for row in rows:
        if not row['head_gate']:continue
        root=RUN_DIR/row['source'];metrics=read(root/'metrics.json')
        validation=metrics['validation'];steps=[0,1,32,128,256,512,712]
        style='-' if row['kappa']==0 else '--';marker='o' if row['kappa']==0 else '^'
        label=f"{LABELS[row['kind']]}, kappa={row['kappa']:g}"
        axes[0].plot(steps,[validation[f'step_{step:06d}']['loss'] for step in steps],style,
            marker=marker,color=COLORS[row['kind']],label=label)
        sparse=[]
        for step in (0,32,128,712):
            path=root/'diagnostics/activation_statistics.json' if step==712 else root/'diagnostics'/f'step_{step:06d}'/'activation_statistics.json'
            sparse.append(100*reduce_head(read(path))['head_8x16_empty_fraction'])
        axes[1].plot([0,32,128,712],sparse,style,marker=marker,color=COLORS[row['kind']],label=label)
    axes[0].set(ylabel='Full validation loss (nats)',ylim=(0,None),title='Loss trajectory')
    axes[1].set(ylabel='Empty head-input 8 x 16 tiles (%)',ylim=(0,100),title='Head tile trajectory')
    for ax in axes:ax.set(xlabel='Optimizer updates',xlim=(0,712));ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    path=RUN_DIR/'figures/03-head-training-trajectories.pdf';fig.savefig(path);plt.close(fig)
    observation(path,'When do head tiles become empty during training?',
        'Left: full validation loss at 0/1/32/128/256/512/712. Right: full head tile diagnostics at 0/32/128/712. Lines connect measured points and do not represent dense sampling. Controls are endpoints only and have no new training trajectory.',result)
    report=['# Run066 — vocabulary-input tiles and pressure','',
        'All six models use existing h/z 8x16 corner tiles. Only the four new conditions add head-input tiles after final affine LayerNorm. The projection weights and logits are dense.','',
        'All 500 validation documents, 338 complete T=2048 blocks, excluded 1,444-token tail; seed 1234. Each new condition trains 712 matched updates. Counts are pooled before fractions.','',
        '| kappa | Head execution | Pressure | Loss | Delta loss vs parent | Head exact zeros | Empty 8x16 head tiles | Removed head energy | Zero head rows |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        energy='N/A' if row['head_removed_energy_fraction'] is None else f"{100*row['head_removed_energy_fraction']:.3f}%"
        report.append(f"| {row['kappa']:g} | {'Tiled gate + dense matmul' if row['head_gate'] else 'Ungated dense matmul'} | {'Ph + head' if row['kind']=='ph_head' else 'Ph'} | {row['loss']:.6f} | {row['loss_delta_vs_control']:+.6f} | {100*row['head_exact_zero_fraction']:.3f}% | {100*row['head_8x16_empty_fraction']:.3f}% | {energy} | {100*row['head_zero_row_fraction']:.3f}% |")
    treatments=[row for row in rows if row['head_gate']]
    if not all(row['loss_delta_vs_control']<0 and row['head_exact_zero_count']==128*row['head_8x16_empty'] for row in treatments):
        raise ValueError('Endpoint evidence does not support the recorded result statements')
    pairs={(row['kappa'],row['kind']):row for row in treatments}
    for kappa in (0.,.1):
        ph,pressured=pairs[kappa,'ph'],pairs[kappa,'ph_head']
        if not (pressured['loss']<ph['loss'] and pressured['head_8x16_empty']>ph['head_8x16_empty']):
            raise ValueError('Endpoint evidence does not support the pressure comparison')
    best=min(treatments,key=lambda row:row['loss'])
    report+=['',
        f"All four new conditions have lower observed loss than their same-kappa parents while producing {100*min(row['head_8x16_empty_fraction'] for row in treatments):.3f}-{100*max(row['head_8x16_empty_fraction'] for row in treatments):.3f}% empty 8x16 head tiles. In these measurements, the native exact-zero element fractions equal the empty 8x16 tile fractions.",
        '',
        f"At both kappa values, Ph+head has lower loss and more empty head tiles than Ph. The lowest-loss endpoint is {LABELS[best['kind']]} at kappa={best['kappa']:g}: loss {best['loss']:.6f}, {100*best['head_8x16_empty_fraction']:.3f}% empty tiles. Its rejected head tiles contain {100*best['head_removed_energy_fraction']:.3f}% of the measured pre-gate squared activation energy. This is evidence of structured sparsity with no observed loss penalty against these parents in this matched test.",
        '',
        'Figures: [quality/tiles](../figures/01-head-tiles-and-loss.pdf), [zeros/occupancy](../figures/02-head-zero-elements-and-tiles.pdf), [trajectories](../figures/03-head-training-trajectories.pdf).','',
        'The JSON/CSV retain all tile shapes, head RMS and h/z spillover. Actual head dtypes, per-row source paths and full validation coverage are in the JSON.','',
        'Ph+head averages seven tensor means; Ph averages six, changing each h coefficient from 1/6 to 1/7. This is part of the intervention, so the comparison does not isolate head pressure from h-pressure reweighting. One seed, architecture and training budget limit generalization. No acceptable-loss bound was declared. Inference matmuls remain dense and latency is unmeasured in this experiment.']
    (RUN_DIR/'results/REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (RUN_DIR/'observations/INDEX.md').write_text('# Run066 observations\n\n'+''.join(f'- [{path.stem}]({path.name})\n' for path in sorted((RUN_DIR/'observations').glob('0*.md'))),encoding='utf-8')
    sources=[]
    for folder in ('results','figures','observations'):
        for path in sorted((RUN_DIR/folder).glob('*')):
            if path.is_file() and path.name!='report-inventory.json':sources.append({'path':path.relative_to(RUN_DIR).as_posix(),'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()})
    write_json(RUN_DIR/'results/report-inventory.json',{'files':sources,'source_script':'18_report.py'})
    print(json.dumps({'rows':len(rows),'figures':3,'source_script':'18_report.py'}))


if __name__=='__main__':main()
