"""Read verified Run058/060 artifacts; compare complete T2/Ph gate operators."""
from pathlib import Path
import csv
import hashlib
import json
import math
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {'elementwise': ROOT/'runs/058-2026-10-03-pythia14m-t2-p2-versus-ph',
        'corner_tile': ROOT/'runs/060-2026-10-04-pythia14m-temporal-corner-tile'}
SITES = ('a','m','h','z')
COLORS = {'elementwise':'#3975a5','corner_tile':'#d87832'}
INITIAL = 'ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57'
SCHEDULE = 'f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa'

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def coverage(row):
    if (row['sequences'],row['input_tokens'],row['excluded_tail_tokens'],row['complete_block_coverage']) != (338,692224,1444,True):
        raise ValueError('Comparison requires complete matched validation')

def statistics(data):
    moments = {r['name']:r for r in data['pooled_by_site']}
    occupancy = {r['name']:r for r in data['occupancy_pooled_by_site']}
    result = {}
    for site in SITES:
        m, o = moments[site], occupancy[site]
        total, zeros = m['total'], m['exact_zero_count']
        if not isinstance(total,int) or not isinstance(zeros,int) or not 0 <= zeros <= total:
            raise ValueError('Invalid integer element counts')
        histogram = o['row_nonzero_histogram']
        if sum(histogram)!=o['rows'] or sum(i*n for i,n in enumerate(histogram))!=total-zeros:
            raise ValueError('Row and exact-zero counts disagree')
        for shape, tile in o['tiles'].items():
            if tile['total']*math.prod(int(v) for v in shape.split('x'))!=total or not 0<=tile['empty']<=tile['total']:
                raise ValueError('Tile counts do not partition validation')
        result[site] = {'elements':total,'zeros':zeros,'zero_percent':100*zeros/total,
            'tiles':o['tiles'],'empty_8x16_percent':100*o['tiles']['8x16']['empty']/o['tiles']['8x16']['total'],
            'rms':m['rms'],'mean':m['mean'],'near_zero_fractions':m['threshold_fractions']}
    for row in data.get('corner_rows',[]):
        if row['certificate_mismatches'] or row['empty'] != row['rejected'] or sum(row['feature_group_rejected'])!=row['rejected']:
            raise ValueError('Corner certificate/counters disagree')
    for row in data.get('corner_pooled_by_site',[]):
        site = result[row['name']]
        tile = site['tiles']['8x16']
        if (row['tiles'],row['empty']) != (tile['total'],tile['empty']):
            raise ValueError('Corner and independent occupancy counts disagree')
        if (sum(row['position_elements']),sum(row['position_zeros'])) != (site['elements'],site['zeros']):
            raise ValueError('Temporal-position and element counts disagree')
    for row in data['sign_energy_rows']:
        if not math.isclose(row['pre_square_sum'],row['post_square_sum']+row['rejected_square_sum'],rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError('Gate energy accounting does not reconcile')
    return result

def collect():
    rows, sources = [], {HERE/'01_analyze.py'}
    configs = {kind:yaml.safe_load((run/'config.yaml').read_text(encoding='utf-8')) for kind,run in RUNS.items()}
    for field in ('training','data','seeds','runtime','recipe','checkpoints'):
        if configs['elementwise'][field] != configs['corner_tile'][field]:
            raise ValueError(f'Unmatched recipe field: {field}')
    for kind, run in RUNS.items():
        verification = load(run/'artifacts/verification.json')
        if verification['status']!='verified': raise ValueError('Verified source run required')
        sources.update([run/'artifacts/verification.json', run/'config.yaml'])
        found = []
        for attempt in sorted((run/'artifacts/attempts').iterdir()):
            if not attempt.is_dir(): continue
            metrics, manifest = load(attempt/'metrics.json'), load(attempt/'manifest.json')
            cond = metrics['condition']
            if cond['pressure_recipe']!='ph': continue
            if manifest['status']!='completed' or manifest['initial_parameter_sha256']!=INITIAL or metrics['training']['schedule_hash']!=SCHEDULE:
                raise ValueError('Initial bytes, schedule or completion differs')
            if (cond['active_sites'],cond['pressure_sites'],cond['pressure_method'],cond['pressure_weight'],cond['step_budget']) != (['h','z'],['h'],'orthogonal_l1',1.,1.):
                raise ValueError('Unmatched topology/pressure')
            expected_operator = 'one_sided_threshold' if kind=='elementwise' else 'tile_corner_threshold'
            if cond['gate_operator']!=expected_operator: raise ValueError('Wrong operator')
            if kind=='corner_tile' and (cond['tile_tokens'],cond['tile_features'])!=(8,16): raise ValueError('Wrong tile')
            for v in metrics['validation'].values(): coverage(v)
            actpath=attempt/'diagnostics/activation_statistics.json'
            act=load(actpath); site_stats=statistics(act)
            logical_path=attempt/'diagnostics/logical_products.json'
            logical=load(logical_path); counts=logical['measured']
            if counts['block_zero_product_count']!=sum(o['zero_product_count'] for o in counts['per_operation'].values()):
                raise ValueError('Logical counts do not pool')
            loss=metrics['validation']['final']['loss']
            events=[json.loads(line) for line in (attempt/'events.jsonl').read_text().splitlines()]
            updates=[e for e in events if e.get('event')=='train']
            if len(updates)!=712 or any(e['optimizer_step_skipped'] for e in updates): raise ValueError('Incomplete updates')
            curve=[{'step':e['step'],'loss':e['loss']} for e in events if e.get('event')=='validation']
            cadence=[]
            if kind=='corner_tile':
                for step in configs[kind]['diagnostics']['steps']:
                    p=actpath if step==712 else attempt/'diagnostics'/f'step_{step:06d}'/'activation_statistics.json'
                    s=load(p); cadence.append({'step':step,'sites':statistics(s),'corners':s['corner_pooled_by_site']})
                    sources.add(p)
            row={'kind':kind,'condition':cond['id'],'kappa':cond['gate_threshold'],
                'loss':loss,'perplexity':math.exp(loss),'sites':site_stats,
                'R_model_percent':100*counts['R_model'],'logical_counts':counts,
                'corners':act.get('corner_pooled_by_site',[]),
                'loss_curve':curve,'diagnostic_curve':cadence,
                'training_curve':[{'step':e['step'],'loss':e['task_loss']} for e in updates],
                'attempt':attempt.relative_to(ROOT).as_posix()}
            rows.append(row);found.append(cond['gate_threshold'])
            sources.update(attempt/p for p in ('manifest.json','metrics.json','config.yaml','events.jsonl',
                'diagnostics/activation_statistics.json','diagnostics/logical_products.json'))
        if sorted(found)!=[0.,.1]: raise ValueError('Missing/duplicate endpoints')
    pairs=[]
    for k in (0.,.1):
        a=next(r for r in rows if r['kind']=='elementwise' and r['kappa']==k)
        b=next(r for r in rows if r['kind']=='corner_tile' and r['kappa']==k)
        pairs.append({'kappa':k,'loss_delta_tile_minus_elementwise':b['loss']-a['loss'],
            'perplexity_relative_change_percent':100*math.expm1(b['loss']-a['loss']),
            'R_model_delta_pp':b['R_model_percent']-a['R_model_percent'],
            'site_delta_pp':{s:{key:b['sites'][s][key]-a['sites'][s][key] for key in ('zero_percent','empty_8x16_percent')} for s in SITES}})
    identity=[]
    for path in sorted(sources):
        with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
        identity.append({'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest})
    return sorted(rows,key=lambda r:(r['kappa'],r['kind']=='corner_tile')),pairs,identity

def tables(rows,pairs):
    text='# T2/Ph: scalar versus causal corner tile gates\n\n'
    text+='Full FP16 validation, 338 complete 2048-token blocks; excluded tail 1444 tokens. Percentages pool integer counts over all six layers. Empty tile means all 128 values are exactly zero.\n\n'
    columns=['Gate','kappa','Loss','Perplexity','R_model (%)']
    for s in SITES: columns += [f'{s.upper()} zeros (%)',f'{s.upper()} empty8x16 (%)']
    text+='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']+['---:']*(len(columns)-1))+' |\n'
    flat=[]
    for r in rows:
        values=[r['kind'],f"{r['kappa']:g}",f"{r['loss']:.6f}",f"{r['perplexity']:.3f}",f"{r['R_model_percent']:.4f}"]
        record={key:r[key] for key in ('kind','kappa','loss','perplexity','R_model_percent')}
        for s in SITES:
            for key in ('zero_percent','empty_8x16_percent'):
                values.append(f"{r['sites'][s][key]:.4f}"); record[f'{s}_{key}']=r['sites'][s][key]
        flat.append(record);text+='| '+' | '.join(values)+' |\n'
    text+='\nDifferences below are corner tile minus elementwise; positive loss change means worse quality.\n\n'
    text+='| kappa | Loss delta | Perplexity change (%) | H empty tiles delta (pp) | Z empty tiles delta (pp) |\n|---:|---:|---:|---:|---:|\n'
    for p in pairs:
        text+=f"| {p['kappa']:g} | {p['loss_delta_tile_minus_elementwise']:+.6f} | {p['perplexity_relative_change_percent']:+.3f} | {p['site_delta_pp']['h']['empty_8x16_percent']:+.4f} | {p['site_delta_pp']['z']['empty_8x16_percent']:+.4f} |\n"
    text+='\nThe tile operator changes which signed values survive and rejects equality. This compares complete operators, not geometry alone. One seed and a fixed 712-update budget; no runtime measurement.\n'
    (HERE/'results/comparison.md').write_text(text,encoding='utf-8')
    with (HERE/'results/comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    layers=[]
    for r in rows:
        act=load(ROOT/r['attempt']/'diagnostics/activation_statistics.json')
        moments={x['name']:x for x in act['rows']}; occ={x['name']:x for x in act['occupancy_rows']}
        for s in SITES:
            for i in range(6):
                name=f'{s}.layer_{i}';m=moments[name];t=occ[name]['tiles']['8x16']
                layers.append({'kind':r['kind'],'kappa':r['kappa'],'site':s,'layer':i,
                    'elements':m['total'],'zeros':m['exact_zero_count'],'zero_percent':100*m['exact_zero_count']/m['total'],
                    'tiles':t['total'],'empty_tiles':t['empty'],'empty_tile_percent':100*t['empty']/t['total']})
    with (HERE/'results/per-layer.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(layers[0]));writer.writeheader();writer.writerows(layers)

def figures(rows):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.25),layout='constrained')
    fig.suptitle('Pythia-14M T2/Ph | 1 seed | 712 updates | 338 validation blocks of 2048 tokens',fontsize=9)
    for ax,field,title in zip(axes,('loss','h','z'),('Validation loss','H: empty 8x16 tiles','Z: empty 8x16 tiles')):
        for j,kind in enumerate(('elementwise','corner_tile')):
            rr=sorted([r for r in rows if r['kind']==kind],key=lambda r:r['kappa'])
            values=[r['loss'] if field=='loss' else r['sites'][field]['empty_8x16_percent'] for r in rr]
            positions=[i+(j-.5)*.34 for i in range(2)]
            ax.bar(positions,values,width=.32,color=COLORS[kind],
                hatch='//' if kind=='corner_tile' else '',label=kind.replace('_',' '))
            for x,v in zip(positions,values):
                label=f'{v:.3f}' if field=='loss' else (f'{v:.2f}' if 0<v<1 else f'{v:.1f}')
                ax.annotate(label,(x,v),
                    xytext=(0,3 if field=='loss' or v<94 else -10),
                    textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([0,1],['0','0.1']);ax.set_xlabel('kappa');ax.set_title(title)
        ax.set_ylabel('NLL' if field=='loss' else 'Empty tiles (%)')
        if field!='loss':ax.set_ylim(0,100)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.94,fontsize=8,loc='lower left')
    fig.savefig(HERE/'figures/01-quality-and-structure.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.25),layout='constrained')
    fig.suptitle('Pythia-14M T2/Ph | 1 seed | Complete validation at each recorded update',fontsize=9)
    for k,color in ((0.,'#3975a5'),(.1,'#d87832')):
        r=next(r for r in rows if r['kind']=='corner_tile' and r['kappa']==k)
        reference=next(r for r in rows if r['kind']=='elementwise' and r['kappa']==k)
        curve=sorted(r['loss_curve'],key=lambda v:v['step'])
        style='-o' if k==0 else '--s'
        axes[0].plot([v['step'] for v in curve],[v['loss'] for v in curve],style,ms=3,color=color,label=f'tile kappa={k:g}')
        axes[0].axhline(reference['loss'],color=color,ls=':',lw=1)
        for ax,s in zip(axes[1:],('h','z')):
            curve=r['diagnostic_curve']
            ax.plot([v['step'] for v in curve],[v['sites'][s]['empty_8x16_percent'] for v in curve],style,ms=3,color=color,label=f'kappa={k:g}')
    for ax,title in zip(axes,('Full-validation learning curve','H: evolving tile emptiness','Z: evolving tile emptiness')):
        ax.set_xlabel('Optimizer update');ax.set_title(title);ax.set_xlim(-8,730);ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('NLL (dotted: scalar final reference)')
    for ax in axes[1:]:ax.set_ylim(0,100);ax.set_ylabel('Empty 8x16 tiles (%)')
    fig.savefig(HERE/'figures/02-learning-and-tile-structure.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(11.6,5),sharey=True,layout='constrained')
    fig.suptitle('Pythia-14M T2/Ph | 1 seed | 712 updates | Counts pooled over 6 layers and 338 validation blocks',fontsize=9)
    for row,k in enumerate((0.,.1)):
        for col,s in enumerate(SITES):
            ax=axes[row,col]
            for j,kind in enumerate(('elementwise','corner_tile')):
                r=next(r for r in rows if r['kind']==kind and r['kappa']==k)
                values=[r['sites'][s]['zero_percent'],r['sites'][s]['empty_8x16_percent']]
                ax.bar([i+(j-.5)*.34 for i in range(2)],values,width=.32,color=COLORS[kind],
                    hatch='//' if kind=='corner_tile' else '',label=kind.replace('_',' '))
            ax.set_title(f'{s.upper()}, kappa={k:g}');ax.set_xticks([0,1],['Exact zeros','Empty 8x16']);ax.set_ylim(0,100)
            selected=[r for r in rows if r['kappa']==k]
            if all(r['sites'][s]['zero_percent']==r['sites'][s]['empty_8x16_percent']==0 for r in selected):
                ax.text(.5,.42,'Measured: 0% exact zeros\n0% empty tiles',transform=ax.transAxes,ha='center',fontsize=9,color='#444444')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axes[row,0].set_ylabel('Percent (%)')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(HERE/'figures/03-site-sparsity-and-tiles.pdf');plt.close(fig)

def main():
    rows,pairs,sources=collect()
    for name in ('results','data','figures'): (HERE/name).mkdir(exist_ok=True)
    write(HERE/'results/summary.json',{'rows':rows,'pairs':pairs,'coverage':{'documents':500,'blocks':338,'input_tokens':692224,'predictions':691886,'excluded_tail':1444},'interpretation':'one-seed complete-operator quality/structure comparison; latency deferred'})
    write(HERE/'data/source-identity.json',{'files':sources,'generator':'01_analyze.py'})
    tables(rows,pairs);figures(rows)
    print(json.dumps({'pairs':pairs,'source_files':len(sources),'figures':3},indent=2))

if __name__=='__main__':main()
