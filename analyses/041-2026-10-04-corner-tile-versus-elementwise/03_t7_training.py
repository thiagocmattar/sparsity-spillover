"""Verified T7/Ph scalar versus corner tiles: canonical quality and occupancy."""
import csv
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TILE=ROOT/'runs/062-2026-10-04-pythia14m-t7-corner-tile-ph'
SCALAR=ROOT/'runs/032-2026-09-16-pythia14m-a7-h-only-ol1'
SITES=('a','m','h','q_post','k_post','v','z')
INITIAL='ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57'
SCHEDULE='f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa'
COLORS={'scalar':'#3975a5','tile':'#8C62A8'}

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def identity(path):
    with path.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':digest}
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def coverage(row):
    assert (row['sequences'],row['input_tokens'],row['excluded_tail_tokens'],row['complete_block_coverage'])==(338,692224,1444,True)
def stats(data):
    moments={r['name']:r for r in data['pooled_by_site']}
    occupancy={r['name']:r for r in data['occupancy_pooled_by_site']}
    result={}
    for site in SITES:
        m,o=moments[site],occupancy[site]; count,zero=m['total'],m['exact_zero_count']; tile=o['tiles']['8x16']
        assert type(count) is int and type(zero) is int and 0<=zero<=count
        assert count==338*2048*6*(512 if site=='h' else 128)
        assert sum(i*n for i,n in enumerate(o['row_nonzero_histogram']))==count-zero
        assert tile['total']*128==count and 0<=tile['empty']<=tile['total']
        result[site]={'elements':count,'zeros':zero,'zero_percent':100*zero/count,
          'empty_tiles':tile['empty'],'total_tiles':tile['total'],'empty_8x16_percent':100*tile['empty']/tile['total'],
          'mean':m['mean'],'rms':m['rms'],'near_zero_fractions':m['threshold_fractions']}
    return result

def collect():
    assert read(TILE/'artifacts/verification.json')['status']=='verified'
    assert read(SCALAR/'artifacts/verification.json')['status']=='verified'
    sources={Path(__file__),TILE/'artifacts/verification.json',SCALAR/'artifacts/verification.json'}
    for row in read(TILE/'transfer/training-inventory-001.json')['files']:
        actual=identity(TILE/row['path']); assert actual['bytes']==row['bytes'] and actual['sha256']==row['sha256']
    sources.add(TILE/'transfer/training-inventory-001.json')
    rows=[]
    for family,run in [('scalar',SCALAR),('tile',TILE)]:
        found=[]
        for attempt in sorted((run/'artifacts/attempts').iterdir()):
            metric_path=attempt/'metrics.json'
            if not metric_path.exists(): continue
            metrics=read(metric_path); cond=metrics['condition']; k=cond['gate_threshold']
            if k not in (0.,.1): continue
            manifest=read(attempt/'manifest.json'); training=metrics['training']
            assert manifest['status']=='completed' and manifest['initial_parameter_sha256']==INITIAL
            assert training['schedule_hash']==SCHEDULE and training['completed_steps']==712
            assert not training['overflow_steps'] and training['optimizer_step_count']==712
            assert cond['active_sites']==list(SITES) and cond['pressure_sites']==['h']
            assert (cond['pressure_method'],cond['pressure_weight'],cond['step_budget'])==('orthogonal_l1',1.,1.)
            coverage(metrics['validation']['final'])
            data_path=(TILE/'artifacts/scalar-structure'/('c36' if k==0 else 'c39')/'activation_statistics.json'
                       if family=='scalar' else attempt/'diagnostics/activation_statistics.json')
            data=read(data_path)
            if family=='scalar':
                assert data['status']=='complete'; coverage(data['coverage'])
                for source in data['source_files']: assert identity(TILE/source['path'])['sha256']==source['sha256']
            if family=='tile':
                assert (cond['tile_tokens'],cond['tile_features'])==(8,16)
                assert all(not r['certificate_mismatches'] and r['empty']==r['rejected'] for r in data['corner_rows'])
            logical_path=attempt/'diagnostics/logical_products.json'; logical=read(logical_path)['measured']
            assert logical['block_zero_product_count']==sum(x['zero_product_count'] for x in logical['per_operation'].values())
            events=[json.loads(line) for line in (attempt/'events.jsonl').read_text().splitlines()]
            updates=[e for e in events if e.get('event')=='train']; assert len(updates)==712
            loss=metrics['validation']['final']['loss']
            ports=data['occupancy_rows']
            empty_ranges={}
            for site in SITES:
                values=[100*r['tiles']['8x16']['empty']/r['tiles']['8x16']['total']
                        for r in ports if r['name'].split('.')[0]==site]
                assert len(values)==6
                empty_ranges[site]={'min':min(values),'max':max(values)}
            cadence=[]
            if family=='tile':
                for step in [0,32,128,712]:
                    path=data_path if step==712 else attempt/'diagnostics'/f'step_{step:06d}'/'activation_statistics.json'
                    cadence.append({'step':step,'sites':stats(read(path))}); sources.add(path)
            rows.append({'family':family,'condition':cond['id'],'kappa':k,'loss':loss,'perplexity':math.exp(loss),
                'sites':stats(data),'R_model_percent':100*logical['R_model'],'logical_counts':logical,
                'fully_empty_ports':[r['name'] for r in ports if r['tiles']['8x16']['empty']==r['tiles']['8x16']['total']],
                'per_layer_empty_8x16_range_percent':empty_ranges,
                'loss_curve':[{'step':e['step'],'loss':e['loss']} for e in events if e.get('event')=='validation'],
                'diagnostic_curve':cadence,'attempt':attempt.relative_to(ROOT).as_posix(),
                'final_checkpoint_content_sha256':manifest['checkpoints']['final']['content_sha256'],
                'structure_replay_loss':data.get('coverage',{}).get('loss')})
            sources.update([metric_path,attempt/'manifest.json',attempt/'events.jsonl',data_path,logical_path]); found.append(k)
        assert sorted(found)==[0.,.1], (family,found)
    pairs=[]
    for k in [0.,.1]:
        scalar=next(r for r in rows if r['family']=='scalar' and r['kappa']==k)
        tile=next(r for r in rows if r['family']=='tile' and r['kappa']==k)
        delta=tile['loss']-scalar['loss']
        pairs.append({'kappa':k,'loss_delta_tile_minus_scalar':delta,'perplexity_change_percent':100*math.expm1(delta),
            'site_deltas_pp':{s:{v:tile['sites'][s][v]-scalar['sites'][s][v] for v in ['zero_percent','empty_8x16_percent']} for s in SITES}})
    return {'rows':sorted(rows,key=lambda r:(r['kappa'],r['family'])),'pairs':pairs,
      'coverage':{'documents':500,'blocks':338,'input_tokens':692224,'predictions':691886,'excluded_tail':1444},
      'sources':[identity(p) for p in sorted(sources)],'seed':1234,'updates':712,
      'scope':'Canonical native FP16 autocast with FP32 parameters. Integer counts pooled over all six layers; QKV tiles remain headwise.',
      'limits':'One seed, fixed budget, complete-operator comparison. Corner equality and signed interior preservation also change. No causal claim about representation redistribution.'}

def tables(data):
    lines=['# 14M T7/Ph: scalar versus 8x16 corner tiles','',data['scope'],'',
      '| Gate | kappa | FP16 loss | Perplexity | R_model (%) | '+' | '.join(f'{s} zeros (%) | {s} empty tiles (%)' for s in SITES)+' |',
      '|---|'+ '|'.join(['---:']*(4+2*len(SITES)))+'|']
    flat=[]
    for r in data['rows']:
        record={k:r[k] for k in ['family','kappa','loss','perplexity','R_model_percent']}
        fields=[r['family'],f"{r['kappa']:g}",f"{r['loss']:.6f}",f"{r['perplexity']:.3f}",f"{r['R_model_percent']:.3f}"]
        for s in SITES:
            for key in ['zero_percent','empty_8x16_percent']:
                value=r['sites'][s][key]; fields.append(f'{value:.3f}'); record[s+'_'+key]=value
        flat.append(record); lines.append('| '+' | '.join(fields)+' |')
    lines+=['','| kappa | Tile minus scalar loss | Perplexity change (%) |','|---:|---:|---:|']
    lines += [f"| {p['kappa']:g} | {p['loss_delta_tile_minus_scalar']:+.6f} | {p['perplexity_change_percent']:+.3f} |" for p in data['pairs']]
    lines += ['',data['limits'],'','R_model is a logical zero-product opportunity, not a speedup. All500 validation documents; excluded1444-token tail.']
    (HERE/'results/t7-comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    with (HERE/'results/t7-comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)

def figures(data):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,7,figsize=(14.8,5.2),sharey=True,layout='constrained')
    fig.suptitle('14M T7/Ph | 1 seed | 712 updates | Six-layer pooled counts over 338 validation blocks',fontsize=11)
    for row,k in enumerate([0.,.1]):
        for column,s in enumerate(SITES):
            ax=axes[row,column]
            for j,family in enumerate(['scalar','tile']):
                r=next(r for r in data['rows'] if r['family']==family and r['kappa']==k)
                ax.bar([i+(j-.5)*.34 for i in range(2)],[r['sites'][s][v] for v in ['zero_percent','empty_8x16_percent']],
                       width=.32,color=COLORS[family],hatch='//' if family=='tile' else '',label=family)
            ax.set(title=f'{s}, kappa={k:g}',ylim=(0,100),xticks=[0,1],xticklabels=['Zeros','Empty\n8x16'])
            ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
        axes[row,0].set_ylabel('Percent (%)')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(HERE/'figures/05-t7-site-sparsity-and-tiles.pdf',metadata={'CreationDate':None,'ModDate':None}); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.9),layout='constrained')
    fig.suptitle('14M T7/Ph corner gates | Full validation at each recorded update | 1 seed',fontsize=11)
    for k,color,style in [(0.,'#3975a5','-o'),(.1,'#8C62A8','--s')]:
        r=next(r for r in data['rows'] if r['family']=='tile' and r['kappa']==k)
        scalar=next(r for r in data['rows'] if r['family']=='scalar' and r['kappa']==k)
        curve=sorted(r['loss_curve'],key=lambda x:x['step'])
        axes[0].plot([v['step'] for v in curve],[v['loss'] for v in curve],style,color=color,ms=3,label=f'tile kappa={k:g}')
        axes[0].axhline(scalar['loss'],color=color,ls=':',lw=1)
        for j,site in enumerate(['h','z']):
            curve=r['diagnostic_curve']
            axes[j+1].plot([v['step'] for v in curve],[v['sites'][site]['empty_8x16_percent'] for v in curve],style,color=color,ms=3,label=f'kappa={k:g}')
    for ax,title in zip(axes,['Validation learning','H: empty tiles','Z: empty tiles']):
        ax.set(xlabel='Optimizer update',title=title,xlim=(-8,730)); ax.grid(alpha=.15); ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('NLL (dotted: scalar final loss)')
    for ax in axes[1:]: ax.set(ylim=(0,100),ylabel='Empty 8x16 tiles (%)')
    fig.savefig(HERE/'figures/06-t7-learning-and-tile-structure.pdf',metadata={'CreationDate':None,'ModDate':None}); plt.close(fig)

def main():
    data=collect(); tables(data); figures(data)
    data['figures']=[identity(HERE/'figures'/n) for n in ['05-t7-site-sparsity-and-tiles.pdf','06-t7-learning-and-tile-structure.pdf']]
    write(HERE/'results/t7-training.json',data)
    print(json.dumps({'status':'verified','pairs':data['pairs']},indent=2))

if __name__=='__main__': main()
