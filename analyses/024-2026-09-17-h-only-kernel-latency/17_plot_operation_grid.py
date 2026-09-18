"""Absolute operation contributions at three thresholds and two model sizes."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'data/paper-checkpoints.json'
REFERENCE=HERE/'data/paper-derived.json'
OUTPUT=HERE/'figures/09-14m-70m-operation-contributions.pdf'
TITLE='Operation Contributions to Model-wide Sparsity on Pythia 14M and 70M'
KAPPAS=[0,.05,.5]
Y_LIMITS={'14M':[0,30],'70M':[0,45]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    reference=next(f for f in json.loads(REFERENCE.read_text(encoding='utf-8'))['figures']
                   if f['file']=='04-operation-sparsity-changes.pdf')
    palette=reference['palette']
    preserved={p.name:sha(p) for p in (HERE/'figures').glob('*.pdf') if p!=OUTPUT}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11.5,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,3,figsize=(12.8,6.4),sharey='row')
    fig.subplots_adjust(left=.065,right=.985,top=.86,bottom=.17,hspace=.60,wspace=.20)
    fig.suptitle(TITLE,fontsize=14,y=.985)
    pressure_labels={'h':'h','all':r'\mathrm{all}'}
    panels=[]; records=[]
    for i,size in enumerate(['14M','70M']):
        recipes=[(s,p) for s in ['4','7'] for p in ['h','all']]
        positions=[0,1,2.6,3.6]
        labels=[rf'$T_{s}/P_{{{pressure_labels[p]}}}$' for s,p in recipes]
        for j,k in enumerate(KAPPAS):
            ax=axes[i,j]
            available=[r for r in data['checkpoints'] if r['comparison_cohort']
                       and r['model']==size and r['kappa']==k and r['scope'] in ['4','7']
                       and r['pressure'] in ['h','all']]
            assert len(available)==len(recipes)
            assert {(r['scope'],r['pressure']) for r in available}==set(recipes)
            keys=[]
            for x,(scope,pressure) in zip(positions,recipes):
                row=next(r for r in available if (r['scope'],r['pressure'])==(scope,pressure))
                den=row['counts']['model_product_count']; bottom=0.; components={}
                for operation in palette:
                    op=operation['operation']
                    count=row['counts']['per_operation'][op]['zero_product_count']
                    value=100*count/den
                    components[op]={'zero_product_count':count,'model_denominator':den,'contribution_pp':value}
                    ax.bar(x,value,bottom=bottom,color=operation['color'],width=.72,
                           edgecolor='white',lw=.4,zorder=3)
                    bottom+=value
                assert sum(c['zero_product_count'] for c in components.values())==row['counts']['block_zero_product_count']
                assert abs(bottom-row['sparsity'])<1e-12
                assert Y_LIMITS[size][0]<=bottom<=Y_LIMITS[size][1]
                keys.append(row['checkpoint_key'])
                records.append({'model':size,'kappa':k,'scope':scope,'pressure':pressure,
                    'checkpoint_key':row['checkpoint_key'],'components':components,'total_pp':bottom})
            ax.set_xticks(positions,labels)
            ax.set_xlim(-.65,positions[-1]+.65)
            ax.set_title(f'({chr(97+3*i+j)}) {size}: '+rf'Threshold $\kappa={k:g}$',loc='left',pad=11)
            ax.set_xlabel('Threshold / pressure recipe',labelpad=9)
            ax.tick_params(length=3,width=.65)
            ax.grid(axis='y',color='#E8EAED',lw=.6,zorder=0)
            ax.set_axisbelow(True)
            panels.append({'panel':chr(97+3*i+j),'model':size,'kappa':k,
                           'checkpoint_keys':keys,'recipe_labels':labels})
        axes[i,0].set_ylabel(r'Contribution to $S_{\mathrm{model}}$ (pp)')
        axes[i,0].set_ylim(Y_LIMITS[size])
        axes[i,0].set_yticks([0,10,20,30] if size=='14M' else [0,10,20,30,40])
    handles=[plt.Rectangle((0,0),1,1,color=p['color'],label=p['label']) for p in palette]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=11,
               handlelength=1.2,columnspacing=1.5,bbox_to_anchor=(.53,.018))
    fig.savefig(OUTPUT,metadata={'Title':TITLE,'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert len(records)==24 and len({r['checkpoint_key'] for r in records})==24
    assert preserved=={p.name:sha(p) for p in (HERE/'figures').glob('*.pdf') if p!=OUTPUT}
    result={'title':TITLE,'output':OUTPUT.relative_to(HERE).as_posix(),'output_sha256':sha(OUTPUT),
        'script':Path(__file__).name,'script_sha256':sha(Path(__file__)),
        'sources_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in [SOURCE,REFERENCE]},
        'reference_figure':'figures/04-operation-sparsity-changes.pdf',
        'layout':{'rows':2,'columns':3,'width_inches':12.8,'height_inches':6.4,
                  'y_limits_by_model':Y_LIMITS,'y_scale_sharing':'within each row'},
        'typography_pt':reference['typography_pt'],'palette':palette,
        'estimand':reference['estimand'],'panels':panels,'checkpoints':records,
        'recipe_scope':'Ph and Pall only for both topologies, model sizes and all three thresholds.',
        'preserved_pdf_sha256':preserved}
    (HERE/'data/14m-70m-operation-contributions.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Created {OUTPUT.name}: 12 retained checkpoints per model size, Ph/Pall only, with separate row scales.')


if __name__=='__main__':
    main()
