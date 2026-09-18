"""Figure 6 extended to both sizes, with the measured Run036 clipping latencies."""
import hashlib
import json
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'data/paper-checkpoints.json'
CLIPPING=ROOT/'runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json'
STYLE_SOURCE=HERE/'14_plot_14m_quality_latency.py'
STYLES=import_module(STYLE_SOURCE.stem).STYLES
OUTPUT=HERE/'figures/08-14m-70m-quality-sparsity-latency.pdf'
TITLE='Sparsity vs. Quality trade-off and Latency on Pythia 14M and 70M'
LIMITS={
    '14M':{'sparsity':[-.75,31],'loss':[5.12,6.19],'latency_ms':[.435,.825]},
    '70M':{'sparsity':[-1.25,51],'loss':[4.0,5.58],'latency_ms':[1.48,3.73]},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joined_points(data,timings):
    """Join retained clipping quality/counts to qualified timings by identity and dose."""
    assert timings['status']=='complete_verified_reduction'
    trained=[r for r in data['checkpoints'] if r['model'] in LIMITS and r['comparison_cohort']
             and (r['scope'] in ['0','1'] or r['pressure'] in ['h','all'])]
    assert len(trained)==44 and len({r['checkpoint_key'] for r in trained})==44
    timing_index={(p['checkpoint_key'],p['p']):p for p in timings['points']}
    assert len(timing_index)==len(timings['points'])==40
    trained_index={r['checkpoint_key']:r for r in trained}
    clips=[]
    for point in data['clipping']:
        if not point['main_clipping'] or point['model'] not in LIMITS:
            continue
        timing=timing_index[point['checkpoint_key'],point['target']]
        original=trained_index[point['checkpoint_key']]
        assert timing['qualified'] and all(r['qualified'] for r in timing['replicates'])
        assert timing['scale']==point['model']==original['model']
        assert timing['family']=={'0':'A0','1':'A1-H'}[point['scope']]
        assert point['loss']==timing['retained_fp16_loss']
        assert point['counts']==timing['retained_fp16_counts']
        assert abs(point['sparsity']-timing['sparsity_percent'])<1e-10
        assert timing['kernel'].lower()==original['kernel'].lower()
        clips.append({k:point[k] for k in ['id','checkpoint_key','model','scope','target','loss','sparsity']}|{
            'latency_ms':timing['candidate_gm_ms'],'condition':timing['condition'],
            'kernel':timing['kernel'],'qualified':True,'timing_session':'Run036',
            'timing_device_uuid':timings['device_uuid'],
            'zero_product_count':point['counts']['block_zero_product_count'],
            'model_product_count':point['counts']['model_product_count'],
            'coverage':point['coverage']})
    assert len(clips)==40 and len({p['id'] for p in clips})==40
    return trained,clips


def main():
    preserved={p.name:sha(p) for p in (HERE/'figures').glob('*.pdf') if p!=OUTPUT}
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    timing_data=json.loads(CLIPPING.read_text(encoding='utf-8'))
    trained,clips=joined_points(data,timing_data)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11.5,
        'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,2,figsize=(10.8,6.8),sharex='row')
    fig.subplots_adjust(left=.08,right=.985,top=.87,bottom=.19,hspace=.55,wspace=.28)
    fig.suptitle(TITLE,fontsize=14,y=.985)
    series=[]
    panels=[]
    for i,size in enumerate(['14M','70M']):
        rows=[r for r in trained if r['model']==size]
        clipping=[p for p in clips if p['model']==size]
        assert len(rows)==22 and len(clipping)==20
        for scope,pressure,label,color,ls in STYLES:
            group=sorted([r for r in rows if (r['scope'],r['pressure'])==(scope,pressure)],
                         key=lambda r:-1 if r['kappa'] is None else r['kappa'])
            assert len(group)==(1 if scope in ['0','1'] else 5)
            if scope in ['4','7']: assert [r['kappa'] for r in group]==[0,.01,.05,.1,.5]
            for ax,metric in zip(axes[i],['loss','latency_ms']):
                ax.plot([r['sparsity'] for r in group],[r[metric] for r in group],
                        color=color,ls=ls,lw=1.6,marker='o',ms=9 if scope in ['0','1'] else 5.8,
                        mfc='white' if scope=='0' else color,mec=color if scope=='0' else 'white',
                        mew=1.5 if scope=='0' else .7,zorder=5 if scope in ['0','1'] else 4)
            series.append({'model':size,'scope':scope,'pressure':pressure,'label':label,
                           'color':color,'linestyle':ls,'keys':[r['checkpoint_key'] for r in group]})
        for scope,color in [('0',STYLES[0][3]),('1',STYLES[1][3])]:
            group=sorted([p for p in clipping if p['scope']==scope],key=lambda p:p['target'])
            assert [p['target'] for p in group]==[p/10 for p in range(10)]
            for ax,metric in zip(axes[i],['loss','latency_ms']):
                ax.plot([p['sparsity'] for p in group],[p[metric] for p in group],
                        color=color,lw=1.3,ls=':',zorder=2)
        # One subtle note identifies both control paths in each panel.
        qx,qy,angle=(2.,5.62,78) if size=='14M' else (10.5,4.72,68)
        axes[i,0].text(qx,qy,'Post-hoc',rotation=angle,rotation_mode='anchor',
                       color='#646970',fontsize=9.5,ha='left',va='bottom')
        axes[i,1].text(4.5 if size=='14M' else 10.5,.803 if size=='14M' else 3.61,
                       'Post-hoc',color='#646970',fontsize=9.5,ha='left',va='center')
        for scope in ['4','7']:
            ceiling=data['coverage'][size][scope]['R_model_max_percent']
            axes[i,0].axvline(ceiling,color='#92969B',lw=.8,ls=(0,(2,3)),alpha=.8,zorder=1)
            label_on_right=scope=='4' and size=='14M'
            axes[i,0].text(ceiling+.4 if label_on_right else ceiling-.25,.98,rf'$T_{scope}$ ceiling',
                           transform=axes[i,0].get_xaxis_transform(),
                           ha='left' if label_on_right else 'right',va='top',fontsize=9.5,color='#6C7177')
        for j,(metric,label,title) in enumerate([
                ('loss','Validation loss','quality-sparsity trade-off'),
                ('latency_ms','Full-model latency (ms)','full-model latency')]):
            ax=axes[i,j]
            ax.set(xlim=LIMITS[size]['sparsity'],ylim=LIMITS[size][metric],ylabel=label,
                   xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
            ax.set_title(f'({chr(97+2*i+j)}) {size}: {title}',loc='left',pad=11)
            ax.set_xticks([0,10,20,30] if size=='14M' else [0,10,20,30,40,50])
            ax.tick_params(length=3,width=.65,labelbottom=True)
            ax.grid(axis='y',color='#E8EAED',lw=.6,zorder=0); ax.set_axisbelow(True)
            lo,hi=LIMITS[size][metric]
            assert all(lo<=r[metric]<=hi for r in rows)
            outside=[p['id'] for p in clipping if not lo<=p[metric]<=hi]
            if metric=='latency_ms': assert not outside
            panels.append({'panel':chr(97+2*i+j),'model':size,'metric':metric,
                           'trained_keys':[r['checkpoint_key'] for r in rows],
                           'clipping_ids':[p['id'] for p in clipping],
                           'clipping_outside_y':outside})
        axes[i,0].set_yticks([5.2,5.4,5.6,5.8,6.0] if size=='14M' else [4.0,4.4,4.8,5.2])
    handles=[Line2D([],[],color=c,ls=ls,marker='o',ms=9 if s in ['0','1'] else 5.8,
                    mfc='white' if s=='0' else c,mec=c if s=='0' else 'white',
                    mew=1.5 if s=='0' else .7,lw=1.6,label=l) for s,_,l,c,ls in STYLES]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=11,
               handlelength=2.3,columnspacing=2.3,labelspacing=.7,bbox_to_anchor=(.53,.005))
    fig.savefig(OUTPUT,metadata={'Title':TITLE,'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert preserved=={p.name:sha(p) for p in (HERE/'figures').glob('*.pdf') if p!=OUTPUT}
    result={'title':TITLE,'output':OUTPUT.relative_to(HERE).as_posix(),'output_sha256':sha(OUTPUT),
        'script':Path(__file__).name,'script_sha256':sha(Path(__file__)),
        'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [SOURCE,CLIPPING,STYLE_SOURCE]},
        'style_reference':'figures/06-14m-quality-sparsity-latency.pdf',
        'latency_reference':'figures/07-controls-posthoc-final-latency.pdf',
        'layout':{'rows':2,'columns':2,'width_inches':10.8,'height_inches':6.8},
        'typography_pt':{'title':14,'panel_titles':11.5,'axes':11,'ticks':10,'legend':11,'notes':9.5},
        'limits':LIMITS,'series':series,'panels':panels,
        'trained_points':[{k:r[k] for k in ['checkpoint_key','model','scope','pressure','kappa',
                           'sparsity','loss','latency_ms','kernel','timing_session','timing_device_uuid']}
                          for r in trained],
        'clipping_points':clips,'ceilings':{size:data['coverage'][size] for size in LIMITS},
        'preserved_pdf_sha256':preserved,
        'clipping_latency_note':'Run036 full-model candidate geometric means including recurring clipping; no shift to historical trained control timings.',
        'quality_view_note':'Focused trained-model quality range, as in Figure 6; all clipping records retained, high-loss tails outside the view recorded per panel.',
        'precision_note':timing_data['precision_note'],'kernel_note':timing_data['kernel_note']}
    (HERE/'data/14m-70m-quality-sparsity-latency.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pdf':str(OUTPUT),'trained':44,'clipping':40,
                      'quality_outside':{p['model']:len(p['clipping_outside_y']) for p in panels if p['metric']=='loss'}}))


if __name__=='__main__':
    main()
