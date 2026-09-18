"""Requested clean, two-row 14M quality/sparsity and full-model latency variant."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'data/paper-checkpoints.json'
OUTPUT=HERE/'figures/06-14m-quality-sparsity-latency.pdf'
STYLES=[
    ('0','none','Base model','#52565C','none'),
    ('1','none',r'GeLU $\rightarrow$ ReLU','#718052','none'),
    ('4','h','4-Thresh-h-Pressure','#008B87',(0,(3,2))),
    ('4','all','4-Thresh-all-Pressure','#3679AD','-'),
    ('7','h','7-Thresh-h-Pressure','#8A669C',(0,(3,2))),
    ('7','all','7-Thresh-all-Pressure','#C27539','-'),
]


def main():
    data=json.loads(SOURCE.read_text(encoding='utf-8'))
    original=HERE/'figures/01-quality-sparsity-tradeoffs.pdf'
    original_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    rows=[r for r in data['checkpoints'] if r['model']=='14M' and r['comparison_cohort']
          and (r['scope'] in ['0','1'] or r['pressure'] in ['h','all'])]
    assert len(rows)==22 and len({r['checkpoint_key'] for r in rows})==22
    clips=[p for p in data['clipping'] if p['model']=='14M' and p['main_clipping']]
    assert len(clips)==20
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
                         'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,1,figsize=(6.8,7.1),sharex=True)
    fig.subplots_adjust(left=.13,right=.965,top=.905,bottom=.18,hspace=.38)
    limits={'loss':(5.12,6.19),'latency_ms':(.438,.676)}
    series=[]
    for scope,pressure,label,color,ls in STYLES:
        group=sorted([r for r in rows if (r['scope'],r['pressure'])==(scope,pressure)],
                     key=lambda r:-1 if r['kappa'] is None else r['kappa'])
        assert len(group)==(1 if scope in ['0','1'] else 5)
        if len(group)>1: assert [r['kappa'] for r in group]==[0,.01,.05,.1,.5]
        xs=[r['sparsity'] for r in group]
        for ax,metric in zip(axes,['loss','latency_ms']):
            ax.plot(xs,[r[metric] for r in group],color=color,ls=ls,lw=1.15,marker='o',
                    ms=3.8,mfc=color,mec='white',mew=.35,zorder=4)
        series.append({'label':label.replace(r'$\rightarrow$','->'),'scope':scope,'pressure':pressure,
                       'color':color,'marker':'o','keys':[r['checkpoint_key'] for r in group]})
    for scope,color in [('0',STYLES[0][3]),('1',STYLES[1][3])]:
        group=sorted([p for p in clips if p['scope']==scope],key=lambda p:p['target'])
        axes[0].plot([p['sparsity'] for p in group],[p['loss'] for p in group],
                     color=color,lw=1,ls=':',zorder=2)
    # Direct, understated labels replace a separate post-hoc legend entry.
    base_anchor=next(p for p in clips if p['scope']=='0' and p['target']==.4)
    relu_anchor=next(p for p in clips if p['scope']=='1' and p['target']==.5)
    axes[0].annotate('Base model\npost-hoc',(base_anchor['sparsity'],base_anchor['loss']),xytext=(.6,5.98),
                     color=STYLES[0][3],fontsize=7.8,
                     arrowprops={'arrowstyle':'-','color':STYLES[0][3],'lw':.55},zorder=5)
    axes[0].annotate('GeLU '+r'$\rightarrow$'+' ReLU\npost-hoc',(relu_anchor['sparsity'],relu_anchor['loss']),xytext=(7.7,6.03),
                     color=STYLES[1][3],fontsize=7.8,
                     arrowprops={'arrowstyle':'-','color':STYLES[1][3],'lw':.55},zorder=5)
    for scope in ['4','7']:
        ceiling=data['coverage']['14M'][scope]['R_model_max_percent']
        axes[0].axvline(ceiling,color='#92969B',lw=.65,ls=(0,(2,3)),alpha=.8,zorder=1)
        axes[0].text(ceiling+.4 if scope=='4' else ceiling-.25,.96,f'{scope}-Threshold ceiling',
                     transform=axes[0].get_xaxis_transform(),ha='left' if scope=='4' else 'right',va='top',
                     fontsize=7.5,color='#6C7177')
    for ax,metric,title in zip(axes,['loss','latency_ms'],
                               ['(a) Quality-sparsity trade-off','(b) Full-model latency']):
        ax.set(xlim=(-.75,31),ylim=limits[metric],xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
        ax.set_title(title,loc='left',pad=10)
        ax.tick_params(axis='x',labelbottom=True)
        ax.tick_params(length=3,width=.65)
        ax.grid(axis='y',color='#E8EAED',lw=.55,zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel('Validation loss (nats/token)')
    axes[1].set_ylabel('Full-model latency (ms)')
    fig.text(.965,.955,'14M',ha='right',fontsize=10,color='#53575D')
    handles=[Line2D([],[],color=c,ls=ls,marker='o',ms=4,mfc=c,mec='white',mew=.35,lw=1.15,label=l)
             for _,_,l,c,ls in STYLES]
    # Row order: controls, 4-Threshold recipes, 7-Threshold recipes.
    fig.legend(handles=[handles[i] for i in [0,2,4,1,3,5]],loc='lower center',ncol=2,
               frameon=False,fontsize=8.5,handlelength=2.8,columnspacing=2.8,
               labelspacing=.8,bbox_to_anchor=(.55,.016))
    fig.savefig(OUTPUT,metadata={'Title':'14M quality, sparsity, and full-model latency',
                                'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
    result={'source':SOURCE.relative_to(HERE).as_posix(),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_figure_sha256':original_sha,'output':OUTPUT.relative_to(HERE).as_posix(),
            'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'model':'14M','series':series,
            'trained_points':[{k:r[k] for k in ['checkpoint_key','scope','pressure','kappa','sparsity','loss','latency_ms','timing_session']}
                              for r in rows],
            'panel_a_checkpoint_keys':[r['checkpoint_key'] for r in rows],
            'panel_b_checkpoint_keys':[r['checkpoint_key'] for r in rows],
            'panel_a_clipping_ids':[p['id'] for p in clips],
            'clipping_outside_y':[p['id'] for p in clips if not limits['loss'][0]<=p['loss']<=limits['loss'][1]],
            'posthoc_latency':'Not measured; clipping points are shown only in panel (a).',
            'limits':limits,'ceilings':data['coverage']['14M']}
    assert result['panel_a_checkpoint_keys']==result['panel_b_checkpoint_keys']
    assert all(limits['loss'][0]<=r['loss']<=limits['loss'][1] for r in rows)
    assert all(limits['latency_ms'][0]<=r['latency_ms']<=limits['latency_ms'][1] for r in rows)
    (HERE/'data/14m-quality-sparsity-latency.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(f'Created {OUTPUT.name}: 22 matched trained checkpoints, two annotated post-hoc paths, no nondominated outlines.')


if __name__=='__main__':
    main()
