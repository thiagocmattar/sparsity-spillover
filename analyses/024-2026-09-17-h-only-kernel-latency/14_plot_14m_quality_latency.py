"""Requested clean, side-by-side 14M quality/sparsity and full-model latency variant."""
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
TITLE='Sparsity vs. Quality trade-off and Latency on Pythia-14M'
STYLES=[
    ('0','none','Base model','#52565C','none'),
    ('1','none',r'GeLU $\rightarrow$ ReLU','#718052','none'),
    ('4','h',r'$T_4/P_h$','#008B87',(0,(3,2))),
    ('4','all',r'$T_4/P_{\mathrm{all}}$','#3679AD','-'),
    ('7','h',r'$T_7/P_h$','#8A669C',(0,(3,2))),
    ('7','all',r'$T_7/P_{\mathrm{all}}$','#C27539','-'),
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
    # Wide landscape layout; labels remain roughly 7 points at 5.5-inch paper width.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':11.5,
                         'axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
                         'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.85),sharex=True)
    fig.subplots_adjust(left=.085,right=.985,top=.80,bottom=.33,wspace=.34)
    fig.suptitle(TITLE,fontsize=14,y=.985)
    limits={'loss':(5.12,6.19),'latency_ms':(.438,.676)}
    series=[]
    for scope,pressure,label,color,ls in STYLES:
        group=sorted([r for r in rows if (r['scope'],r['pressure'])==(scope,pressure)],
                     key=lambda r:-1 if r['kappa'] is None else r['kappa'])
        assert len(group)==(1 if scope in ['0','1'] else 5)
        if len(group)>1: assert [r['kappa'] for r in group]==[0,.01,.05,.1,.5]
        xs=[r['sparsity'] for r in group]
        control=scope in ['0','1']
        marker_size=9 if control else 5.8
        marker_face='white' if scope=='0' else color
        marker_edge=color if scope=='0' else 'white'
        marker_width=1.5 if scope=='0' else .7
        for ax,metric in zip(axes,['loss','latency_ms']):
            ax.plot(xs,[r[metric] for r in group],color=color,ls=ls,lw=1.6,marker='o',
                    ms=marker_size,mfc=marker_face,mec=marker_edge,mew=marker_width,zorder=5 if control else 4)
        series.append({'label':label.replace(r'$\rightarrow$','->'),'scope':scope,'pressure':pressure,
                       'color':color,'marker':'o','marker_size_pt':marker_size,'marker_face':marker_face,
                       'keys':[r['checkpoint_key'] for r in group]})
    for scope,color in [('0',STYLES[0][3]),('1',STYLES[1][3])]:
        group=sorted([p for p in clips if p['scope']==scope],key=lambda p:p['target'])
        axes[0].plot([p['sparsity'] for p in group],[p['loss'] for p in group],
                     color=color,lw=1.3,ls=':',zorder=2)
    # A single shared note identifies both control clipping paths.
    axes[0].text(2.0,5.62,'Post-hoc',rotation=78,rotation_mode='anchor',
                 color='#646970',fontsize=9.5,ha='left',va='bottom',zorder=5)
    for scope in ['4','7']:
        ceiling=data['coverage']['14M'][scope]['R_model_max_percent']
        axes[0].axvline(ceiling,color='#92969B',lw=.8,ls=(0,(2,3)),alpha=.8,zorder=1)
        axes[0].text(ceiling+.4 if scope=='4' else ceiling-.25,.98,rf'$T_{scope}$ ceiling',
                     transform=axes[0].get_xaxis_transform(),ha='left' if scope=='4' else 'right',va='top',
                     fontsize=9.5,color='#6C7177')
    for ax,metric,title in zip(axes,['loss','latency_ms'],
                               ['(a) Quality-sparsity trade-off','(b) Full-model latency']):
        ax.set(xlim=(-.75,31),ylim=limits[metric],xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
        ax.set_title(title,loc='left',pad=11)
        ax.tick_params(axis='x',labelbottom=True)
        ax.tick_params(length=3,width=.65)
        ax.grid(axis='y',color='#E8EAED',lw=.6,zorder=0); ax.set_axisbelow(True)
        ax.set_xticks([0,10,20,30])
    axes[0].set_ylabel('Validation loss')
    axes[0].set_yticks([5.2,5.4,5.6,5.8,6.0])
    axes[1].set_ylabel('Full-model latency (ms)')
    handles=[Line2D([],[],color=c,ls=ls,marker='o',ms=9 if s in ['0','1'] else 5.8,
                    mfc='white' if s=='0' else c,mec=c if s=='0' else 'white',
                    mew=1.5 if s=='0' else .7,lw=1.6,label=l) for s,_,l,c,ls in STYLES]
    # Separate rows distinguish the controls; columns group T4 and T7 recipes.
    fig.legend(handles=handles,loc='lower center',ncol=3,
               frameon=False,fontsize=11,handlelength=2.3,columnspacing=2.3,
               labelspacing=.7,bbox_to_anchor=(.53,.005))
    fig.savefig(OUTPUT,metadata={'Title':TITLE,
                                'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert hashlib.sha256(original.read_bytes()).hexdigest()==original_sha
    result={'source':SOURCE.relative_to(HERE).as_posix(),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'original_figure_sha256':original_sha,'output':OUTPUT.relative_to(HERE).as_posix(),
            'output_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),'model':'14M',
            'title':TITLE,'layout':{'rows':1,'columns':2,'width_inches':8.6,'height_inches':3.85},
            'typography_pt':{'title':14,'panel_titles':11.5,'axes':11,'ticks':10,'legend':11,'posthoc':9.5,'ceilings':9.5},
            'posthoc_annotation':'One shared Post-hoc note for both control paths.',
            'paper_width_inches':5.5,
            'control_markers':'Base: open gray circle; GeLU -> ReLU: filled olive circle. Exact coordinates, no jitter.',
            'series':series,
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
    print(f'Created {OUTPUT.name}: 22 matched trained checkpoints and one shared post-hoc note.')


if __name__=='__main__':
    main()
