"""Vector figures from verified records; no scientific measurements."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analyses/018-2026-09-08-results-materials'))
from plots import STYLE, configure


def csv_read(name):
    rows=list(csv.DictReader((HERE/'data'/name).open(encoding='utf-8')))
    numeric={'loss','S_model','S_block','delta_loss_vs_A0','trained_dose','clipping_p','R_arch_actual','R_arch_A7',
             'input_tokens','task_loss','adamw_gradient_norm_pre_clip','learning_rate','step'}
    return [{k:float(v) if k in numeric and v else v for k,v in r.items()} for r in rows]


def save(fig,name):
    fig.savefig(HERE/'figures'/name,metadata={'Creator':'Analysis 020; retained evidence', 'CreationDate':None,'ModDate':None})
    plt.close(fig)
    shutil.copyfile(HERE/'figures'/name,ROOT/'manuscript/draft/figures'/name)


def cross_size(rows,full=False):
    chosen=[r for r in rows if (r['kind']=='trained') or (r['kind']=='clipped' and r['family'] in {'A0','A1-H'})]
    assert len(chosen)==96
    fig,axes=plt.subplots(2,3,figsize=(5.5,4.55 if full else 4.35),sharey='row')
    fig.subplots_adjust(left=.10,right=.99,bottom=.23,top=.91,wspace=.14,hspace=.37)
    omitted=[]
    for j,scale in enumerate(['14M','70M','410M']):
        group=[r for r in chosen if r['scale']==scale]
        for i,(x,y) in enumerate([('S_model','loss'),('S_block','delta_loss_vs_A0')]):
            ax=axes[i,j]
            for family in ['A0','A1-H','A4-OL1','A7-OL1']:
                clip=family in {'A0','A1-H'}
                series=sorted([r for r in group if r['family']==family and r['kind']==('clipped' if clip else 'trained')],
                              key=lambda r:r['clipping_p'] if clip else r['trained_dose'])
                color,marker=STYLE[family]
                ax.plot([100*r[x] for r in series],[r[y] for r in series],color=color,marker=marker,
                        ms=3.5,lw=.8,ls=':' if clip else '-',mfc='white' if clip else color,mew=.6)
                if not clip:
                    end=series[-1]
                    ax.scatter([100*end[x]],[end[y]],s=55,facecolors='none',edgecolors=color,lw=.8,zorder=5)
                else:
                    base=next(r for r in group if r['family']==family and r['kind']=='trained')
                    ax.scatter([100*base[x]],[base[y]],s=26,c=color,marker=marker,zorder=6)
            if i==0:
                for family,label in [('A4-OL1','A4'),('A7-OL1','A7')]:
                    reach=next(r['R_arch_actual'] for r in group if r['family']==family and r['kind']=='trained')
                    ax.axvline(100*reach,color=STYLE[family][0],lw=.55,ls='--',alpha=.65)
                    ax.text(100*reach-.4,9.32 if full else 6.92,label,rotation=90,ha='right',va='top',fontsize=7,color=STYLE[family][0])
                ax.set_title(scale,fontsize=10)
                ax.set_xlim(-.02*100*group[0]['R_arch_A7'],1.04*100*group[0]['R_arch_A7'])
                ax.set_ylim(3.9,9.5 if full else 7)
                ax.set_xticks([0,10,20,30] if scale=='14M' else [0,20,40] if scale=='70M' else [0,40,80])
            else:
                ax.axhline(0,color='.7',lw=.6)
                ax.set(xlim=(-2,104),ylim=(-.15,5.25 if full else 2.0),xticks=[0,50,100])
            if not full:
                hidden=[r['id'] for r in group if r[y]>ax.get_ylim()[1]]
                omitted.append(dict(scale=scale,row=i,ids=hidden))
                if hidden:ax.text(.03,.97,f'{len(hidden)} clip points above',transform=ax.transAxes,fontsize=6.8,va='top',bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=.5))
            ax.grid(color='.92',lw=.4)
            ax.tick_params(labelsize=7.5)
    axes[0,0].set_ylabel('Validation loss',fontsize=8)
    axes[1,0].set_ylabel(r'Loss $-$ same-size A0',fontsize=8)
    axes[0,1].set_xlabel(r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',fontsize=8)
    axes[1,1].set_xlabel(r'Block-only sparsity $\mathcal{S}_{\mathrm{block}}$ (%)',fontsize=8)
    handles=[]
    for f in ['A4-OL1','A7-OL1','A0','A1-H']:
        c,m=STYLE[f];clip=f in {'A0','A1-H'}
        handles.append(Line2D([],[],color=c,marker=m,mfc='white' if clip else c,ls=':' if clip else '-',ms=4,
                              label=f+' + clipping' if clip else f))
    handles.extend([Line2D([],[],color='.4',marker='o',ls='none',ms=4,label='Filled: trained endpoint'),
                    Line2D([],[],color='.4',marker='o',mfc='none',ls='none',ms=7,label=r'Ring: $\kappa=0.5$')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.005),ncol=2,frameon=False,fontsize=7.5,columnspacing=2.)
    fig.suptitle('Quality-sparsity trade-offs across the Pythia family',fontsize=10.5,y=.99)
    name='09-cross-size-full.pdf' if full else '03-v4-cross-size-interventions.pdf'
    save(fig,name)
    return dict(evaluations=96,trained=36,clipped=60,omitted=omitted)


def dynamics(rows):
    fig,axes=plt.subplots(3,3,figsize=(5.5,5.9),sharex=True,sharey='row')
    fig.subplots_adjust(left=.12,right=.99,bottom=.13,top=.88,wspace=.12,hspace=.18)
    colors=['#444444','#6f4c9b','#009E73']
    for j,family in enumerate(['A0','A4-OL1','A7-OL1']):
        for scale,color in zip(['14M','70M','410M'],colors):
            group=[r for r in rows if r['family']==family and r['scale']==scale]
            assert len(group)==712
            x=[r['input_tokens']/1e9 for r in group]
            for i,key in enumerate(['task_loss','adamw_gradient_norm_pre_clip','learning_rate']):
                axes[i,j].plot(x,[r[key] for r in group],color=color,lw=.55,alpha=.85)
        axes[0,j].set_title(family+('' if family=='A0' else r', $\kappa=.5$'),fontsize=9)
        for i in range(3):
            axes[i,j].axvspan(570*2097152/1e9,712*2097152/1e9,color='.92',zorder=-2)
            axes[i,j].grid(color='.94',lw=.4);axes[i,j].tick_params(labelsize=7)
        axes[1,j].set_yscale('log')
        axes[2,j].ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
        axes[2,j].set_xticks([0,.5,1,1.5])
    for i,label in enumerate(['Task loss','Pre-clip task norm','Learning rate']):axes[i,0].set_ylabel(label,fontsize=8)
    axes[2,1].set_xlabel('Input tokens (billions)',fontsize=8)
    fig.legend(handles=[Line2D([],[],color=c,label=s) for s,c in zip(['14M','70M','410M'],colors)],
               loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.54,.005))
    fig.suptitle('Retained training trajectories (raw optimizer-boundary records)',fontsize=10,y=.97)
    save(fig,'10-training-dynamics.pdf')


def runtime():
    data=json.loads((ROOT/'analyses/019-2026-09-09-manuscript-rewrite-audit/runtime-audit.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(5.5,3.55))
    fig.subplots_adjust(left=.11,right=.99,bottom=.31,top=.83,wspace=.34)
    a,b=axes
    for r in data['rows']:
        c,m=STYLE[r['family']]
        off,on=[r['candidate_latency_ms'][k] for k in ['k050-no-skip','k050']]
        y=r['BF16_validation_loss']
        a.plot([off,on],[y,y],color=c,lw=.65,alpha=.65)
        a.scatter([off],[y],s=16,marker=m,edgecolors=c,facecolors='white',linewidth=.65,zorder=3)
        a.scatter([on],[y],s=19,marker=m,color=c,linewidth=.6,zorder=4)
        b.scatter([100*r['canonical_FP16_S_model']],[r['sparse_path_factor']],c=c,marker=m,s=23,linewidth=.5)
    for condition,label,xytext in [('c01','A0',(.653,5.32)),('c15',r'A4, $\kappa=.5$',(.50,5.57)),('c25',r'A7, $\kappa=.5$',(.50,5.84))]:
        r=next(r for r in data['rows'] if r['condition']==condition)
        a.annotate(label,xy=(r['candidate_latency_ms']['k050'],r['BF16_validation_loss']),xytext=xytext,
                   fontsize=7,ha='left' if condition!='c01' else 'right',arrowprops=dict(arrowstyle='-',lw=.5,color='.4'))
    fit=data['association_sensitivity']['all']['sparse_path'];x=np.array([0,.30])
    b.plot(100*x,fit['intercept']+fit['slope']*x,color='.5',ls='--',lw=.7)
    selected=next(r for r in data['rows'] if r['condition']=='c30')
    b.scatter([100*selected['canonical_FP16_S_model']],[selected['sparse_path_factor']],s=80,facecolors='none',edgecolors='black',lw=.7)
    b.axhline(1,color='.6',ls=':',lw=.7)
    b.text(.03,.96,r'Descriptive OLS $R^2=0.504$',transform=b.transAxes,fontsize=7,va='top')
    a.set(xlim=(.44,.74),ylim=(5.05,6.16),xlabel='Full-model latency (ms)',ylabel='BF16 reference loss',title='(a) Quality and measured latency')
    b.set(xlim=(-1,31),ylim=(.86,1.49),xlabel=r'FP16 $\mathcal{S}_{\mathrm{model}}$ (%)',ylabel='Sparse-path speed factor',title='(b) Increment from sparse paths')
    for ax in axes:ax.grid(axis='y',color='.93',lw=.4);ax.tick_params(labelsize=7.5);ax.title.set_fontsize(8.5)
    handles=[Line2D([],[],color=c,marker=m,ls='none',ms=4,label=f) for f,(c,m) in STYLE.items()]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=7.5,bbox_to_anchor=(.53,.02))
    fig.text(.12,.18,'Open: all skips off; filled: K050. Each pair shares a reference loss.',fontsize=7)
    fig.suptitle('Quality, latency and the conditional benefit of sparse execution',fontsize=10,y=.98)
    save(fig,'07-v2-kernel-quality-latency.pdf')


def main():
    (HERE/'figures').mkdir(exist_ok=True)
    configure()
    rows=csv_read('cross_size_interventions.csv')
    checks={'cross_size_main':cross_size(rows),'cross_size_full':cross_size(rows,full=True)}
    dynamics(csv_read('training-curves.csv'))
    runtime()
    paths=[HERE/'data/cross_size_interventions.csv',HERE/'data/training-curves.csv',
           ROOT/'analyses/019-2026-09-09-manuscript-rewrite-audit/runtime-audit.json',
           ROOT/'analyses/018-2026-09-08-results-materials/plots.py',Path(__file__)]
    checks['sources']={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (HERE/'figures/SOURCES.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(checks['cross_size_main'],indent=2))


if __name__=='__main__':main()
