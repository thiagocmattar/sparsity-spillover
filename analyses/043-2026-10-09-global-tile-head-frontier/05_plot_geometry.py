"""Pressure geometry: exact normalizations plus the manuscript's two-panel view."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from evidence import HERE

COLORS={'Ph':'#0072B2','Ph+head':'#D55E00'}
FAMILY_COLORS={'T4 / Ph':'#008B87','T4 / Pall':'#3679AD','T7 / Ph':'#8A669C','T7 / Pall':'#C27539',
               'T2 tile / Ph':'#0072B2','Head8 / Ph':'#56B4E9','Head8 / Ph+head':'#CC79A7','Head16 / Ph+head':'#D55E00'}


def markers(head):
    return {0:'o',8:'D',16:'*'}[head]


def dots(axis,rows,key,title,xlabel,log=False,reference=None):
    for i,row in enumerate(rows):
        values=row['summary'][key]
        med=values['median']
        axis.errorbar(med,i,xerr=[[med-values['p25']],[values['p75']-med]],
            fmt=markers(row['head_tile']),color=COLORS[row['pressure_recipe']],capsize=3,
            ms=8 if row['head_tile']==16 else 5,elinewidth=1.0)
    axis.set(title=title,xlabel=xlabel)
    if log:axis.set_xscale('log')
    else:axis.set_xlim(left=0)
    if reference is not None:axis.axvline(reference,color='#777777',ls=':',lw=1)
    axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)


def overview(data):
    rows=data['conditions'];y=np.arange(len(rows))
    fig,axes=plt.subplots(2,3,figsize=(16,9.3),sharey=True)
    fig.subplots_adjust(left=.17,right=.985,top=.84,bottom=.13,wspace=.24,hspace=.28)
    specs=[('rho_task_percent','(a) Manuscript: removed component','Removed norm / task norm (%; log scale)',True),
           ('removed_pressure_percent','(b) Fraction of pressure removed','Removed component norm / pressure norm (%)',False),
           ('pre_projection_pressure_norm','(d) Pressure norm before projection','Preconditioned pressure L2 norm (log scale)',True),
           ('pre_projection_norm_ratio','(e) Pressure/task norm before projection','||w|| / ||u|| (log scale)',True),
           ('pre_cap_ratio','(f) Manuscript: after projection, before cap','r / b (log scale)',True)]
    for axis,(key,title,label,log) in zip([axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[1,2]],specs):
        dots(axis,rows,key,title,label,log=log,reference=1 if key=='pre_cap_ratio' else None)
        if log:
            ticks={'rho_task_percent':[2,4,8,16], 'pre_projection_pressure_norm':[100,200,400],
                   'pre_projection_norm_ratio':[.1,.2,.4,.8], 'pre_cap_ratio':[.1,.3,1]}[key]
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
            axis.xaxis.set_minor_formatter(NullFormatter())
    axis=axes[0,2]
    colors=[COLORS[r['pressure_recipe']] for r in rows]
    axis.barh(y-.16,[r['summary']['raw_gradient_conflict']['percent'] for r in rows],height=.29,
              color=colors,hatch='///',edgecolor='white',linewidth=.3,alpha=.65,label='Raw gradients')
    axis.barh(y+.16,[r['summary']['adaptive_conflict']['percent'] for r in rows],height=.29,
              color=colors,edgecolor='white',linewidth=.3,label='Adaptive directions')
    axis.set(xlim=(0,105),title='(c) Conflict frequency',xlabel='Conflicting optimizer boundaries (%)')
    axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    for axis in axes.flat:
        axis.set_yticks(y,[r['label'] for r in rows]);axis.set_ylim(len(rows)-.5,-.5)
        axis.tick_params(axis='y',labelsize=8)
        for split in (1.5,3.5,5.5):axis.axhline(split,color='#DDDDDD',lw=.7)
    fig.suptitle('Orthogonalization resistance: h-only versus h + vocabulary-head pressure',fontsize=15,y=.97)
    fig.text(.5,.91,'Global eligible-parameter geometry · 712 boundaries/condition · medians and interquartile ranges\nPh: 6 tensors weighted 1/6; Ph+head: 7 tensors weighted 1/7. The head contribution is not isolated.',ha='center',fontsize=10)
    legend=[Line2D([],[],color=c,marker='o',ls='none',label=p) for p,c in COLORS.items()]
    legend += [Line2D([],[],color='#555555',marker=m,ls='none',label=l,ms=8) for m,l in [('o','Head threshold off'),('D','Head8 tile'),('*','Head16 tile')]]
    legend += [Patch(facecolor='#888888',hatch='///',alpha=.65,label='Raw gradients'),
               Patch(facecolor='#888888',label='Adaptive directions')]
    fig.legend(handles=legend,loc='lower center',ncol=7,frameon=False,bbox_to_anchor=(.5,.036),fontsize=9)
    fig.text(.5,.012,'Adaptive projection is frequent in every condition; the norm cap is active in only 1–3 of 712 boundaries here.\nPercent removed refers to a norm, not a count of gradient coordinates or pressure energy.',ha='center',fontsize=9)
    fig.savefig(HERE/'figures/05-orthogonalization-head-comparison.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def manuscript_style(data):
    rows=data['manuscript_context']+data['conditions'];families=list(FAMILY_COLORS)
    fig,axes=plt.subplots(1,2,figsize=(14.5,6.4));fig.subplots_adjust(left=.075,right=.985,top=.80,bottom=.26,wspace=.28)
    threshold_positions={0.:0,.01:1,.05:2,.1:3,.5:4}
    for j,family in enumerate(families):
        members=sorted([r for r in rows if r['family']==family],key=lambda r:r['kappa'])
        color=FAMILY_COLORS[family];context=family.startswith(('T4','T7'))
        values=[r['summary']['rho_task_percent'] for r in members]
        med=np.array([v['median'] for v in values]);low=np.array([v['p25'] for v in values]);high=np.array([v['p75'] for v in values])
        assert np.all(low>0)
        x=np.array([threshold_positions[r['kappa']] for r in members])+(j-3.5)*.025
        axes[0].errorbar(x,med,yerr=[med-low,high-med],color=color,
            marker='o' if context else markers(members[0]['head_tile']),ms=4 if context else 7,
            ls='--' if context else '-',lw=.8 if context else 1.3,capsize=2,alpha=.6 if context else 1)
        traces=np.array([r['trace']['pre_cap_ratio'] for r in members]);steps=np.arange(1,713)
        for trace in traces:axes[1].plot(steps,trace,color=color,lw=.4,alpha=.09)
        axes[1].plot(steps,np.median(traces,axis=0),color=color,lw=.9 if context else 1.5,
                     ls='--' if context else '-',alpha=.6 if context else 1,label=family)
    axes[0].set(yscale='log',xlim=(-.4,4.5),xlabel='Swept threshold κ (categorical positions)',
                ylabel='Removed component / task-direction norm (%)',title='(a) Opposing component: median and IQR')
    axes[0].set_xticks(range(5),['0','0.01','0.05','0.1','0.5'])
    axes[1].set(yscale='log',xlim=(1,712),xlabel='Optimizer boundary',ylabel='Post-projection, pre-cap ratio r / b',
                title='(b) Budget regime: pointwise median over thresholds')
    axes[1].axhline(1,color='#555555',ls=':',lw=1);axes[1].text(700,1.08,'norm cap',ha='right',fontsize=8,color='#555555')
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Vocabulary-head pressure in the manuscript’s OL1 geometry view',fontsize=14,y=.97)
    fig.text(.5,.90,'Dashed: 20 original manuscript 14M conditions · solid: 8 retained tile/head conditions\nHead16 sweeps head κ only at h/z κ=0; Head8 sweeps both h/z and head κ. No head-only pressure.',ha='center',fontsize=10)
    handles=[Line2D([],[],color=FAMILY_COLORS[f],ls='--' if f.startswith(('T4','T7')) else '-',
                     label=f+' (5 κ)' if f.startswith(('T4','T7')) else f+' (2 κ)',lw=1.5) for f in families]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.04),fontsize=9)
    fig.text(.5,.012,'IQRs describe training-step variation; bold trajectories are unsmoothed pointwise medians. One seed per condition.',ha='center',fontsize=9)
    fig.savefig(HERE/'figures/06-manuscript-style-geometry.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    data=json.loads((HERE/'data/geometry.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    overview(data);manuscript_style(data)
    print({'status':'geometry-plotted','figures':2},flush=True)


if __name__=='__main__':
    main()
