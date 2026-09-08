"""Paper-width figures for Analysis 018; numeric detail lives in the tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from evidence import HERE, FAMILIES, OPS, SCALES, one

# Consistent symbols across figures, distinguishable without color.
STYLE = {'A0': ('#444444', 'o'), 'A1-H': ('#888888', 's'),
         'A1-H-L1': ('#009E73', 'v'), 'A1-H-OL1': ('#CC79A7', '^'),
         'A4': ('#0072B2', 'D'), 'A4-OL1': ('#0072B2', 'P'),
         'A7': ('#D55E00', 'X'), 'A7-OL1': ('#D55E00', '*')}
OP_COLORS = ('#b3cde3', '#6497c5', '#275e91', '#8ac3a3', '#f4b66c', '#c96d4c')
OP_LABELS = ('QKV projections', 'FFN up', 'FFN down', 'Attention output', 'QK scores', 'PV product')
S_LABEL = r'$\mathcal{S}_{\mathrm{model}}$ (%)'
U_LABEL = r'Ceiling utilization $U_{\mathrm{arch}}$ (%)'


def configure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.axisbelow': True,
                         'axes.linewidth': .6, 'lines.linewidth': 1.0})


def save(fig, name):
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures'/name,
                metadata={'Creator': 'Analysis 018', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def handles(families):
    return [Line2D([], [], color=STYLE[f][0], marker=STYLE[f][1], linestyle='none',
                   markersize=6, label=f) for f in families]


def curve(ax, rows, family, x='R_model', y='loss', clipping=False):
    rows = sorted(rows, key=lambda r: r['dose'] if r['dose'] is not None else -1)
    c, marker = STYLE[family]
    ax.plot([100*r[x] for r in rows], [r[y] for r in rows], color=c,
            marker=marker, ms=3.5, lw=1., ls=':' if clipping else '-',
            mfc='white' if clipping else c, mew=.7,
            label=family+' + clipping' if clipping else family)


def overview(d):
    rows = [r for r in d['trained'] if r['scale']=='14M']
    fig, ax = plt.subplots(figsize=(5.5,3.7))
    fig.subplots_adjust(left=.13, right=.98, bottom=.16, top=.80)
    for f in FAMILIES:
        rr=[r for r in rows if r['family']==f]; c,m=STYLE[f]
        ax.scatter([100*r['R_model'] for r in rr], [r['loss'] for r in rr],
                   color=c, marker=m, s=33, linewidths=.5, zorder=3)
    ax.set(xlim=(-.5,29), ylim=(5.04,6.15), xlabel=S_LABEL, ylabel='Validation loss (nats/token)')
    ax.set_yticks(np.arange(5.2,6.2,.2)); ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(handles=handles(FAMILIES),loc='upper center',ncol=4,frameon=False,
               columnspacing=1.4,handletextpad=.4,bbox_to_anchor=(.53,.98))
    save(fig,'01-14m-overview.pdf')


def effects(d):
    blocks=['GELU to ReLU','Add L1 at h','L1 to OL1 at h','A1-H to A4',
            'Add OL1 to A4','A4 to A7 gates','Add OL1 to A7']
    labels=['GELU\n→ ReLU','Add L1\nat h','L1 → OL1\nat h','Gate\na, m, z',
            'Add OL1\nto A4','Gate\nq, k, v','Add OL1\nto A7']
    colors=['#444444','#009E73','#CC79A7','#0072B2','#0072B2','#D55E00','#D55E00']
    fig,axs=plt.subplots(2,1,figsize=(5.5,4.3),sharex=True)
    fig.subplots_adjust(left=.15,right=.99,bottom=.20,top=.85,hspace=.12)
    dose_markers={0.:'o',.01:'s',.05:'^',.1:'D',.5:'P',1.:'*'}
    for n,(block,color) in enumerate(zip(blocks,colors)):
        rows=sorted([r for r in d['contrasts'] if r['block']==block],key=lambda r:r['dose'] or 0)
        xx=n+np.linspace(-.30,.30,len(rows)) if len(rows)>1 else np.array([n])
        for ax,key in zip(axs,['delta_loss','delta_R_pp']):
            if n%2==0: ax.axvspan(n-.48,n+.48,color='.96',zorder=-2)
            ax.vlines(xx,0,[r[key] for r in rows],color=color,lw=.8)
            for x,r in zip(xx,rows):
                ax.scatter(x,r[key],color=color,marker=dose_markers.get(r['dose'],'o'),s=20,zorder=3)
    for ax in axs:
        ax.axhline(0,color='.25',lw=.7); ax.grid(axis='y',color='.92',lw=.6)
        ax.tick_params(axis='x',length=0);ax.set_xlim(-.48,6.48)
    axs[0].set(ylabel='Δ validation loss',ylim=(-.20,.42),yticks=[-.2,0,.2,.4])
    axs[1].set(ylabel='Δ '+S_LABEL.replace('(%)','(pp)'),ylim=(-.7,13),yticks=[0,4,8,12])
    axs[1].set_xticks(range(7),labels)
    dose_handles=[Line2D([],[],color='.3',marker=m,ls='none',ms=4,label=f'{v:g}') for v,m in dose_markers.items()]
    fig.legend(handles=dose_handles,loc='upper center',ncol=6,frameon=False,
               title='Dose (λ for local pressure; κ for A4/A7)',title_fontsize=8,
               handletextpad=.3,columnspacing=1.2,bbox_to_anchor=(.56,1.0))
    save(fig,'02-blocked-intervention-effects.pdf')


def scaling(d):
    fig,axs=plt.subplots(2,3,figsize=(5.5,4.7),sharey='row')
    fig.subplots_adjust(left=.12,right=.96,bottom=.17,top=.84,wspace=.14,hspace=.44)
    for j,scale in enumerate(SCALES):
        rows=[r for r in d['trained'] if r['scale']==scale]
        clips=[r for r in d['clipping'] if r['scale']==scale and r['control'] is not None]
        for i,(x,y) in enumerate((('R_model','loss'),('U_arch','delta_loss_vs_A0'))):
            ax=axs[i,j]
            for f in ('A4-OL1','A7-OL1'):
                curve(ax,[r for r in rows if r['family']==f],f,x=x,y=y)
            for f in ('A0','A1-H'):
                curve(ax,[r for r in clips if r['control']==f],f,x=x,y=y,clipping=True)
            ax.grid(axis='y',color='.92',lw=.6)
            ax.set_xlim(-3,103);ax.set_xticks([0,50,100])
            if i==1: ax.axhline(0,color='.6',lw=.7)
        axs[0,j].set_title(scale)
    axs[0,0].set_ylabel('Validation loss\n(nats/token)')
    axs[1,0].set_ylabel('Δ loss vs A0\n(nats/token)')
    axs[0,0].set_ylim(3.8,9.6);axs[1,0].set_ylim(-.35,5.4)
    axs[0,1].set_xlabel(S_LABEL);axs[1,1].set_xlabel(U_LABEL)
    h,l=axs[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,1.),
               handlelength=2.2,columnspacing=1.8)
    save(fig,'03-scale-transfer-and-ceilings.pdf')


def operations(d):
    fig,ax=plt.subplots(figsize=(5.5,3.6))
    fig.subplots_adjust(left=.13,right=.99,bottom=.23,top=.79)
    rows=[one(d['trained'],scale=s,family=f,dose=.5) for s in SCALES for f in ('A4-OL1','A7-OL1')]
    xx=np.array([0,1,3,4,6,7]);base=np.zeros(6)
    for op,c,label,hatch in zip(OPS,OP_COLORS,OP_LABELS,('', '/', '\\', '.', 'xx', '--')):
        values=np.array([100*r['counts']['per_operation'][op]['zero_product_count']/r['counts']['model_product_count'] for r in rows])
        ax.bar(xx,values,bottom=base,color=c,width=.68,label=label,hatch=hatch,
               edgecolor='white',linewidth=.35);base+=values
    ax.set(ylim=(0,100),ylabel='Contribution to '+S_LABEL.replace('(%)','(pp)'))
    ax.set_xticks(xx,['A4','A7']*3)
    for center,scale in zip((.5,3.5,6.5),SCALES):
        ax.text(center,-.22,scale,ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
    ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
               columnspacing=1.0,handlelength=1.5,bbox_to_anchor=(.55,1.))
    save(fig,'04-operation-accounting.pdf')


def activations(d):
    sites=('h','m','q_post','k_post','v');families=('A0','A4-OL1','A7-OL1')
    fig,axs=plt.subplots(2,2,figsize=(5.5,4.4),sharex=True,sharey='row')
    fig.subplots_adjust(left=.14,right=.99,bottom=.14,top=.83,wspace=.15,hspace=.17)
    for j,k in enumerate((0.,.5)):
        for n,f in enumerate(families):
            r=one(d['trained'],scale='14M',family=f,dose=None if f=='A0' else k)
            zero=[];small=[]
            for site in sites:
                a=r['sites'][site]
                zero.append(100*a['exact_zero_count']/a['total'])
                small.append(100*(a['threshold_hits']['0.01']-a['exact_zero_count'])/a['total'])
            for ax,values in zip(axs[:,j],(zero,small)):
                c,m=STYLE[f];xx=np.arange(5)+(n-1)*.22
                ax.vlines(xx,0,values,color=c,lw=2.5,alpha=.7)
                ax.scatter(xx,values,color=c,marker=m,s=25,zorder=3)
        axs[0,j].set_title(r'$\kappa = '+f'{k:g}'+r'$')
        axs[0,j].set_ylim(-3,105);axs[0,j].set_yticks([0,50,100])
        axs[1,j].set_ylim(-3,105);axs[1,j].set_yticks([0,50,100])
        for ax in axs[:,j]:
            ax.set_xticks(range(5),[r'$h$',r'$m$',r'$q$',r'$k$',r'$v$'])
            ax.grid(axis='y',color='.92',lw=.6)
    axs[0,0].set_ylabel('Exact-zero mass (%)')
    axs[1,0].set_ylabel('Small nonzero mass (%)')
    fig.legend(handles=handles(families),loc='upper center',ncol=3,frameon=False,
               bbox_to_anchor=(.56,.99))
    fig.supxlabel('Activation site',y=.015,fontsize=9)
    save(fig,'05-activation-mass-grid.pdf')


def all_clipping(d):
    clip=[r for r in d['clipping'] if r['scale']=='14M']
    def family(r):
        name=r['family']
        if name=='gelu-control': return 'A0'
        if name=='relu-control': return 'A1-H'
        if name.startswith('relu-l1n'): return 'A1-H-L1'
        if name.startswith('relu-ol1'): return 'A1-H-OL1'
        assert name.startswith('a4z-')
        return 'A4'
    fig,ax=plt.subplots(figsize=(5.5,3.7))
    fig.subplots_adjust(left=.13,right=.98,bottom=.16,top=.80)
    families=('A0','A1-H','A1-H-L1','A1-H-OL1','A4')
    for f in families:
        rr=[r for r in clip if family(r)==f];c,m=STYLE[f]
        ax.scatter([100*r['R_model'] for r in rr],[r['loss'] for r in rr],
                   edgecolors=c,facecolors='none',marker=m,s=23,linewidths=.8)
    ax.set(xlim=(-.3,13),ylim=(5.,9.6),xlabel=S_LABEL,ylabel='Validation loss (nats/token)')
    ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(handles=handles(families),loc='upper center',ncol=3,frameon=False,
               title='Source checkpoint family (uniform clipping)',title_fontsize=8,
               bbox_to_anchor=(.55,1.),handletextpad=.4,columnspacing=1.1)
    save(fig,'06-complete-posthoc-comparison.pdf')


def kernels(d):
    runtime=d['runtime'];fig,axs=plt.subplots(1,2,figsize=(5.5,2.9))
    fig.subplots_adjust(left=.11,right=.97,bottom=.21,top=.84,wspace=.34)
    progress=runtime['progress']
    axs[0].step([p['iteration'] for p in progress],[p['speedup'] for p in progress],where='post',color='#333333')
    axs[0].set(xlabel='Kernel iteration',ylabel='Speedup (×)',title='(a) Qualified incumbent',ylim=(.75,1.90))
    rows=[p for p in runtime['points'] if p['candidate']=='k050']
    for f in FAMILIES:
        rr=[r for r in rows if r['family']==f];c,m=STYLE[f]
        axs[1].scatter([100*r['R_model'] for r in rr],[r['speedup'] for r in rr],color=c,marker=m,s=22)
    fit=runtime['k050_regression'];xx=np.array([0,.30])
    axs[1].plot(100*xx,fit['intercept']+fit['slope_per_fraction']*xx,color='.55',ls='--',lw=.8,zorder=0)
    axs[1].set(xlabel=S_LABEL,title='(b) Final kernel',ylim=(.75,1.90),xlim=(-1,31))
    for ax in axs:
        ax.axhline(1,color='.65',ls=':',lw=.8);ax.grid(axis='y',color='.92',lw=.6)
    # Recipe symbols follow Figure 01; caption makes this cross-reference explicit.
    save(fig,'07-kernel-realization.pdf')


def ceilings(d):
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    fig.subplots_adjust(left=.14,right=.96,bottom=.20,top=.83)
    for f,ls in zip(('A0','A1-H','A4','A7'),(':','--','-.','-')):
        rr=[one(d['ceilings'],scale=s,family=f) for s in SCALES];c,m=STYLE[f]
        ax.plot([r['parameters'] for r in rr],[r['R_model_max_percent'] for r in rr],
                color=c,marker=m,ls=ls,ms=5,label=f)
    ax.set_xscale('log');ax.set_xticks([one(d['exposure'],scale=s)['parameters'] for s in SCALES],SCALES)
    ax.minorticks_off();ax.set(ylim=(-3,100),xlabel='Model size (parameters)',
                              ylabel=r'Analytic ceiling $\mathcal{S}_{\mathrm{model}}^{\max}$ (%)')
    ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,
               bbox_to_anchor=(.55,1.),handlelength=2.3)
    save(fig,'08-ceiling-vs-model-size.pdf')


def make_figures(d):
    configure()
    overview(d);effects(d);scaling(d);operations(d);activations(d);all_clipping(d);kernels(d);ceilings(d)
