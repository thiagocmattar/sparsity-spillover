"""Paper-width figures for Analysis 018; numeric detail lives in the tables."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from evidence import HERE, FAMILIES, OVERVIEW_FAMILIES, OPS, SCALES, CASE_DOSES, CASE_SITES, one

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
    fig, ax = plt.subplots(figsize=(5.5,4.6))
    fig.subplots_adjust(left=.12, right=.98, bottom=.25, top=.91)
    # Thin evaluation trajectories sit behind the unchanged training sweeps.
    series_rows={name:[one(d['trained']+d['overview_clipping'],id=i) for i in ids]
                 for name,ids in d['overview_series'].items()}
    for series,rows in series_rows.items():
        if not series.endswith(' + clipping'):
            continue
        family=rows[0]['family']
        color,marker=STYLE[family]
        ax.plot([100*r['R_model'] for r in rows],[r['loss'] for r in rows],
                color=color,marker=marker,ms=1.8,mfc='white',mew=.35,
                ls='--',lw=.45,alpha=.28,zorder=2,label=series)
    legend=[]
    for family in OVERVIEW_FAMILIES:
        rows=series_rows[family];color,marker=STYLE[family]
        pressured=family.endswith(('-L1','-OL1'))
        linestyle='--' if pressured else '-'
        ax.plot([100*r['R_model'] for r in rows],[r['loss'] for r in rows],
                color=color,marker=marker,ms=4,mew=.65,lw=1.15,
                ls=linestyle,zorder=4,label=family)
        legend.append(Line2D([],[],color=color,marker=marker,ms=4,lw=1.15,
                             ls=linestyle if len(rows)>1 else 'none',label=family))
    ax.set(xlim=(-.5,29),ylim=(5.04,6.15),xlabel=S_LABEL,ylabel='Validation loss')
    ax.set_yticks(np.arange(5.2,6.2,.2))
    ax.grid(axis='both',color='.88',lw=.4)
    fig.suptitle('Quality-sparsity frontiers',y=.985,fontsize=11)
    fig.legend(handles=legend,loc='upper center',ncol=5,frameon=False,
               fontsize=7.5,columnspacing=.9,handletextpad=.45,handlelength=1.6,
               bbox_to_anchor=(.55,.145))
    clipping_key=Line2D([],[],color='.65',marker='o',mfc='white',mew=.35,
                        ms=1.8,lw=.45,ls='--',label='Post-hoc clipping')
    fig.legend(handles=[clipping_key],loc='lower center',frameon=False,fontsize=7.5,
               handlelength=2.5,bbox_to_anchor=(.55,.043))
    save(fig,'01-14m-overview.pdf')


def effects(d):
    blocks=['GELU to ReLU','Add L1 at h','L1 to OL1 at h','A1-H to A4',
            'Add OL1 to A4','A4 to A7 gates','Add OL1 to A7']
    labels=['Use ReLU\nat h','Add L1\nat h','L1 → OL1\nat h',
            r'Apply $G_+$'+'\nat a,m,h,z','Add OL1\nat a,m,h,z',
            r'Apply $G_{\pm}$'+'\nat q,k,v','Add OL1\nat a,m,h,z,\nq,k,v']
    pairs=['A1-H\n− A0','A1-H-L1\n− A1-H','A1-H-OL1\n− A1-H-L1',
           'A4\n− A1-H','A4-OL1\n− A4','A7\n− A4','A7-OL1\n− A7']
    colors=['#444444','#009E73','#CC79A7','#0072B2','#0072B2','#D55E00','#D55E00']
    fig,axs=plt.subplots(2,1,figsize=(5.5,4.6),sharex=True)
    fig.subplots_adjust(left=.16,right=.99,bottom=.30,top=.90,hspace=.12)
    # Compact the single comparison so the numeric threshold ticks have room.
    edges=np.r_[0.,np.cumsum([.85,1.,1.1,1.2,1.2,1.2,1.2])]
    centers=(edges[:-1]+edges[1:])/2
    for n,(block,color) in enumerate(zip(blocks,colors)):
        rows=sorted([r for r in d['contrasts'] if r['block']==block],key=lambda r:r['dose'] or 0)
        inset=.13 if len(rows)==4 else .09
        xx=(np.linspace(edges[n]+inset,edges[n+1]-inset,len(rows))
            if len(rows)>1 else np.array([centers[n]]))
        for ax,key in zip(axs,['delta_loss','delta_R_pp']):
            if n%2==0: ax.axvspan(edges[n],edges[n+1],color='.96',zorder=-2)
            ax.vlines(xx,0,[r[key] for r in rows],color=color,lw=.8)
            for x,r in zip(xx,rows):
                ax.scatter(x,r[key],color=color,marker='o',s=18,zorder=3)
        for x,r in zip(xx,rows):
            value='—' if r['dose'] is None else f'{r["dose"]:g}'.removeprefix('0.')
            if r['dose'] is not None and 0 < r['dose'] < 1:
                value='.'+value
            axs[1].annotate(value,(x,0),xycoords=('data','axes fraction'),
                            xytext=(0,-5),textcoords='offset points',
                            ha='center',va='top',fontsize=6.5,fontfamily='STIXGeneral')
        axs[1].plot([edges[n]+.05,edges[n+1]-.05],[-.16,-.16],
                    transform=axs[1].get_xaxis_transform(),clip_on=False,color='.65',lw=.5)
        parameter={'lambda':r'$\lambda$','kappa':r'$\kappa$','none':''}[rows[0]['dose_kind']]
        axs[1].annotate(parameter,(centers[n],0),xycoords=('data','axes fraction'),
                        xytext=(0,-18),textcoords='offset points',
                        ha='center',va='top',fontsize=7)
        axs[1].annotate(labels[n],(centers[n],0),
                        xycoords=('data','axes fraction'),xytext=(0,-42),
                        textcoords='offset points',ha='center',va='center',fontsize=6.5)
        axs[1].annotate(pairs[n],(centers[n],0),
                        xycoords=('data','axes fraction'),xytext=(0,-72),
                        textcoords='offset points',ha='center',va='center',
                        fontsize=6.5,color='black')
    for ax in axs:
        ax.axhline(0,color='.25',lw=.7); ax.grid(axis='y',color='.92',lw=.6)
        ax.tick_params(axis='x',length=0);ax.set_xlim(edges[0],edges[-1])
    axs[0].set(ylabel='Δ loss',ylim=(-.20,.42),yticks=[-.2,0,.2,.4])
    axs[1].set(ylabel='Δ '+S_LABEL.replace('(%)','(pp)'),ylim=(-.7,13),yticks=[0,4,8,12])
    axs[1].set_xticks([])
    for text,offset in [('Intervention:',-42),('Paired:',-72)]:
        axs[1].annotate(text,(0,0),xycoords='axes fraction',xytext=(-7,offset),
                        textcoords='offset points',ha='right',va='center',fontsize=7.5)
    fig.suptitle('Paired intervention effects',y=.985,fontsize=11)
    save(fig,'02-blocked-intervention-effects.pdf')


def effects_v2(d):
    """Alternative Figure 02: shared intervention rows, two effect columns."""
    blocks=['GELU to ReLU','Add L1 at h','L1 to OL1 at h','A1-H to A4',
            'Add OL1 to A4','A4 to A7 gates','Add OL1 to A7']
    labels=['GELU\n→ ReLU','Add L1\nat h','L1 → OL1\nat h','A1-H\n→ A4',
            'Add OL1\nto A4','A4\n→ A7','Add OL1\nto A7']
    colors=['#444444','#009E73','#CC79A7','#0072B2','#0072B2','#D55E00','#D55E00']
    fig,axs=plt.subplots(1,2,figsize=(5.5,5.5),sharey=True)
    fig.subplots_adjust(left=.30,right=.985,bottom=.10,top=.89,wspace=.23)
    centers=[]
    cursor=0
    for n,(block,color) in enumerate(zip(blocks,colors)):
        rows=sorted([r for r in d['contrasts'] if r['block']==block],key=lambda r:r['dose'] or 0)
        yy=cursor+np.arange(len(rows))
        centers.append(float(np.mean(yy)))
        for ax,key in zip(axs,['delta_loss','delta_R_pp']):
            if n%2==0:
                ax.axhspan(yy[0]-.5,yy[-1]+.5,color='.96',zorder=-2)
            ax.hlines(yy,0,[r[key] for r in rows],color=color,lw=.8)
            ax.scatter([r[key] for r in rows],yy,color=color,marker='o',s=18,zorder=3)
        for y,r in zip(yy,rows):
            value='—' if r['dose'] is None else f'{r["dose"]:g}'
            axs[0].annotate(value,(0,y),xycoords=('axes fraction','data'),
                            xytext=(-6,0),textcoords='offset points',
                            ha='right',va='center',fontsize=8,fontfamily='STIXGeneral')
        parameter={'lambda':r'$\lambda$','kappa':r'$\kappa$','none':''}[rows[0]['dose_kind']]
        axs[0].annotate(parameter,(0,centers[-1]),xycoords=('axes fraction','data'),
                        xytext=(-30,0),textcoords='offset points',
                        ha='center',va='center',fontsize=9)
        cursor+=len(rows)+1
    for ax in axs:
        ax.axvline(0,color='.25',lw=.7)
        ax.grid(axis='x',color='.92',lw=.6)
        ax.tick_params(axis='y',length=0)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(cursor-1.5,-.75)
    axs[0].set_yticks(centers,labels)
    axs[0].tick_params(axis='y',pad=48)
    axs[0].set_ylabel('Intervention',labelpad=12)
    axs[0].set(xlabel='Δ loss (nats/token)',xlim=(-.20,.42),xticks=[-.2,0,.2,.4])
    axs[1].set(xlabel='Δ '+S_LABEL.replace('(%)','(pp)'),xlim=(-.7,13),xticks=[0,4,8,12])
    axs[0].set_title('(a) Validation loss',fontsize=9,pad=8)
    axs[1].set_title('(b) Model sparsity',fontsize=9,pad=8)
    fig.suptitle('Matched intervention effects (14M)',y=.985,fontsize=11)
    save(fig,'02-v2-blocked-intervention-effects.pdf')


def scaling(d):
    fig,axs=plt.subplots(2,3,figsize=(5.5,5.1),sharey='row')
    fig.subplots_adjust(left=.12,right=.96,bottom=.14,top=.76,wspace=.14,hspace=.44)
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
            if i==0:
                for f,ls in (('A4-OL1','--'),('A7-OL1','-.')):
                    ceiling=one(rows,family=f,dose=0.)['ceiling']['R_model_max_percent']
                    ax.axvline(ceiling,color=STYLE[f][0],ls=ls,lw=.8,alpha=.8,zorder=0)
        axs[0,j].set_title(scale)
    axs[0,0].set_ylabel('Validation loss\n(nats/token)')
    axs[1,0].set_ylabel('Δ loss vs A0\n(nats/token)')
    axs[0,0].set_ylim(3.8,9.6);axs[1,0].set_ylim(-.35,5.4)
    axs[0,1].set_xlabel(S_LABEL);axs[1,1].set_xlabel(U_LABEL)
    h,l=axs[0,0].get_legend_handles_labels()
    h += [Line2D([],[],color=STYLE[f][0],ls=ls,lw=.8) for f,ls in (('A4-OL1','--'),('A7-OL1','-.'))]
    l += ['A4 / clipping ceiling','A7 ceiling']
    fig.legend(h,l,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.55,.94),
               handlelength=2.2,columnspacing=1.8)
    fig.suptitle('Quality-sparsity transfer across model sizes',y=.99,fontsize=11)
    save(fig,'03-scale-transfer-and-ceilings.pdf')


def operations(d):
    fig,ax=plt.subplots(figsize=(5.5,3.9))
    fig.subplots_adjust(left=.13,right=.99,bottom=.22,top=.74)
    rows=[one(d['trained'],scale=s,family=f,dose=.5) for s in SCALES for f in ('A4-OL1','A7-OL1')]
    xx=np.array([0,1,3,4,6,7]);base=np.zeros(6)
    for op,c,label in zip(OPS,OP_COLORS,OP_LABELS):
        values=np.array([100*r['counts']['per_operation'][op]['zero_product_count']/r['counts']['model_product_count'] for r in rows])
        ax.bar(xx,values,bottom=base,color=c,width=.68,label=label,
               edgecolor='white',linewidth=.35);base+=values
    ax.set(ylim=(0,100),ylabel='Contribution to '+S_LABEL.replace('(%)','(pp)'))
    ax.set_xticks(xx,['A4','A7']*3)
    for center,scale in zip((.5,3.5,6.5),SCALES):
        ax.text(center,-.22,scale,ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
    ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,
               columnspacing=1.0,handlelength=1.5,bbox_to_anchor=(.55,.91))
    fig.suptitle('Operation contributions to model-wide sparsity',y=.99,fontsize=11)
    save(fig,'04-operation-accounting.pdf')


def activations(d):
    sites=CASE_SITES;families=('A0','A4-OL1','A7-OL1')
    fig,axs=plt.subplots(3,5,figsize=(5.5,6.2),sharex=True,sharey=True)
    fig.subplots_adjust(left=.14,right=.99,bottom=.19,top=.82,wspace=.12,hspace=.22)
    for i,k in enumerate(CASE_DOSES):
        for j,site in enumerate(sites):
            ax=axs[i,j]
            for f in families:
                r=one(d['trained'],scale='14M',family=f,dose=None if f=='A0' else k)
                a=r['sites'][site];c,m=STYLE[f]
                ax.plot(range(4),[100*n/a['total'] for n in a['mass_bands']],
                        color=c,marker=m,ms=3,lw=.8)
            ax.set_yscale('symlog',linthresh=.01,linscale=.5)
            ax.set_ylim(-.002,140);ax.set_yticks([0,.01,1,100],['0','.01','1','100'])
            ax.set_xticks(range(4),['0','(0,.001]','(.001,.01]','>.01'],rotation=90)
            ax.grid(axis='y',color='.92',lw=.6)
            if i==0: ax.set_title(['$h$','$m$','$q$','$k$','$v$'][j])
            if j==0: ax.set_ylabel(r'$\kappa='+f'{k:g}'+r'$'+'\nMass (%)')
    fig.legend(handles=handles(families),loc='upper center',ncol=3,frameon=False,
               bbox_to_anchor=(.56,.94))
    fig.suptitle('14M activation magnitude distributions',y=.99,fontsize=11)
    fig.supxlabel(r'Activation magnitude $|x|$ bin',y=.055,fontsize=9)
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
    fig,ax=plt.subplots(figsize=(5.5,4.3))
    fig.subplots_adjust(left=.13,right=.98,bottom=.14,top=.70)
    families=('A0','A1-H','A1-H-L1','A1-H-OL1','A4')
    for f in families:
        rr=[r for r in clip if family(r)==f];c,m=STYLE[f]
        ax.scatter([100*r['R_model'] for r in rr],[r['loss'] for r in rr],
                   edgecolors=c,facecolors='none',marker=m,s=23,linewidths=.8)
    ax.set(xlim=(-.3,13),ylim=(5.,9.6),xlabel=S_LABEL,ylabel='Validation loss (nats/token)')
    ax.grid(axis='y',color='.92',lw=.6)
    hh=handles(families)
    for h,n in zip(hh,(1,1,4,4,5)):
        h.set_label(h.get_label()+f' ({n})');h.set_markerfacecolor('white')
    fig.legend(handles=hh,loc='upper center',ncol=3,frameon=False,
               title='Source family (number of checkpoints)',title_fontsize=8,
               bbox_to_anchor=(.55,.885),handletextpad=.4,columnspacing=1.1)
    fig.suptitle('14M post-hoc clipping',y=.99,fontsize=11)
    fig.text(.55,.925,'15 checkpoints, 10 clipping targets each',ha='center',fontsize=9)
    save(fig,'06-complete-posthoc-comparison.pdf')


def kernels(d):
    runtime=d['runtime'];fig,axs=plt.subplots(1,2,figsize=(5.5,3.65))
    fig.subplots_adjust(left=.11,right=.97,bottom=.17,top=.68,wspace=.34)
    progress=runtime['progress']
    axs[0].step([p['iteration'] for p in progress],[p['speedup'] for p in progress],where='post',color='#333333')
    axs[0].set(xlabel='Kernel iteration',ylabel='Speedup (×)',title='(a) Search progress',ylim=(.75,1.90))
    rows=[p for p in runtime['points'] if p['candidate']=='k050']
    for f in FAMILIES:
        rr=[r for r in rows if r['family']==f];c,m=STYLE[f]
        axs[1].scatter([100*r['R_model'] for r in rr],[r['speedup'] for r in rr],color=c,marker=m,s=22)
    fit=runtime['k050_regression'];xx=np.array([0,.30])
    axs[1].plot(100*xx,fit['intercept']+fit['slope_per_fraction']*xx,color='.55',ls='--',lw=.8,zorder=0)
    axs[1].set(xlabel=S_LABEL,title='(b) Final kernel',ylim=(.75,1.90),xlim=(-1,31))
    for ax in axs:
        ax.axhline(1,color='.65',ls=':',lw=.8);ax.grid(axis='y',color='.92',lw=.6)
    fig.suptitle('From kernel development to inference speedup',y=.99,fontsize=11)
    fig.legend(handles=handles(FAMILIES),loc='upper center',ncol=4,frameon=False,
               bbox_to_anchor=(.55,.91),handletextpad=.4,columnspacing=1.2)
    save(fig,'07-kernel-realization.pdf')


def ceilings(d):
    fig,ax=plt.subplots(figsize=(5.5,3.65))
    fig.subplots_adjust(left=.14,right=.96,bottom=.18,top=.74)
    for f,ls in zip(('A0','A1-H','A4','A7'),(':','--','-.','-')):
        rr=[one(d['ceilings'],scale=s,family=f) for s in SCALES];c,m=STYLE[f]
        ax.plot([r['parameters'] for r in rr],[r['R_model_max_percent'] for r in rr],
                color=c,marker=m,ls=ls,ms=5,label=f)
    ax.set_xscale('log');ax.set_xticks([one(d['exposure'],scale=s)['parameters'] for s in SCALES],SCALES)
    ax.minorticks_off();ax.set(ylim=(-3,100),xlabel='Model size (parameters)',
                              ylabel=r'Analytic ceiling $\mathcal{S}_{\mathrm{model}}^{\max}$ (%)')
    ax.grid(axis='y',color='.92',lw=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False,
               bbox_to_anchor=(.55,.91),handlelength=2.3)
    fig.suptitle('Theoretical sparsity ceiling by model size',y=.99,fontsize=11)
    save(fig,'08-ceiling-vs-model-size.pdf')


def make_figures(d):
    configure()
    overview(d);effects(d);effects_v2(d);scaling(d);operations(d);activations(d);all_clipping(d);kernels(d);ceilings(d)
