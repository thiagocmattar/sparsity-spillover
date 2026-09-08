"""Publication PDFs for this analysis only."""
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evidence import HERE, ROOT, FAMILIES, OPS, SCALES, one

STYLE = {'A0': ('#444444', 'o'), 'A1-H': ('#888888', 's'),
         'A1-H-L1': ('#009E73', 'o'), 'A1-H-OL1': ('#CC79A7', '^'),
         'A4': ('#0072B2', 'o'), 'A4-OL1': ('#0072B2', 's'),
         'A7': ('#D55E00', 'o'), 'A7-OL1': ('#D55E00', '^'),
         'A4-OL1[h]': ('#927523', 'D')}
OP_COLORS = ('#9ecae1', '#6baed6', '#2171b5', '#71bd9b', '#f6ac52', '#ce5446')
OP_LABELS = ('QKV', 'FFN up', 'FFN down', 'Attn. output', 'QK', 'PV')
S_LABEL = r'$\mathcal{S}_{\mathrm{model}}$ (%)'
U_LABEL = r'$100\mathcal{S}_{\mathrm{model}}/\mathcal{S}_{\mathrm{model}}^{\max}$ (%)'


def configure():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                        'axes.titlesize': 10, 'legend.fontsize': 8,
                        'pdf.fonttype': 42, 'axes.spines.top': False,
                        'axes.spines.right': False, 'axes.axisbelow': True})


def save(fig, name):
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures'/name, bbox_inches='tight',
                metadata={'Creator': 'Analysis 018', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def curve(ax, rows, family, x='R_model', y='loss', label=True, annotate=False):
    rows = sorted(rows, key=lambda r: r['dose'] if r['dose'] is not None else -1)
    c, marker = STYLE[family]
    ax.plot([100*r[x] for r in rows], [r[y] for r in rows], color=c,
            marker=marker, ms=4.5, lw=1.15,
            ls='--' if family in ('A4', 'A7', 'A4-OL1[h]') else '-',
            label=family if label else '_nolegend_')
    if annotate:
        for r in [r for r in rows if r['dose'] in (0., .5)]:
            offset=(4,-12) if r['dose']==0. and family=='A4-OL1' else (4,5)
            ax.annotate(f'{r["dose"]:g}', (100*r[x], r[y]), xytext=offset,
                        textcoords='offset points', fontsize=7, color=c)


def overview(d):
    rows = [r for r in d['trained'] if r['scale']=='14M']
    clip = [r for r in d['clipping'] if r['scale']=='14M' and r['family'] in ('gelu-control','relu-control')]
    fig, axs = plt.subplots(1,2,figsize=(10.5,4.7),gridspec_kw={'width_ratios':[2.5,1]},layout='constrained')
    for ax in axs:
        for f in FAMILIES:
            curve(ax, [r for r in rows if r['family']==f], f)
        for f,c in (('gelu-control','#444444'),('relu-control','#888888')):
            rr=sorted([r for r in clip if r['family']==f],key=lambda r:r['dose'])
            ax.plot([100*r['R_model'] for r in rr],[r['loss'] for r in rr],':',color=c,marker='.',ms=3,
                    label=('A0' if f=='gelu-control' else 'A1-H')+' + uniform clipping')
        for key, style in (('14M_main_trained', '-'), ('14M_all_trained', ':')):
            ids=d['frontiers'][key]; rr=[one(rows,id=i) for i in ids]
            ax.plot([100*r['R_model'] for r in rr],[r['loss'] for r in rr],color='#222222',lw=.7,ls=style,zorder=0)
            ax.scatter([100*r['R_model'] for r in rr],[r['loss'] for r in rr],s=64,facecolors='none',edgecolors='#222222',lw=.7)
        ax.set_xlabel(S_LABEL); ax.grid(alpha=.2)
    axs[0].set_ylim(5.04,6.15); axs[0].set_xlim(-.5,30)
    axs[0].set_ylabel('Final validation cross-entropy (nats/token)')
    axs[0].set_title('(a) Quality-focused view')
    axs[1].set_title('(b) Complete loss range'); axs[1].set_xlim(-.5,30)
    axs[1].set_ylim(5.0,max(r['loss'] for r in clip+rows)+.12)
    above=sum(r['loss']>6.15 for r in clip+rows)
    axs[1].text(.04,.96,f'{above} points above\npanel (a)',transform=axs[1].transAxes,va='top',fontsize=8)
    handles,labels=axs[0].get_legend_handles_labels()
    handles += [Line2D([0],[0],color='#222222',lw=.7,marker='o',mfc='none',ls=s) for s in ('-',':')]
    labels += ['Frontier: main ladder','Frontier: includes OL1[h]']
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('14M: pressure, thresholding and placement span different trade-offs',fontsize=12)
    save(fig,'01-14m-overview.pdf')


def effects(d):
    blocks=['GELU to ReLU','Add L1 at h','L1 to OL1 at h','A1-H to A4',
            'Add OL1 to A4','A4 to A7 gates','Add OL1 to A7']
    labels=['Replace GELU\nA0 → A1-H','Add local pressure\nA1-H → +L1',
            'Change method\nL1 → OL1 at h','Expand gates\nA1-H → A4',
            'Add pressure\nA4 → A4-OL1','Expand gates\nA4 → A7','Add pressure\nA7 → A7-OL1']
    colors=['#555555','#009E73','#CC79A7','#0072B2','#0072B2','#D55E00','#D55E00']
    fig,axs=plt.subplots(2,1,figsize=(12.3,5.4),sharex=True,layout='constrained')
    tickx=[];ticklabels=[];cursor=0
    for n,(block,label,color) in enumerate(zip(blocks,labels,colors)):
        rows=[r for r in d['contrasts'] if r['block']==block]
        xx=np.arange(len(rows))+cursor
        for ax,key in zip(axs,['delta_loss','delta_R_pp']):
            if n%2==0: ax.axvspan(xx[0]-.5,xx[-1]+.5,color='#eeeeee',zorder=-2)
            ax.vlines(xx,0,[r[key] for r in rows],color=color,lw=1)
            ax.scatter(xx,[r[key] for r in rows],c=color,s=27,zorder=3)
        center=(xx[0]+xx[-1])/2
        axs[0].text(center,1.03,label,ha='center',va='bottom',fontsize=8,
                    transform=axs[0].get_xaxis_transform())
        tickx.extend(xx);ticklabels.extend(['—' if r['dose'] is None else f'{r["dose"]:g}' for r in rows])
        axs[1].text(center,-.28,('λ' if rows[0]['dose_kind']=='lambda' else 'κ') if rows[0]['dose_kind']!='none' else '',
                    ha='center',fontsize=9,transform=axs[1].get_xaxis_transform())
        cursor=xx[-1]+1.6
    for ax in axs:
        ax.axhline(0,color='#222222',lw=.8);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Δ validation loss\nnegative is better');axs[0].set_ylim(-.2,.42)
    axs[1].set_ylabel('Δ model-wide sparsity (pp)\npositive is more');axs[1].set_ylim(-.65,13)
    axs[1].set_xticks(tickx,ticklabels,fontsize=8)
    fig.suptitle('14M matched effects: child minus its named comparator',fontsize=12)
    save(fig,'02-blocked-intervention-effects.pdf')


def scaling(d):
    fig,axs=plt.subplots(2,3,figsize=(10.7,6.0),sharey=True,layout='constrained')
    for j,scale in enumerate(SCALES):
        rows=[r for r in d['trained'] if r['scale']==scale]
        for i,key in enumerate(('R_model','U_arch')):
            ax=axs[i,j]
            for f in ('A4-OL1','A7-OL1'):
                curve(ax,[r for r in rows if r['family']==f],f,x=key,y='delta_loss_vs_A0',annotate=True)
            if i==0:
                for f in ('A0','A1-H'):
                    curve(ax,[r for r in rows if r['family']==f],f,y='delta_loss_vs_A0')
                for f,c in (('A4-OL1','#0072B2'),('A7-OL1','#D55E00')):
                    r=one(rows,family=f,dose=0.)
                    ax.axvline(100*r['ceiling']['R_model_max_fraction'],color=c,ls=':',lw=.8)
                ax.set_xlim(-2,100)
            else:
                ax.axvline(100,color='#444444',ls=':',lw=.8);ax.set_xlim(0,108)
            ax.axhline(0,color='#555555',lw=.8);ax.grid(alpha=.2)
            ax.set_xlabel(S_LABEL if i==0 else U_LABEL)
        ex=one(d['exposure'],scale=scale)
        axs[0,j].set_title(f'{scale} | {ex["tokens_per_parameter"]:.1f} tokens/parameter')
    for ax in axs[:,0]: ax.set_ylabel('Δ validation loss vs same-scale A0')
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Selected recipes transfer partly; reach normalization changes the question',fontsize=12)
    save(fig,'03-scale-transfer-and-ceilings.pdf')


def operations(d):
    fig,axs=plt.subplots(2,3,figsize=(10.8,6.1),gridspec_kw={'height_ratios':[1,1.1]},layout='constrained')
    for j,scale in enumerate(SCALES):
        rows=[one(d['trained'],scale=scale,family=f,dose=k) for f in ('A4-OL1','A7-OL1') for k in (0.,.5)]
        base=np.zeros(4); ax=axs[0,j]
        for op,c,label in zip(OPS,OP_COLORS,OP_LABELS):
            values=np.array([100*r['counts']['per_operation'][op]['zero_product_count']/r['counts']['model_product_count'] for r in rows])
            ax.bar(range(4),values,bottom=base,color=c,width=.64,label=label);base+=values
        for x,r in enumerate(rows):
            ax.plot([x-.36,x+.36],[100*r['ceiling']['R_model_max_fraction']]*2,color='#111111',lw=1.2)
            ax.text(x,base[x]-1.2,f'{base[x]:.1f}',ha='center',va='top',fontsize=8,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':.4})
        ax.set_ylim(0,100);ax.set_title(scale);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(4),['A4\n0','A4\n0.5','A7\n0','A7\n0.5'])
        vals=np.array([[100*r['counts']['per_operation'][op]['zero_product_count']/r['counts']['per_operation'][op]['product_count'] for r in rows] for op in OPS])
        ax=axs[1,j];ax.imshow(vals,cmap='Blues',vmin=0,vmax=100,aspect='auto')
        for y in range(6):
            for x in range(4):
                v=vals[y,x];ax.text(x,y,'<.01' if 0<v<.01 else f'{v:.1f}',ha='center',va='center',fontsize=8,color='white' if v>65 else '#111111')
        ax.set_xticks(range(4),['A4\n0','A4\n0.5','A7\n0','A7\n0.5'])
        ax.set_yticks(range(6),OP_LABELS if j==0 else ['']*6)
        ax.set_xlabel('OL1 recipe / κ')
    axs[0,0].set_ylabel('Contribution to model sparsity (pp)')
    axs[1,0].set_ylabel('Zero-product fraction within operation (%)')
    h,l=axs[0,0].get_legend_handles_labels();h.append(Line2D([0],[0],color='black',lw=1.2));l.append('Selected-site ceiling')
    fig.legend(h,l,loc='outside lower center',ncol=7,frameon=False)
    fig.suptitle('Separate learned zero rates from the work those operations represent',fontsize=12)
    save(fig,'04-operation-accounting.pdf')


def activations(d):
    sites=('m','h','q_post','k_post','v','attention_output')
    titles=('FFN input m','FFN hidden h','Query q (post-RoPE)','Key k (post-RoPE)','Value v','Attention output (post-Wo)')
    fig,axs=plt.subplots(2,6,figsize=(13.0,6.2),sharey=True,layout='constrained')
    for i,k in enumerate((0.,.5)):
        for j,(site,title) in enumerate(zip(sites,titles)):
            ax=axs[i,j]
            for n,f in enumerate(('A0','A4-OL1','A7-OL1')):
                r=one(d['trained'],scale='14M',family=f,dose=None if f=='A0' else k)
                a=r['sites'][site];c,m=STYLE[f]
                ax.plot(range(4),[100*v/a['total'] for v in a['mass_bands']],color=c,marker=m,lw=.9,ms=4,label=f)
                ax.text(.04,.91-n*.075,f'RMS {f}: {a["rms"]:.3g}',transform=ax.transAxes,color=c,fontsize=7.5)
            ax.set_yscale('symlog',linthresh=.01,linscale=.55);ax.set_ylim(-.001,2000)
            ax.set_yticks([0,.01,.1,1,10,100],['0','.01','.1','1','10','100'])
            ax.set_xticks(range(4),['0','(0,.001]','(.001,.01]','>.01'],rotation=50,ha='right',fontsize=7.5)
            ax.grid(axis='y',alpha=.2)
            if i==0: ax.set_title(title,fontsize=9)
            if j==0: ax.set_ylabel(f'κ = {k:g}\nActivation mass (%)')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('14M activation mass and RMS: four measured |x| bands at six matched sites',fontsize=12)
    save(fig,'05-activation-mass-grid.pdf')


def all_clipping(d):
    rows=[r for r in d['trained'] if r['scale']=='14M']
    clip=[r for r in d['clipping'] if r['scale']=='14M']
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.6),gridspec_kw={'width_ratios':[2,1]},layout='constrained')
    for ax in axs:
        for f in sorted({r['family'] for r in clip}):
            rr=sorted([r for r in clip if r['family']==f],key=lambda r:r['dose'])
            ax.plot([100*r['R_model'] for r in rr],[r['loss'] for r in rr],color='#bbbbbb',lw=.65,marker='.',ms=2,zorder=0)
        for f in FAMILIES: curve(ax,[r for r in rows if r['family']==f],f,label=False)
        ids=d['frontiers']['14M_all_trained_and_clipped']; rr=[one(rows+clip,id=i) for i in ids]
        ax.plot([100*r['R_model'] for r in rr],[r['loss'] for r in rr],color='#202020',lw=1.4,ls=':',marker='o',ms=4,mfc='white',label='Evaluated nondominated envelope')
        ax.set_xlabel(S_LABEL);ax.set_xlim(-.5,30);ax.grid(alpha=.2)
    axs[0].set_ylim(5.04,6.15);axs[1].set_ylim(5,max(r['loss'] for r in clip+rows)+.2)
    axs[0].set_ylabel('Final validation cross-entropy');axs[0].set_title('(a) Quality-focused view')
    axs[1].set_title('(b) All 185 evaluated points')
    axs[0].legend(frameon=False,loc='upper left')
    fig.suptitle('All 150 retained 14M clipping evaluations against all 35 trained conditions',fontsize=12)
    fig.supxlabel('Gray: 15 checkpoints × 10 uniform clipping targets. Colored: trained recipes (Figure 1).',fontsize=9)
    save(fig,'06-complete-posthoc-comparison.pdf')


def make_figures(d):
    configure()
    overview(d);effects(d);scaling(d);operations(d);activations(d);all_clipping(d)
    shutil.copyfile(ROOT/d['kernel_pdf_source'],HERE/'figures/07-kernel-realization.pdf')
