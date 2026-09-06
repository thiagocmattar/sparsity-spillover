"""Two vector publication figures from the complete audited study only."""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from common import RUN,read_json,write_json,record


def main():
    data=read_json(RUN/'results/summary.json');rows=data['rows']
    if len(rows)!=35 or not all(r['complete'] for r in rows):raise ValueError('Keep incomplete evidence visible; do not publish complete-cohort figure')
    families=list(dict.fromkeys(r['family'] for r in rows))
    colors=['#222222','#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7','#332288','#777777']
    markers=['o','s','^','v','D','P','X','*','h']
    style={f:(colors[i],markers[i]) for i,f in enumerate(families)}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,
        'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'lines.linewidth':.8,'savefig.transparent':False})
    def panel(ax,comparison,title,ylabel,*,detail=False):
        ax.axhline(1,color='#555555',lw=.65,ls=(0,(4,3)),zorder=1)
        for row in rows:
            c=row['comparisons'][comparison];y=c['ratio'];x=row['R_model_percent'];color,marker=style[row['family']]
            ax.errorbar(x,y,yerr=[[max(0,y-c['process_min'])],[max(0,c['process_max']-y)]],
                color=color,marker=marker,markersize=4.7 if marker!='*' else 6.5,
                markerfacecolor=color if c['qualified'] else 'white',markeredgewidth=.7,
                capsize=1.7,elinewidth=.65,lw=0,zorder=3)
            if not c['qualified']:ax.plot(x,y,color=color,marker='x',markersize=2.6,lw=0,zorder=4)
        ax.set(xlim=(-.6,30),xlabel=r'Canonical $R_{\mathrm{model}}$ (%)',ylabel=ylabel,title=title)
        ax.grid(axis='y',color='#dddddd',lw=.45);ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        if detail:
            low=min(r['comparisons'][comparison]['process_min'] for r in rows)
            high=max(r['comparisons'][comparison]['process_max'] for r in rows)
            ax.set_ylim(math.floor((min(low,1)-.005)*100)/100,math.ceil((max(high,1)+.005)*100)/100)
        else:
            high=max(r['comparisons'][comparison]['process_max'] for r in rows)
            ax.set_ylim(0,math.ceil(high*4)/4+.025)
    handles=[Line2D([],[],color=style[f][0],marker=style[f][1],lw=0,markersize=5,
                   label=f+(' (historical)' if f=='A4+OL1@h' else '')) for f in families]
    files=[];dest=RUN/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.55))
    panel(axes[0],'graph','(a) Full-model acceleration',r'Native graph / new sparse graph ($\times$)')
    panel(axes[1],'skip','(b) Skipping contribution (detail)',r'All skips disabled / enabled ($\times$)',detail=True)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=3,frameon=False,columnspacing=1.5)
    qualified=sum(r['modes']['sparse_graph']['qualified'] for r in rows)
    fig.text(.5,.975,'Pythia-14M | RTX 5090 | BF16 | batch 1 x 2,048 tokens | full logits',ha='center',va='top',fontsize=8)
    fig.text(.5,.932,f'35 variants; 3 processes x 64 inputs x 7 paired passes; {qualified}/35 new kernels qualified',ha='center',va='top',fontsize=7.3)
    fig.text(.5,.023,'Bars: three-process min-max (not confidence intervals). Panel (b) uses an expanded vertical scale.',ha='center',fontsize=6.8)
    fig.subplots_adjust(left=.09,right=.985,bottom=.32,top=.845,wspace=.36)
    path=dest/'01-rmodel-speedup-and-skip-contribution.pdf'
    fig.savefig(path,metadata={'Title':'R_model, full-model acceleration and isolated skipping contribution',
        'Creator':'Run028/66_figures.py','CreationDate':None,'ModDate':None});plt.close(fig);files.append(record(path))

    fig,axes=plt.subplots(2,2,figsize=(7.2,7.4))
    panel(axes[0,0],'eager','(a) Matched eager execution',r'Native eager / new sparse eager ($\times$)')
    panel(axes[0,1],'new_over_previous_graph','(b) Previous-kernel comparator',r'Previous / new sparse graph ($\times$)')
    panel(axes[1,0],'attention','(c) Attention skipping alone (detail)',r'Attention skips disabled / enabled ($\times$)',detail=True)
    row=next(r for r in rows if r['condition']=='c30')
    operations=['qkv_projection','mlp_w1','mlp_w2','attention_output_projection','qk_scores','probability_value']
    labels=['a QKV','m W1','h W2','z Wo','QK','PV'];x=np.arange(6)
    ax=axes[1,1]
    ax.bar(x-.19,[100*row['operations'][o]['zero_fraction_lower_bound'] for o in operations],.36,
           color='#999999',label='BF16 scalar zero products (lower bound)')
    ax.bar(x+.19,[100*row['operations'][o]['mma_skipped_fraction'] for o in operations],.36,
           color='#0072B2',hatch='//',label='Skipped tensor-core MMA instructions')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,105),ylabel='Fraction within operation (%)',
           title='(d) Zero pattern versus skippable work',xlabel=r'A7+OL1@7, $\kappa=0.5$ (c30)')
    ax.tick_params(axis='x',labelsize=7);ax.grid(axis='y',color='#dddddd',lw=.45);ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,-.34),frameon=False,fontsize=6.5)
    extra=Line2D([],[],color='#555555',marker='x',lw=0,label='Open + crossed: numerical gate failed')
    fig.legend(handles=handles+[extra],loc='lower center',bbox_to_anchor=(.5,.063),ncol=3,frameon=False,columnspacing=1.4)
    fig.text(.5,.982,'Controls and granularity | all 35 variants retained | complete 338-block numerical checks',ha='center',va='top',fontsize=8)
    fig.text(.5,.023,'Canonical R: source FP16. Timings and (d): BF16. MMA counts include causal padding; a/m weight loads remain.',ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.09,right=.985,bottom=.265,top=.925,wspace=.37,hspace=.58)
    path=dest/'02-controls-and-zero-granularity.pdf'
    fig.savefig(path,metadata={'Title':'Eager, previous-kernel and attention controls; zero granularity',
        'Creator':'Run028/66_figures.py','CreationDate':None,'ModDate':None});plt.close(fig);files.append(record(path))
    write_json(RUN/'results/figure-provenance.json',{'figures':files,'source':record(RUN/'results/summary.json'),'script':record(__file__)})
    lines=['# All 35 variants: full-model speedup and sparse contribution','',
        'Ratios are paired geometric means across three fresh processes. Brackets are the minimum and maximum process geometric means, not confidence intervals. R_model is the canonical source FP16 logical opportunity; timing uses BF16. Each ratio compares the same checkpoint. Dose is lambda for A1-H pressure families and kappa for A4/A7. Historical A4+OL1@h is not four-site pressure. No numerical failure is excluded.','',
        '| ID | Variant | Dose | R_model (%) | Eager speedup | Graph speedup | Skip contribution | Attention contribution | New qualified | Previous qualified |',
        '|---|---|---:|---:|---:|---:|---:|---:|:---:|:---:|']
    for r in rows:
        def fmt(label):
            c=r['comparisons'][label]
            return f"{c['ratio']:.3f} [{c['process_min']:.3f}, {c['process_max']:.3f}]"+(' FAILED' if not c['qualified'] else '')
        dose='N/A' if r['dose'] is None else f"{r['dose']:g}"
        lines.append(f"| {r['condition']} | {r['family']} | {dose} | {r['R_model_percent']:.3f} | {fmt('eager')} | {fmt('graph')} | {fmt('skip')} | {fmt('attention')} | {'yes' if r['modes']['sparse_graph']['qualified'] else 'FAILED'} | {'yes' if r['modes']['previous_graph']['qualified'] else 'FAILED'} |")
    (RUN/'results/per-variant.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'figures':files})


if __name__=='__main__':main()
