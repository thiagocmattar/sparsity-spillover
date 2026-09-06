"""Publication PDFs from audited observations, with no runtime selection."""
import csv
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
from run027_common import RUN, read_json, record, write_json


def main():
    data=read_json(RUN/'results/summary.json'); rows=data['rows']
    families=list(dict.fromkeys(r['family'] for r in rows))
    colors=['#222222','#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#CC79A7','#332288','#777777']
    markers=['o','s','^','v','D','P','X','*','h']
    style={f:(colors[i],markers[i]) for i,f in enumerate(families)}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
        'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'lines.linewidth':.8,'savefig.transparent':False})
    dest=RUN/'figures';dest.mkdir(exist_ok=True)
    files=[]

    def panel(ax,xfield,comparison,title,xlabel,xmax):
        ax.axhline(1,color='#555555',lw=.65,ls=(0,(4,3)),zorder=1)
        for family in families:
            group=[r for r in rows if r['family']==family]
            color,marker=style[family]
            # Connect dose-ordered measurements only; no fitted performance law.
            valid=[r for r in group if r[comparison+'_qualified']]
            if len(valid)>1:
                ax.plot([r[xfield] for r in valid],[r[comparison+'_speedup'] for r in valid],
                    color=color,alpha=.30,lw=.65,zorder=2)
            for r in group:
                x=r[xfield];y=r[comparison+'_speedup'];good=r[comparison+'_qualified']
                ax.errorbar(x,y,yerr=[[y-r[comparison+'_ci_low']],[r[comparison+'_ci_high']-y]],
                    color=color if good else '#aaaaaa',marker=marker if good else 'x',
                    markersize=4.8 if marker!='*' else 7,markeredgewidth=.65,
                    markerfacecolor='white' if r['historical'] else color,
                    capsize=1.5,elinewidth=.65,lw=0,zorder=3)
        ax.set(xlim=(0,xmax),xlabel=xlabel,title=title)
        ax.grid(axis='y',color='#dddddd',lw=.4)
        ax.set_axisbelow(True);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(.5))

    handles=[Line2D([],[],color=style[f][0],marker=style[f][1],lw=.65,markersize=5,
        markerfacecolor='white' if all(r['historical'] for r in rows if r['family']==f) else style[f][0],
        label=f+(' (historical)' if all(r['historical'] for r in rows if r['family']==f) else '')) for f in families]
    if any(not r['sparse_qualified'] for r in rows):
        handles.append(Line2D([],[],color='#aaaaaa',marker='x',lw=0,label='Numerical gate failed'))
    rx=math.ceil(max(r['R_model_percent'] for r in rows)/5)*5
    common_max=math.ceil(max(r[c+'_ci_high'] for r in rows for c in ['overall','skip_only','beyond_qkv_fusion'])*2)/2
    common_max=max(2.,common_max)
    specs=[('01-speedup-vs-rmodel.pdf',[
        ('R_model_percent','overall','(a) End-to-end benefit',r'Logical opportunity $R_{\mathrm{model}}$ (%)',rx,r'Stock eager / sparse latency ($\times$)'),
        ('R_model_percent','skip_only','(b) Zero-skipping contribution',r'Logical opportunity $R_{\mathrm{model}}$ (%)',rx,r'Same fusion, no skip / sparse ($\times$)')]),
        ('02-fusion-and-eligible-sparsity.pdf',[
        ('R_model_percent','beyond_qkv_fusion','(a) Beyond QKV fusion',r'Logical opportunity $R_{\mathrm{model}}$ (%)',rx,r'QKV fusion + dense / sparse ($\times$)'),
        ('eligible_zero_percent','skip_only','(b) Actually eligible zero products',r'BF16 zeros in $h/z$ projection products (%)',100,r'Same fusion, no skip / sparse ($\times$)')])]
    for r in rows:r['eligible_zero_percent']=100*r['eligible_BF16_zero_fraction']
    for name,panels in specs:
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.9),sharey=True)
        for ax,(field,comparison,title,xlabel,xmax,ylabel) in zip(axes,panels):
            panel(ax,field,comparison,title,xlabel,xmax);ax.set_ylabel(ylabel)
            ax.set_ylim(0,common_max*1.025)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.01),ncol=3,
            frameon=False,columnspacing=1.3,handletextpad=.5)
        fig.text(.5,.975,'Pythia-14M · RTX 5090 · BF16 · batch 1 × 2,048 tokens · full logits',
            ha='center',va='top',fontsize=8)
        fig.subplots_adjust(left=.085,right=.985,bottom=.28,top=.855,wspace=.36)
        target=dest/name
        fig.savefig(target,metadata={'Title':name,'Subject':'35 existing trained checkpoints; paired full-model latency and count-pooled logical sparsity',
            'Creator':'Run027/11_plot.py','CreationDate':None,'ModDate':None})
        plt.close(fig);files.append(record(target))
    write_json(RUN/'results/figure-provenance.json',{'figures':files,'source':record(RUN/'results/summary.json'),'script':record(__file__)})
    lines=['# Per-variant speedup','',
        'Canonical count-pooled R_model. Ratios are geometric means of paired full-model latencies; brackets are 95% crossed process/input bootstrap intervals. Three processes, 64 inputs, seven passes. Gate failures are explicitly marked. Historical A4+OL1@h is not the corrected four-site pressure intervention.','',
        '| Variant | Dose | R_model (%) | Native / sparse (×) | No-skip / sparse (×) | Native → sparse (ms) | Qualified |',
        '|---|---:|---:|---:|---:|---:|:---:|']
    for r in rows:
        fmt=lambda key:f'{r[key+"_speedup"]:.3f} [{r[key+"_ci_low"]:.3f}, {r[key+"_ci_high"]:.3f}]'
        lines.append(f'| {r["family"]}{" (historical)" if r["historical"] else ""} | {r["dose"]} | {r["R_model_percent"]:.3f} | {fmt("overall")} | {fmt("skip_only")} | {r["native_latency_ms"]:.3f} → {r["sparse_latency_ms"]:.3f} | {"yes" if r["sparse_qualified"] else "FAILED"} |')
    (RUN/'results/per-variant.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({'figures':files,'families':families,'y_max':common_max})


if __name__=='__main__':main()
