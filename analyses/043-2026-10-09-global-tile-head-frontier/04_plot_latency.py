"""Measured CUPTI budgets and separate, overlapping vocabulary-head boundaries."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from evidence import HERE

MODES=('dense','custom_no_skip','custom_sparse')
MODE_COLORS=('#0072B2','#E69F00','#009E73')
MODE_LABELS=('Native dense','Custom no-skip','Custom sparse')


def main():
    data=json.loads((HERE/'data/latency.json').read_text(encoding='utf-8'))
    d=data['decomposition'];styles=d['semantic_styles']
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axis=plt.subplots(figsize=(10.5,6.1));axes=[axis]
    fig.subplots_adjust(left=.10,right=.985,top=.79,bottom=.24,wspace=.18)
    for ax,model in zip(axes,('14M',)):
        conditions=[c for c in d['conditions'] if c['model']==model]
        rows=[]
        for c in conditions:
            for mode in ('native',c['selected_sparse']):
                rows.append((c,mode,c['modes'][mode]))
        for index,(condition,mode,row) in enumerate(rows):
            offset=0.
            for category in d['semantic_category_order']:
                value=row['semantic_categories'].get(category,0)/1000
                style=styles[category]
                ax.barh(index,value,left=offset,color=style['color'],hatch=style['hatch'],height=.65,
                        edgecolor='white',linewidth=.3)
                offset+=value
            assert abs(offset-sum(row['categories'].values())/1000)<1e-9
            ax.text(offset+.025,index,f"host {row['host']['geomean']/1000:.3f} ms",va='center',fontsize=8)
        ax.set_yticks(range(len(rows)),[f"κ={c['kappa']} · {'dense' if m=='native' else 'sparse'}" for c,m,_ in rows])
        ax.invert_yaxis()
        ax.set(xlabel='Mean summed CUPTI activities (ms)',xlim=(0,.98))
        ax.set_title(f"{model}: {'h/z tiles 8×16' if model=='14M' else 'h/z scalar gates'}\nVocabulary head threshold OFF",fontsize=10)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    handles=[Patch(facecolor=s['color'],hatch=s['hatch'],label=k) for k,s in styles.items()]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=9,bbox_to_anchor=(.5,.03))
    fig.suptitle('14M full-model latency decomposition — fixed categories and colors',fontsize=14,y=.97)
    fig.text(.5,.89,'Run064 · 12 CUPTI traces per condition/mode · arithmetic activity means\nHost labels are independently measured geometric means; component bars are not rescaled to them.',ha='center',fontsize=9)
    fig.savefig(HERE/'figures/03-existing-full-decomposition.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    rows=data['head_boundaries'];x=np.arange(len(rows))
    ticks=[f"H{r['head_tile_tokens']} {'Ph' if r['pressure']=='h' else 'Ph+head'}\nh/z κ={r['block_kappa']:g}\nhead κ={r['head_kappa']:g}" for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(16,6.0))
    fig.subplots_adjust(left=.065,right=.985,top=.78,bottom=.28,wspace=.25)
    for ax,(boundary,title) in zip(axes,[('projection_only','Projection only'),('gate_cast_projection','Gate + cast + projection'),('full_forward','Complete forward')]):
        for j,mode in enumerate(MODES):
            values=[r['timings'][boundary]['host_ms']['geomean_ms'][mode] for r in rows]
            ranges=[[p['timings'][boundary]['host_ms']['geomean_ms'][mode] for p in r['processes']] for r in rows]
            low=[v-min(a) for v,a in zip(values,ranges)];high=[max(a)-v for v,a in zip(values,ranges)]
            ax.bar(x+(j-1)*.25,values,width=.23,color=MODE_COLORS[j],hatch=('', '..','//')[j],
                   yerr=np.array([low,high]),capsize=2,label=MODE_LABELS[j])
        ax.set(xticks=x,xticklabels=ticks,ylabel='Synchronized host latency (ms)',title=title,ylim=(0,None))
        ax.tick_params(axis='x',labelsize=7.5,labelrotation=35)
        for tick in ax.get_xticklabels():tick.set_horizontalalignment('right')
        ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
    fig.suptitle('Vocabulary-head intervention: measured boundaries with dense transformer projections',fontsize=13,y=.97)
    fig.text(.5,.89,'Runs067/068 · FP16 · 3 fresh processes/checkpoint · 64 inputs × 7 paired passes\nThese three boundaries overlap. Full CUPTI decomposition is unavailable for the head experiments.',ha='center',fontsize=9)
    fig.legend(handles=axes[0].get_legend_handles_labels()[0],labels=MODE_LABELS,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.04))
    fig.savefig(HERE/'figures/04-head-measured-boundaries.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    print({'status':'latency-plotted','figures':2},flush=True)


if __name__=='__main__':
    main()
