"""PDF-only final latency figure and its evidence-linked observation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io_utils import RUN,read,write,record,verify


def main():
    result=read(RUN/'results/results.json')
    for pair in result['pairs']:
        for source in pair['source_files']:
            verify(source)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),gridspec_kw={'width_ratios':[1.5,1]})
    x=np.arange(4);width=.24
    conditions=[(p,f) for p in result['pairs'] for f in ['scalar','tile']]
    colors={'native':'#8B969C','no_skip':'#DCAB65','sparse':'#286780'}
    for mode,offset,label in [('native',-1,'Native'),('no_skip',0,'No H/Z skip'),('sparse',1,'Sparse')]:
        values=[p['host_geomean_ms'][f+'_'+mode] for p,f in conditions]
        bars=axes[0].bar(x+offset*width,values,width,color=colors[mode],label=label)
        for bar,value in zip(bars,values):
            axes[0].annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),
                             textcoords='offset points',ha='center',fontsize=7)
    axes[0].set_xticks(x,[f"{f.capitalize()}\nκ={p['kappa']:g}" for p,f in conditions])
    axes[0].set_ylabel('Full-model host latency (ms; geometric mean)')
    axes[0].set_title('Matched execution: lower is faster')
    maximum=max(p['host_geomean_ms'][f+'_'+m] for p,f in conditions for m in colors)
    axes[0].set_ylim(0,maximum*1.3);axes[0].legend(frameon=False,ncol=3,fontsize=8)
    ratios=[]
    for index,pair in enumerate(result['pairs']):
        values=[p['ratios']['tile_over_scalar_sparse'] for p in pair['processes']]
        ratios+=values
        axes[1].scatter(np.linspace(index-.07,index+.07,3),values,s=30,color='#286780',label='Fresh processes' if index==0 else None)
        pooled=pair['speedup']['tile_over_scalar_sparse']
        axes[1].scatter(index,pooled,marker='D',facecolor='white',edgecolor='#286780',s=45,
                        label='Pooled paired ratio' if index==0 else None)
        axes[1].annotate(f'{pooled:.3f}x',(index,max(values)),xytext=(0,9),textcoords='offset points',ha='center')
    axes[1].axhline(1,color='#777777',linestyle='--',linewidth=1)
    lo,hi=min([1]+ratios),max([1]+ratios);pad=max(.05,(hi-lo)*.4)
    axes[1].set_ylim(lo-pad,hi+pad*1.5);axes[1].set_xlim(-.4,1.4)
    axes[1].set_xticks([0,1],['0','0.1']);axes[1].set_xlabel('Threshold κ')
    axes[1].set_ylabel('Scalar latency / tile latency')
    axes[1].set_title('Sparse tile versus sparse scalar')
    axes[1].legend(frameon=False,fontsize=8,loc='best')
    fig.text(.5,.02,'RTX 5090 · BF16 · B1/T2048 · uncached · full logits · one training seed; different quality',
             ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.055,1,1])
    dest=RUN/'figures/01-corner-tile-latency.pdf';dest.parent.mkdir(exist_ok=True)
    fig.savefig(dest,bbox_inches='tight');plt.close(fig)
    write(RUN/'results/figure-source.json',{'figure':record(dest),'source_script':record(__file__),
                                          'source_result':record(RUN/'results/results.json')})
    observation=RUN/'observations';observation.mkdir(exist_ok=True)
    lines=['# Does corner-tile training lower full-model inference latency?','',
           'Question: compare the retained scalar and causal8x16 corner T2/Ph checkpoints at κ=0/.1.',
           'Method: fixed K050-derived h/z consumer with exact trained gates; matched native, no-h/z-skip and sparse graphs.',
           'Coverage: one RTX5090, BF16 B1/T2048, all50304 logits; each of six fresh processes checks338 validation blocks from500 documents.',
           'Each threshold has64 fixed timing inputs, seven randomized passes, three fresh processes and1344 direct pairs; excluded tail1444 tokens.',
           'Caption: left, full-model host geometric mean latency; right, same-input scalar/tile sparse latency ratios. Ratios above1 favor tile.',
           'The no-skip control disables h/z tile bypass and short-row substitution together; remaining components and gates are matched.','']
    for p in result['pairs']:
        s=p['speedup']
        lines.append(f"At κ={p['kappa']:g}, tile/scalar sparse speedup is {s['tile_over_scalar_sparse']:.6f}x; matched no-skip/sparse ratios are {s['scalar_sparse_over_no_skip']:.6f}x (scalar) and {s['tile_sparse_over_no_skip']:.6f}x (tile).")
    lines+=['','Caveats: one training seed/GPU; repetitions assess systems variability. Tile checkpoints have higher original validation loss. No quality-matched, decoding, scaling or measured DRAM-traffic claim.',
            'Sources: [table](../results/tables.md), [machine results](../results/results.json), [reducer](../20_analyze.py), [figure script](../21_figure.py), [PDF](../figures/01-corner-tile-latency.pdf).']
    (observation/'01-corner-tile-latency.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (observation/'INDEX.md').write_text('# Observations\n\n- [01: full-model corner/scalar latency](01-corner-tile-latency.md).\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
