"""Publication PDF and source-linked observation for the accepted head comparison."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io_utils import RUN, read


def main():
    rows=read(RUN/'results/comparison.json')['rows']
    modes=['dense','custom_no_skip','custom_sparse']
    labels=['Dense cuBLAS','Custom no-skip','Custom sparse']
    hatches=['','//','xx']
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    boundaries=['projection_only','gate_cast_projection','full_forward']
    titles=['Projection only','Gate + cast + projection','Complete forward']
    colors=['#D55E00','#D55E00','#444444']
    x=np.arange(len(rows));width=.24
    for ax,boundary,title,color in zip(axes,boundaries,titles,colors):
        for offset,(mode,label,hatch) in enumerate(zip(modes,labels,hatches)):
            values=[r['timings'][boundary]['host_ms']['geomean_ms'][mode]*1000 for r in rows]
            ax.bar(x+(offset-1)*width,values,width,facecolor=color,edgecolor='black',hatch=hatch,label=label,linewidth=.6)
        ax.set_title(title);ax.set_ylabel('Synchronized host latency (µs)')
        ax.set_xticks(x,[f'κ={r["kappa"]:g}\n'+('Ph+head' if len(r['pressure_sites'])==2 else 'Ph') for r in rows])
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='white',edgecolor='black',hatch=h,label=l) for h,l in zip(hatches,labels)],
               loc='outside upper center',ncol=3,frameon=False)
    fig.suptitle('14M frozen vocabulary-head checkpoints · RTX 5090 · B1/T2048 · FP16 projection',y=1.11)
    output=RUN/'figures/01-sparse-vocabulary-head.pdf';output.parent.mkdir(exist_ok=True)
    fig.savefig(output,bbox_inches='tight');plt.close(fig)
    observations=RUN/'observations';observations.mkdir(exist_ok=True)
    lines=['# Sparse vocabulary-head latency','',
           '**Question:** Does the new consumer accelerate the vocabulary head and full forward of the four frozen Run066 models?','',
           '**Method and coverage:** One physical RTX5090, three fresh processes per model, 64 fixed inputs × 7 paired passes per process. B1/T2048; unchanged FP32 parameters, FP32 8x16 corner gate and FP16 projection. Every full graph qualifies against its own unmodified eager model over all500 documents/338 complete blocks,692224 inputs/691886 predictions;1444 tail tokens excluded. Separate head graphs qualify on every timed operand.','',
           '**Figure and caption:** [PDF](../figures/01-sparse-vocabulary-head.pdf). Three separately measured latency boundaries, with zero-based axes. The head retains the vermillion vocabulary-projection color used in Analysis042. Complete forward is gray. Fill/hatching identifies execution: solid dense cuBLAS, diagonal custom no-skip, crossed custom sparse. Every output vocabulary entry is computed.','',
           '**Result:** See [the complete table](../results/REPORT.md) and [paired/process data](../results/comparison.json). Positive latency reductions indicate lower latency; negative reductions indicate a slowdown. Report both the cuBLAS comparison and the same-custom-kernel skip control.','',
           '**Caveats:** Component boundaries are not additive decomposition segments. Projection-only excludes recurring activation preparation. The M16 consumer skips a K16 group only when both constituent M8 gate tiles are empty; actual bypass differs from 8x16 gate sparsity. Source-level avoided weight loads do not establish measured DRAM savings. Local development timings are excluded. No cached decoding, larger-model or multi-training-seed claim follows.','',
           '**Sources:** [20_analyze.py](../20_analyze.py) reduces verified returned artifacts; [21_figure.py](../21_figure.py) generates this PDF. Frozen checkpoint/runtime/cache identities are in provenance/inputs.json. No manuscript modification is included.']
    (observations/'01-sparse-vocabulary-head.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (observations/'INDEX.md').write_text('# Observations\n\n- [Sparse vocabulary-head latency](01-sparse-vocabulary-head.md) — matched head and full-forward comparison.\n',encoding='utf-8',newline='\n')


if __name__=='__main__':main()
