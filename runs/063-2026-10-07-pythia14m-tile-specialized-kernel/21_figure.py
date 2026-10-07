"""One source-linked PDF after verified full benchmark evidence exists."""
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io_utils import RUN,read,record

def main():
    result=read(RUN/'results/results.json')
    if result['status']!='verified' or len(result['pairs'])!=2:raise ValueError('Verified evidence required')
    fig,axes=plt.subplots(1,2,figsize=(8,3.2),layout='constrained')
    for ax,p in zip(axes,result['pairs']):
        for x,mode,color in [(0,'generic_sparse','#4477aa'),(1,'tile_sparse','#ee7733')]:
            ms=p['host_geomean_ms']['tile_'+mode]
            ax.bar(x,ms,color=color,width=.6)
            ax.text(x,ms,f'{ms:.4f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks([0,1],['Generic','Specialized']);ax.set_ylabel('Full-model host latency (ms)')
        ax.set_ylim(0,max(p['host_geomean_ms']['tile_'+m] for m in ['generic_sparse','tile_sparse'])*1.2)
        s=p['speedup']['specialized_over_generic_sparse']
        ax.set_title(f"kappa={p['kappa']:g} | speedup={s:.3f}x")
        ax.spines[['top','right']].set_visible(False)
    target=RUN/'figures/01-generic-versus-specialized.pdf';target.parent.mkdir(exist_ok=True)
    fig.savefig(target);plt.close(fig)
    obs=RUN/'observations';obs.mkdir(exist_ok=True)
    lines=['# Generic versus corner-specialized inference','',
           'Question: does direct consumption of the existing corner8x16 structure lower latency?',
           'Method: paired synchronized host replay; same checkpoints/gate, five graph implementations.',
           'Coverage: RTX5090 BF16 B1/T2048 full50304 logits;64 inputs x7 passes x3 fresh processes per threshold;',
           'all500 validation documents,338 blocks,1444-token excluded tail. All graphs qualified.',
           'Legend/caption: blue generic Run061 consumer; orange Run063 specialization; bars are pooled geometric mean host ms.',
           'Results:']
    for p in result['pairs']:
        s=p['speedup']['specialized_over_generic_sparse'];process=[q['ratios']['specialized_over_generic_sparse'] for q in p['processes']]
        lines.append(f"- kappa={p['kappa']:g}: {s:.6f}x; latency change={(1/s-1)*100:.3f}%; process ratios={process}.")
    lines+=['','Caveats: one trained seed/GPU; unchanged quality penalty of the corner gate; no decoding/DRAM portability claim.',
            'Source script: [21_figure.py](../21_figure.py). Evidence: [results.json](../results/results.json).',
            'Figure: [PDF](../figures/01-generic-versus-specialized.pdf).',
            f"Result SHA256: {record(RUN/'results/results.json')['sha256']}."]
    (obs/'01-generic-versus-specialized.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    (obs/'INDEX.md').write_text('# Observations\n\n- [01: same-gate kernel specialization](01-generic-versus-specialized.md).\n',encoding='utf-8',newline='\n')

if __name__=='__main__':main()
