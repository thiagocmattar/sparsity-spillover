"""Complete final-kernel clipping sweeps, with all measured targets visible."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json'
OUTPUT=HERE/'figures/07-controls-posthoc-final-latency.pdf'


def main():
    data=json.loads(SOURCE.read_text())
    assert data['status']=='complete_verified_reduction' and len(data['points'])==40
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,3,figsize=(12.0,6.4),sharey='row')
    fig.subplots_adjust(left=.07,right=.99,top=.93,bottom=.15,hspace=.35,wspace=.22)
    styles=[('A0','Base model (GeLU)','#52565C','o'),('A1-H','ReLU / 1-site','#718052','s')]
    series=[]
    for row,scale in enumerate(['14M','70M']):
        for family,label,color,marker in styles:
            points=sorted([p for p in data['points'] if p['scale']==scale and p['family']==family],key=lambda p:p['p'])
            assert [p['p'] for p in points]==[i/10 for i in range(10)]
            series.append({'scale':scale,'family':family,'points':points})
            for col,key in enumerate(['p','sparsity_percent','retained_fp16_loss']):
                ax=axes[row,col]
                # Break at any failed qualification; retain failed timings as crosses.
                ys=[p['candidate_gm_ms'] if p['qualified'] else float('nan') for p in points]
                ax.plot([p[key] for p in points],ys,color=color,marker=marker,ms=4,lw=1.2,ls=':',mec='white',mew=.3)
                for p in points:
                    if not p['qualified']:
                        ax.scatter(p[key],p['candidate_gm_ms'],color=color,marker='x',s=28)
                ax.grid(axis='y',alpha=.16,lw=.5)
            for p in points:
                if p['p'] in (0.,.5,.9):
                    dy=5 if family=='A0' else -11
                    if scale=='70M' and p['p']==.9:dy=-10 if family=='A0' else 8
                    axes[row,2].annotate(f"p={p['p']:g}",(p['retained_fp16_loss'],p['candidate_gm_ms']),
                        xytext=(-6 if p['p']==.9 else 4,dy),
                        ha='right' if p['p']==.9 else 'left',textcoords='offset points',fontsize=7,color=color)
        axes[row,0].set_ylabel(f'{scale}: full-model latency (ms)')
        axes[row,0].set_xlabel('Clipping target p');axes[row,0].set_xticks([0,.3,.6,.9])
        axes[row,1].set_xlabel('Model-wide logical sparsity (%)')
        axes[row,2].set_xlabel('Validation loss (nats; retained FP16)')
        for ax in axes[row]:ax.margins(x=.08,y=.12)
    for ax,title in zip(axes[0],['(a) Clipping target','(b) Logical sparsity','(c) Quality-latency trade-off']):
        ax.set_title(title,loc='left')
    handles=[Line2D([],[],color=color,marker=marker,ls=':',label=label) for _,label,color,marker in styles]
    if not all(p['qualified'] for p in data['points']):
        handles.append(Line2D([],[],color='black',marker='x',ls='none',label='Numerical qualification failed'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.025),ncol=len(handles),frameon=False)
    fig.savefig(OUTPUT);plt.close(fig)
    evidence={'source':SOURCE.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script':Path(__file__).name,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figure':OUTPUT.name,'figure_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        'series':series,'displayed_points':40,'qualified_points':sum(p['qualified'] for p in data['points']),
        'axes':'All targets/full loss range. Y shared within size, independent between sizes. Clipping inside timed graph.'}
    (HERE/'data/controls-clipping-latency.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'pdf':str(OUTPUT),'points':40,'qualified':evidence['qualified_points']}))


if __name__=='__main__':main()
