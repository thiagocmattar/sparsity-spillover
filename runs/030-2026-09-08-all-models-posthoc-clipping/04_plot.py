"""Full-range views of every retained training-plus-clipping trajectory."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
STYLE={'A0':('#444444','o'),'A1-H':('#888888','s'),
       'A1-H-L1':('#009E73','v'),'A1-H-OL1':('#CC79A7','^'),
       'A4':('#0072B2','D'),'A4-OL1':('#0072B2','P'),
       'A7':('#D55E00','X'),'A7-OL1':('#D55E00','*')}


def main():
    data=json.loads((HERE/'results/clipping-points.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,
                         'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.axisbelow':True,'axes.linewidth':.6})
    (HERE/'figures').mkdir(exist_ok=True)
    for scale in ('14M','70M','410M'):
        points=[r for r in data['points'] if r['scale']==scale]
        trained=[r for r in data['trained'] if r['scale']==scale]
        fig,ax=plt.subplots(figsize=(5.5,4.6))
        fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.91)
        for source in trained:
            group=sorted([r for r in points if r['source_checkpoint_id']==source['id']],key=lambda r:r['dose'])
            color,marker=STYLE[source['family']]
            ax.plot([100*r['R_model'] for r in group],[r['loss'] for r in group],
                    color=color,marker=marker,ms=2.2,mfc='white',mew=.45,lw=.5,alpha=.45)
        legend=[]
        for family,(color,marker) in STYLE.items():
            group=sorted([r for r in trained if r['family']==family],
                         key=lambda r:r['training_parameter'] or 0)
            if not group: continue
            ls='--' if family.endswith(('-L1','-OL1')) else '-'
            ax.plot([100*r['R_model'] for r in group],[r['loss'] for r in group],
                    color=color,marker=marker,ms=4,mew=.65,lw=1.15,ls=ls,zorder=4)
            legend.append(Line2D([],[],color=color,marker=marker,ms=4,lw=1.15,
                                 ls=ls if len(group)>1 else 'none',label=family))
        ax.set(xlabel=r'$\mathcal{S}_{\mathrm{model}}$ (%)',ylabel='Validation loss')
        ax.grid(axis='both',color='.88',lw=.4)
        fig.suptitle(f'{scale} training and post-hoc clipping',y=.985,fontsize=11)
        fig.legend(handles=legend,loc='upper center',ncol=4,frameon=False,fontsize=7.5,
                   columnspacing=1,handletextpad=.45,handlelength=2,bbox_to_anchor=(.55,.145))
        key=Line2D([],[],color='.55',marker='o',mfc='white',mew=.45,ms=2.5,lw=.5,
                   label='Post-hoc clipping (thin lines, open markers)')
        fig.legend(handles=[key],loc='lower center',frameon=False,fontsize=7.5,
                   handlelength=2.5,bbox_to_anchor=(.55,.004))
        fig.savefig(HERE/'figures'/f'{scale.lower()}-posthoc-frontiers.pdf',
                    metadata={'Creator':'Run 030','CreationDate':None,'ModDate':None})
        plt.close(fig)


if __name__=='__main__':
    main()
