"""One PDF showing direct matched Ph/P2 latency and process variability."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io_utils import RUN,read,write,record,verify


def main():
    result=read(RUN/'results/results.json')
    for pair in result['pairs']:
        for source in pair['source_files']:verify(source)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.3),gridspec_kw={'width_ratios':[1.2,1]})
    colors={'ph':'#276587','p2':'#D57935'};x=np.arange(2);width=.31
    for pressure,offset in [('ph',-.5),('p2',.5)]:
        heights=[pair['geomean_host_ms'][pressure+'_candidate'] for pair in result['pairs']]
        bars=axes[0].bar(x+offset*width,heights,width,color=colors[pressure],label='Ph' if pressure=='ph' else 'P2')
        for bar,value in zip(bars,heights):
            axes[0].annotate(f'{value:.3f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set_xticks(x,['0','0.1']);axes[0].set_xlabel('Threshold kappa')
    axes[0].set_ylabel('Full-model host latency (ms; geometric mean)')
    axes[0].set_title('Frozen K050: lower latency is faster',fontsize=10)
    axes[0].set_ylim(0,max(pair['geomean_host_ms'][pressure+'_candidate'] for pair in result['pairs'] for pressure in ['ph','p2'])*1.28)
    axes[0].legend(frameon=False,loc='upper right')
    for index,pair in enumerate(result['pairs']):
        values=[process['speedup_p2_over_ph'] for process in pair['processes']]
        accepted=pair['qualified'];color='#276587' if accepted else '#B54840'
        axes[1].scatter(np.linspace(index-.075,index+.075,len(values)),values,s=28,color=color,label='Fresh processes' if index==0 else None,zorder=3)
        axes[1].scatter(index,pair['unqualified_or_qualified_timing_ratio'],marker='D',s=42,facecolor='white',edgecolor=color,zorder=4,label='Pooled paired ratio' if index==0 else None)
        text=f"{pair['unqualified_or_qualified_timing_ratio']:.3f}x"
        if not accepted:text+='\nunqualified'
        axes[1].annotate(text,(index,max(values)),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,color=color)
    axes[1].axhline(1,color='#666666',linestyle='--',linewidth=1)
    ratios=[process['speedup_p2_over_ph'] for pair in result['pairs'] for process in pair['processes']]
    low,high=min([1]+ratios),max([1]+ratios);padding=max(.025,(high-low)*.6)
    axes[1].set_ylim(low-padding*.5,high+padding)
    axes[1].set_xlim(-.45,1.45);axes[1].set_xticks(x,['0','0.1']);axes[1].set_xlabel('Threshold kappa')
    axes[1].set_ylabel('Paired speedup: Ph latency / P2 latency')
    axes[1].set_title('Above 1 means P2 is faster',fontsize=10)
    axes[1].legend(frameon=False,loc='lower center',fontsize=8)
    for axis in axes:axis.set_axisbelow(True);axis.grid(axis='y',alpha=.15)
    fig.suptitle('14M T2: does pressure at H/Z reduce inference latency?',fontsize=12,y=.98)
    fig.subplots_adjust(left=.09,right=.98,bottom=.24,top=.84,wspace=.36)
    qualification='All numerical checks passed.' if all(pair['qualified'] for pair in result['pairs']) else 'Unqualified kernel timings cannot support an accepted speedup claim.'
    fig.text(.5,.09,'RTX 5090 | BF16 | batch 1 | 2,048-token uncached causal inference | all 50,304 logits',ha='center',fontsize=8)
    fig.text(.5,.045,'64 inputs x 7 paired passes x 3 fresh processes per threshold. '+qualification,ha='center',fontsize=8)
    output=RUN/'figures/01-p2-ph-latency.pdf';output.parent.mkdir(exist_ok=True)
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    write(RUN/'results/figure-source.json',{'figure':record(output),'source_script':record(Path(__file__)),
        'reduction':record(RUN/'results/results.json'),'visual_review':'pending'})
    print(output)


if __name__=='__main__':main()
