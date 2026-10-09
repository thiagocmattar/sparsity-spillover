"""Two publication figures; execution colors are shared across both thresholds."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import RUN, read, record, write, verify

COLORS={'native':'#4E79A7','no_skip':'#F28E2B','sparse':'#59A14F'}
def color(mode):return COLORS['native' if mode=='native' else mode.split('/')[1]]
def label(mode):
    if mode=='native':return 'Native dense GEMM'
    name,skip=mode.split('/')
    return name.replace('legacy_','Current ').replace('reuse2_','Reuse2 ').replace('prefetch_','Prefetch ').upper()+(' / skip' if skip=='sparse' else ' / no skip')


def main():
    reduction=read(RUN/'results/comparison.json')
    for row in reduction['sources']:verify(row)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=RUN/'figures';dest.mkdir(exist_ok=True)
    legend=[Patch(color=COLORS[k],label=t) for k,t in [('native','Native dense GEMM'),('no_skip','Custom: MMA bypass disabled'),('sparse','Custom: MMA bypass enabled')]]
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.8))
    for row,kappa in enumerate((0.1,0.5)):
        entries=sorted([r for r in reduction['rows'] if r['head_kappa']==kappa],key=lambda r:(r['mode']!='native',r['mode']))
        for col,boundary in enumerate(('projection_only','gate_cast_projection','full_forward')):
            ax=axes[row,col];values=[r['measurements'][boundary]['host_ms']['geomean_ms']*1000 for r in entries]
            lower=[v-r['measurements'][boundary]['host_ms']['process_min_ms']*1000 for v,r in zip(values,entries)]
            upper=[r['measurements'][boundary]['host_ms']['process_max_ms']*1000-v for v,r in zip(values,entries)]
            y=np.arange(len(entries));ax.barh(y,values,color=[color(r['mode']) for r in entries],height=.68)
            ax.errorbar(values,y,xerr=[lower,upper],fmt='none',ecolor='#222222',capsize=2,linewidth=.8)
            ax.set_yticks(y,[label(r['mode']) for r in entries] if col==0 else ['']*len(entries));ax.invert_yaxis()
            ax.axvline(values[0],color=COLORS['native'],linestyle=':',linewidth=1)
            ax.set_xlim(0,max(values)*1.17);ax.set_xlabel('Host latency (us)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
            for yy,value in zip(y,values):ax.text(value+max(values)*.016,yy,f'{value:.1f}',va='center',fontsize=8)
            title={'projection_only':'Projection only','gate_cast_projection':'Gate + cast + projection','full_forward':'Full forward'}[boundary]
            ax.set_title(f'Head kappa = {kappa} | {title}',fontsize=10)
    fig.suptitle('Qualified head implementations: the dense efficiency gap shrinks, full gains remain modest',fontsize=13,y=.98)
    fig.text(.5,.935,'14M | B1/T2048 | all logits | same gated checkpoints | 3 fresh processes on one RTX 5090',ha='center',fontsize=10)
    fig.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.045),frameon=False,fontsize=9)
    fig.text(.5,.023,'Bars: process geometric means. Whiskers: process min/max, not confidence intervals. Boundaries overlap; do not sum.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.23,right=.985,bottom=.14,top=.875,wspace=.23,hspace=.45)
    final=dest/'01-qualified-boundaries.pdf';fig.savefig(final);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.4),sharey=True)
    variants=[v['name'] for v in read(RUN/'config.json')['variants']]
    for ax,kappa in zip(axes,(0.1,0.5)):
        path=next(p for p in (RUN/'artifacts/screen').glob('*/result.json') if read(p)['qualification'] and f'0p{int(kappa*10)}' in p.parent.name)
        screen=read(path);y=np.arange(len(variants));native=screen['timings']['native']['host_ms']['geomean']*1000
        for shift,mode in ((-.18,'no_skip'),(.18,'sparse')):
            values=[screen['timings'][v+'/'+mode]['host_ms']['geomean']*1000 for v in variants]
            ax.barh(y+shift,values,height=.32,color=COLORS[mode])
        ax.axvline(native,color=COLORS['native'],linestyle=':',linewidth=1.5)
        ax.set_xlim(0,530);ax.set_title(f'Head kappa = {kappa}; native {native:.1f} us');ax.set_xlabel('Projection host latency (us)')
        ax.set_yticks(y,[v.replace('_',' ').upper() for v in variants]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Exploratory screen: reuse helps; direct stores and prefetch do not solve the dense gap',fontsize=12,y=.98)
    fig.text(.5,.925,'14M | 64 real operands x 7 paired passes | one screen process per checkpoint | all trained gates unchanged',ha='center',fontsize=9)
    fig.legend(handles=legend,loc='lower center',ncol=3,bbox_to_anchor=(.5,.005),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.19,right=.98,top=.85,bottom=.15,wspace=.1)
    screen_pdf=dest/'02-exploratory-screen.pdf';fig.savefig(screen_pdf);plt.close(fig)
    write(dest/'provenance.json',{'figures':[record(final),record(screen_pdf)],'qualified_sources':reduction['sources'],
        'screen_sources':[record(p) for p in sorted((RUN/'artifacts/screen').glob('*/result.json'))]})


if __name__=='__main__':main()
