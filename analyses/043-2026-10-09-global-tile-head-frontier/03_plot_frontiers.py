"""Global and execution-cohort Pareto views from unchanged measured coordinates."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evidence import HERE, frontier

COLORS={'base':'#555555','scalar_other':'#B4BAC1','scalar_t2':'#0072B2','scalar_t7':'#E69F00',
        'tile_t2':'#009E73','tile_t7':'#8A669C','head8':'#CC79A7','head16':'#D55E00'}
LABELS={'base':'Base: native/kernel','scalar_other':'Other scalar recipes',
        'scalar_t2':'Scalar T2/Ph','scalar_t7':'Scalar T7 recipes',
        'tile_t2':'T2 corner tiles, head off','tile_t7':'T7 corner tiles, head off',
        'head8':'Head8 tiles (Ph / Ph+head)','head16':'Head16 tiles (Ph+head)'}


def is_native(p):
    mode=p['execution'].lower()
    return ('native' in mode or mode=='dense' or mode=='pytorch') and 'no_skip' not in mode


def scatter(axis, points, edge=True, annotate=False):
    for p in sorted(points,key=lambda p:p['group'].startswith('head')):
        color=COLORS[p['group']]
        marker='x' if p['control'] else 'D' if p['group']=='head8' else '*' if p['group']=='head16' else {'14M':'o','31M':'^','70M':'s'}[p['model']]
        size=85 if marker=='*' else 38 if p['group'].startswith('head') else 26
        face='white' if is_native(p) else color
        if p['control']:
            axis.scatter(p['loss'],p['latency_ms'],c=color,marker=marker,s=size*.65,linewidths=.8,alpha=.55,zorder=2)
        else:
            axis.scatter(p['loss'],p['latency_ms'],facecolor=face,edgecolor=color,marker=marker,
                         s=size,linewidths=.9,alpha=.65 if p['group']=='scalar_other' else .95,zorder=5)
        if p.get('process_latency_ms') and not p['control'] and p['group'] in ('head8','head16','tile_t2','tile_t7'):
            values=p['process_latency_ms']
            axis.vlines(p['loss'],min(values),max(values),color=color,lw=.65,alpha=.7,zorder=3)
    if edge:
        rows=frontier(points)
        axis.plot([p['loss'] for p in rows],[p['latency_ms'] for p in rows],color='#333333',ls=':',lw=1.4,zorder=1)
    if annotate:
        selected=[p for p in points if p['group']=='head16' and p['execution']=='custom_sparse']
        for p in selected:
            axis.annotate(f"Head16 κ={p['head_kappa']:g}",(p['loss'],p['latency_ms']),
                xytext=(8,15 if p['head_kappa']==.1 else -22),textcoords='offset points',fontsize=8,
                color=COLORS['head16'],bbox=dict(facecolor='white',edgecolor='none',pad=1),arrowprops=dict(arrowstyle='-',lw=.6,color=COLORS['head16']))
    xs=[p['loss'] for p in points];ys=[p['latency_ms'] for p in points]
    axis.set(xlabel='Canonical full-validation loss (lower is better)',
             ylabel='Complete forward latency (ms; lower is better)',
             xlim=(min(xs)-.06,max(xs)+.08),ylim=(max(.01,min(ys)*.87),max(ys)*1.12))
    axis.grid(axis='y',alpha=.16);axis.set_axisbelow(True)


def legends(fig):
    handles=[Line2D([],[],color=COLORS[k],marker='D' if k=='head8' else '*' if k=='head16' else 'o',ls='none',label=v,ms=7) for k,v in LABELS.items()]
    handles+=[Line2D([],[],color='black',marker=m,ls='none',label=s,ms=6) for s,m in [('14M','o'),('31M','^'),('70M','s')]]
    handles += [Line2D([],[],color='black',marker='o',mfc='white',ls='none',label='Open: native dense execution'),
                Line2D([],[],color='black',marker='x',ls='none',label='×: no-skip control'),
                Line2D([],[],color='#333333',ls=':',label='Observed point-estimate envelope')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=8,
               bbox_to_anchor=(.5,.008),columnspacing=1.4,handlelength=1.7)


def main():
    data=json.loads((HERE/'data/latency.json').read_text(encoding='utf-8'))
    points=data['points'];(HERE/'figures').mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,6.8))
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.25,wspace=.23)
    scatter(axes[0],points);scatter(axes[1],[p for p in points if p['model']=='14M'],annotate=True)
    axes[0].set_title('(a) All 138 existing execution points',loc='left',fontsize=11)
    axes[1].set_title('(b) 14M detail — all measured head modes retained',loc='left',fontsize=11)
    fig.suptitle('Global quality–latency frontier with tile and vocabulary-head interventions',fontsize=14,y=.97)
    fig.text(.5,.91,'RTX 5090 · B1/T2048 · full logits · mixed sessions and execution packages; no latency rescaling',ha='center',fontsize=9)
    legends(fig)
    fig.savefig(HERE/'figures/01-global-quality-latency.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    cohorts=list(data['compatible_frontier_ids'])
    fig,axes=plt.subplots(1,2,figsize=(14,6.8));fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.25,wspace=.24)
    for ax,cohort in zip(axes,cohorts):
        subset=[p for p in points if p['execution_cohort']==cohort]
        scatter(ax,subset,annotate='FP16' in cohort)
        ax.set_title(f'{cohort} — detail view\n{len(subset)} points; {len(frontier(subset))} nondominated',loc='left',fontsize=10)
    fig.suptitle('Frontiers within the declared execution cohorts',fontsize=14,y=.97)
    fig.text(.5,.89,'BF16 block-kernel and FP16 native-body/head measurements have separate envelopes\nDifferent sessions remain descriptive within each panel; panels use their own labeled scales.',ha='center',fontsize=9)
    legends(fig)
    fig.savefig(HERE/'figures/02-execution-cohort-frontiers.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    print({'status':'frontiers-plotted','figures':2},flush=True)


if __name__=='__main__':
    main()
