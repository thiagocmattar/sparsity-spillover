"""14M-only frontiers using Analysis041 panel (b)'s linear axes and sweep style."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from evidence import HERE, frontier

COLORS={'base':'#52565C','scalar_other':'#B4BAC1','scalar_t2':'#173FAD','scalar_t7':'#C27539',
        'tile_t2':'#008A79','tile_t7':'#8A669C','head8':'#CC79A7','head16':'#D55E00'}
LABELS={'base':'Base: native/kernel','scalar_other':'Other scalar recipes',
        'scalar_t2':'Scalar T2/Ph','scalar_t7':'Scalar T7 recipes',
        'tile_t2':'T2 corner tiles, head off','tile_t7':'T7 corner tiles, head off',
        'head8':'Head8 tiles (Ph / Ph+head)','head16':'Head16 tiles (Ph+head)'}


def is_native(p):
    mode=p['execution'].lower()
    return ('native' in mode or mode=='dense' or mode=='pytorch') and 'no_skip' not in mode


def scatter(axis, points, edge=True, annotate=False):
    assert all(p['model']=='14M' for p in points)
    for group,scope in [('scalar_t2','hz'),('scalar_t7','7')]:
        sweep=sorted([p for p in points if p['group']==group and p['scope']==scope and
                      p['pressure']=='h' and p['id'].startswith('historical-14M-')],key=lambda p:p['kappa'])
        if sweep:
            assert len(sweep)==5
            axis.plot([p['loss'] for p in sweep],[p['latency_ms'] for p in sweep],color=COLORS[group],
                      ls=(0,(3,2)) if group=='scalar_t2' else '-',lw=1.4,zorder=2)
    base=next((p for p in points if p['group']=='base'),None)
    if base:axis.axvline(base['loss'],color='#A1A5AA',lw=.6,ls=(0,(2,3)),zorder=1)
    for p in sorted(points,key=lambda p:p['group'].startswith('head')):
        color=COLORS[p['group']]
        marker='x' if p['control'] else '*' if p['group']=='head16' else 'D' if p['group'].startswith(('tile','head8')) else 'o'
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
        axis.step([p['loss'] for p in rows],[p['latency_ms'] for p in rows],where='post',color='#656C73',ls=(0,(1.5,2.5)),lw=1.2,zorder=1)
    if annotate:
        selected=[p for p in points if (p['group']=='head16' and p['execution']=='custom_sparse') or
                  (p['session']=='Run064' and p['execution']=='tile_sparse')]
        for p in selected:
            head=p['group']=='head16';k=p['head_kappa'] if head else p['block_kappa']
            label=f"{'Head16' if head else 'T2 tile'} κ={k:g}" + ('' if head else ' (specialized)')
            offset=(12,19 if k==.1 else -22) if head else (22,18 if k==0 else -14)
            axis.annotate(label,(p['loss'],p['latency_ms']),
                xytext=offset,textcoords='offset points',fontsize=8,
                color=COLORS[p['group']],bbox=dict(facecolor='white',edgecolor='none',pad=1),arrowprops=dict(arrowstyle='-',lw=.6,color=COLORS[p['group']]))
    xs=[p['loss'] for p in points];ys=[p['latency_ms'] for p in points]
    ys += [v for p in points for v in p.get('process_latency_ms',[])]
    axis.set(xlabel='Canonical full-validation loss (lower is better)',
             ylabel='Complete forward latency (ms; lower is better)',
             xlim=(min(xs)-.045,max(xs)+.095),ylim=(min(ys)-.025,max(ys)+.08))
    axis.grid(axis='y',alpha=.16);axis.set_axisbelow(True)


def legends(fig):
    handles=[Line2D([],[],color=COLORS[k],marker='D' if k.startswith(('tile','head8')) else '*' if k=='head16' else 'o',ls='none',label=v,ms=7) for k,v in LABELS.items()]
    handles += [Line2D([],[],color=COLORS['scalar_t2'],ls=(0,(3,2)),label='Scalar T2/Ph sweep'),
                Line2D([],[],color=COLORS['scalar_t7'],label='Scalar T7/Ph sweep')]
    handles += [Line2D([],[],color='black',marker='o',mfc='white',ls='none',label='Open: native dense execution'),
                Line2D([],[],color='black',marker='x',ls='none',label='×: no-skip control'),
                Line2D([],[],color='#333333',ls=':',label='Observed point-estimate envelope')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=8,
               bbox_to_anchor=(.5,.05),columnspacing=1.4,handlelength=1.7)


def main():
    data=json.loads((HERE/'data/latency.json').read_text(encoding='utf-8'))
    points=data['points'];(HERE/'figures').mkdir(exist_ok=True)
    assert len(points)==87 and all(p['model']=='14M' for p in points)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axis=plt.subplots(figsize=(10.5,7.0))
    fig.subplots_adjust(left=.095,right=.98,top=.84,bottom=.25)
    scatter(axis,points,annotate=True)
    axis.set_title('14M detail: scalar sweeps + tile/head executions',loc='left',fontsize=11)
    fig.suptitle('14M quality–latency frontier with tile and vocabulary-head interventions',fontsize=14,y=.97)
    fig.text(.5,.895,'RTX 5090 · B1/T2048 · full logits · 87 measured executions · linear axes\n'
             'Style follows Analysis041 panel (b); the latency range includes all new head modes.',ha='center',fontsize=9)
    legends(fig)
    fig.text(.5,.015,'Mixed sessions and BF16/FP16 packages; raw coordinates, no normalization.\n'
             'Whiskers span process means; native/no-skip modes keep trained gates.',ha='center',fontsize=8)
    fig.savefig(HERE/'figures/01-global-quality-latency.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    cohorts=list(data['compatible_frontier_ids'])
    fig,axes=plt.subplots(1,2,figsize=(14,6.8));fig.subplots_adjust(left=.065,right=.985,top=.82,bottom=.25,wspace=.24)
    for ax,cohort in zip(axes,cohorts):
        subset=[p for p in points if p['execution_cohort']==cohort]
        scatter(ax,subset,annotate='FP16' in cohort)
        ax.set_title(f'14M: {cohort} — detail view\n{len(subset)} points; {len(frontier(subset))} nondominated',loc='left',fontsize=10)
    fig.suptitle('14M frontiers within the declared execution cohorts',fontsize=14,y=.97)
    fig.text(.5,.89,'BF16 block-kernel and FP16 native-body/head measurements have separate envelopes\nDifferent sessions remain descriptive within each panel; panels use their own labeled scales.',ha='center',fontsize=9)
    legends(fig)
    fig.savefig(HERE/'figures/02-execution-cohort-frontiers.pdf',metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    print({'status':'frontiers-plotted','figures':2},flush=True)


if __name__=='__main__':
    main()
