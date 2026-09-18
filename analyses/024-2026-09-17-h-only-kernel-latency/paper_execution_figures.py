"""Full-model execution comparisons and the retained instruction ablation cohort."""
from paper_style import *


def quality_latency(data):
    fig,axes=plt.subplots(2,2,figsize=(10.8,7.1),sharey='row')
    fig.subplots_adjust(left=.08,right=.98,bottom=.18,top=.93,hspace=.40,wspace=.22)
    audit={}
    for i,size in enumerate(['14M','70M']):
        rows=cohort(data,size); base=pick(data,size,'0','none',None)
        times=[r['latency_ms'] for r in rows]; span=max(times)-min(times)
        limits=(min(times)-.16*span,max(times)+.18*span)
        front=nondominated(rows,'dense_loss_difference','latency_ms')
        for j,x in enumerate(['sparsity','dense_loss_difference']):
            ax=axes[i,j]
            labels=draw_series(ax,rows,x,'latency_ms',annotate=j==0)
            ax.axhline(base['latency_ms'],color=SCOPE['0'][0],lw=.65,alpha=.7)
            ax.set(ylim=limits,xlabel=SMODEL if j==0 else DL,ylabel='Full-model latency (ms)')
            ax.set_title(f'({chr(97+i*2+j)}) {size}',loc='left')
            ax.margins(x=.16 if j else .08); polish(ax)
            if j==1:
                outline(ax,front,x,'latency_ms')
                # Identify a representative low-loss and low-latency endpoint for each size.
                selected=[min(front,key=lambda p:p['loss']), min(front,key=lambda p:p['latency_ms'])]
                labels=[]
                for r in selected:
                    c,_,name=SCOPE[r['scope']]
                    recipe=name if r['scope'] in ['0','1'] else name+' '+('no pressure' if r['pressure']=='none' else f"OL1({r['pressure']})")
                    label=recipe if r['kappa'] is None else recipe+f", {r['kappa']:g}"
                    labels.append((r[x],r['latency_ms'],label,c))
            label_points(ax,labels)
        audit[size]={'checkpoint_keys':[r['checkpoint_key'] for r in rows],
                     'nondominated_keys':[r['checkpoint_key'] for r in front],
                     'y_limits':limits,'dense_latency_ms':base['latency_ms']}
    legend(fig,frontier=True)
    return save(fig,'05-quality-latency.pdf')|{'coverage':audit}


def speedup(data):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5))
    fig.subplots_adjust(left=.08,right=.985,bottom=.25,top=.90,wspace=.23)
    audit={}
    for ax,size,letter in zip(axes,['14M','70M'],'ab'):
        rows=cohort(data,size); base=pick(data,size,'0','none',None)
        labels=draw_series(ax,rows,'sparsity','dense_speedup',annotate=True)
        ax.axhline(1,color='#777777',lw=.7)
        ax.set(xlabel=SMODEL,ylabel=r'Dense-reference speedup ($\times$)')
        ax.set_title(f'({letter}) {size}',loc='left'); ax.margins(x=.08,y=.19)
        label_points(ax,labels); polish(ax)
        audit[size]={'checkpoint_keys':[r['checkpoint_key'] for r in rows],
                     'dense_reference_implementation':base['kernel'],
                     'dense_key':base['checkpoint_key'],'dense_latency_ms':base['latency_ms'],
                     'dense_timing_session':base['timing_session']}
    legend(fig)
    return save(fig,'A3-dense-reference-speedup.pdf')|{'coverage':audit}


def instruction(data):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.4),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.25,top=.89,wspace=.16)
    index={r['checkpoint_key']:r for r in data['checkpoints']}
    for ax,field,label,letter in zip(axes,
            ['projection_scalar_zero_fraction','projection_mma_bypass_fraction'],
            ['Projection scalar zero-product fraction (%)','Projection MMA bypass (%)'],'ab'):
        for point in data['instruction']:
            row=index[point['checkpoint_key']]; c,marker,_=SCOPE[row['scope']]
            ax.scatter(100*point[field],point['projection_sparse_gain'],color=c,marker=marker,
                       facecolors='white' if row['pressure']=='none' else c,s=31,linewidths=.8,zorder=3)
        ax.axhline(1,color='#777777',lw=.7); ax.set_xlabel(label)
        ax.set_title(f'({letter}) 14M',loc='left'); ax.margins(x=.08,y=.15); polish(ax)
    axes[0].set_ylabel(r'Projection-skipping gain $t_0/t_P$ ($\times$)')
    handles=[Line2D([],[],color=c,marker=m,mfc='white',ls='none',label=l) for c,m,l in SCOPE.values()]
    handles += [Line2D([],[],color='#555555',marker='o',mfc=f,ls='none',label=l)
                for f,l in [('white','No pressure'),('#555555','Pressure (local or all-site)')]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.52,.01))
    return save(fig,'A4-instruction-level-explanation.pdf')|{'checkpoint_keys':[r['checkpoint_key'] for r in data['instruction']],
            'regression':'None; the historical 30-checkpoint cohort is shown without extrapolation.'}
