"""Main quality comparison and complete quality appendix."""
from paper_style import *

def make(data,full=False):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.25,top=.88,wspace=.16)
    y='loss' if full else 'dense_loss_difference'
    values=[r[y] for size in ['14M','70M'] for r in cohort(data,size)]
    if full: values += [r[y] for r in data['clipping']]
    span=max(values)-min(values)
    limits=(min(values)-.07*span,max(values)+.14*span)
    audit={}
    for size,ax,letter in zip(['14M','70M'],axes,'ab'):
        rows=cohort(data,size)
        clipping=[p for p in data['clipping'] if p['model']==size and (full or p['main_clipping'])]
        for key in dict.fromkeys(p['checkpoint_key'] for p in clipping):
            group=sorted([p for p in clipping if p['checkpoint_key']==key],key=lambda p:p['target'])
            first=group[0]; color,marker,_=SCOPE[first['scope']]
            control=first['scope'] in ['0','1']
            ax.plot([p['sparsity'] for p in group],[p[y] for p in group],ls=':',color=color,
                    lw=1 if control else .6,alpha=.85 if control else .32,
                    marker=marker if control else '.',ms=3 if control else 1.8,
                    mfc='white' if first['pressure']=='none' else color,mew=.65,zorder=2)
        labels=draw_series(ax,rows,'sparsity',y,annotate=not full)
        ax.set(xlim=(-1,52 if size=='70M' else 32),ylim=limits,xlabel=SMODEL)
        ax.set_title(f'({letter}) {size}',loc='left',pad=10)
        if not full: ax.axhline(0,color='#777777',lw=.7,zorder=1)
        for scope in ['4','7']:
            c=data['coverage'][size][scope]['R_model_max_percent']
            ax.axvline(c,color=SCOPE[scope][0],ls=(0,(1,3)),lw=.7,alpha=.55,zorder=1)
            ax.text(c,1.025,f'{scope}-site coverage',transform=ax.get_xaxis_transform(),
                    color=SCOPE[scope][0],fontsize=6.8,va='bottom',ha='right')
        front=nondominated(rows+clipping,'sparsity',y,maximize_x=True)
        if not full:
            visible=[p for p in front if limits[0]<=p[y]<=limits[1]]
            outline(ax,visible,'sparsity',y)
            label_points(ax,labels)
        audit[size]={'trained_keys':[p['checkpoint_key'] for p in rows],
                     'clipping_ids':[p['id'] for p in clipping],
                     'outside_main_y':[p['id'] for p in clipping if not limits[0]<=p[y]<=limits[1]],
                     'nondominated_ids':[p.get('id',p['checkpoint_key']) for p in front]}
        polish(ax)
    axes[0].set_ylabel('Validation loss (nats/token)' if full else DL)
    legend(fig,clipping=True,frontier=not full)
    name='A1-complete-quality-sparsity.pdf' if full else '01-quality-sparsity-tradeoffs.pdf'
    return save(fig,name)|{'coverage':audit,'y_limits':limits}
