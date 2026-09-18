"""Explicit threshold contrasts and count-based operation contributions."""
from paper_style import *
from paper_evidence import OPS

def difference(data,size,scope,k,treatment,reference):
    a,b=pick(data,size,scope,treatment,k),pick(data,size,scope,reference,k)
    return {'model':size,'scope':scope,'kappa':k,'treatment_key':a['checkpoint_key'],
            'reference_key':b['checkpoint_key'],'loss':a['loss']-b['loss'],
            'sparsity':a['sparsity']-b['sparsity'],'latency_us':1000*(a['latency_ms']-b['latency_ms']),
            'sessions':[a['timing_session'],b['timing_session']]}

def pressure_scope(data):
    pairs=[difference(data,m,s,k,'all','h') for m in ['14M','70M'] for s in ['4','7'] for k in KAPPAS]
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.2),sharex=True,sharey='col')
    fig.subplots_adjust(left=.09,right=.985,bottom=.16,top=.92,hspace=.31,wspace=.28)
    for i,m in enumerate(['14M','70M']):
        for j,(metric,label) in enumerate([('loss','Loss change: OL1(all) − OL1(h)\n(nats/token)'),
                                           ('sparsity','Sparsity change: OL1(all) − OL1(h) (pp)')]):
            ax=axes[i,j]
            for s in ['4','7']:
                rows=[p for p in pairs if p['model']==m and p['scope']==s]
                c,marker,_=SCOPE[s]
                ax.plot(range(5),[p[metric] for p in rows],color=c,marker=marker,ls=LINES['all'],ms=5,lw=1)
            vals=[p[metric] for p in pairs]+[0]; lo,hi=min(vals),max(vals); pad=.21*(hi-lo)
            ax.set_ylim(lo-pad,hi+pad); ax.axhline(0,color='#777777',lw=.7)
            ax.set_title(f'({chr(97+2*i+j)}) {m}',loc='left'); ax.set_ylabel(label); categorical(ax)
            if i==1:
                for k,idx in [(.05,2),(.5,4)]:
                    p=next(p for p in pairs if p['model']==m and p['scope']=='7' and p['kappa']==k)
                    text=f"{p[metric]:+.3f}"+(' nats' if metric=='loss' else ' pp')
                    ax.annotate(text,(idx,p[metric]),xytext=(-8,13 if p[metric]>=0 else -19),textcoords='offset points',
                                ha='right',fontsize=7,color=SCOPE['7'][0],bbox={'fc':'white','ec':'none','pad':.5})
    fig.legend(handles=[Line2D([],[],color=SCOPE[s][0],marker=SCOPE[s][1],label=SCOPE[s][2]) for s in ['4','7']],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.01))
    for k,loss,sparsity in [(.05,.297,-.371),(.5,-.121,8.369)]:
        p=next(p for p in pairs if p['model']=='70M' and p['scope']=='7' and p['kappa']==k)
        assert round(p['loss'],3)==loss and round(p['sparsity'],3)==sparsity
    return save(fig,'03-pressure-scope-threshold.pdf')|{'pairs':pairs}

def pressure_none(data):
    pairs=[]
    fig,axes=plt.subplots(2,3,figsize=(11.5,6),sharex=True,sharey='col')
    fig.subplots_adjust(left=.08,right=.985,bottom=.15,top=.91,hspace=.38,wspace=.32)
    for i,pressure in enumerate(['h','all']):
        rows=[difference(data,'14M',s,k,pressure,'none')|{'pressure':pressure} for s in ['4','7'] for k in KAPPAS]
        pairs+=rows
        for j,(metric,label) in enumerate([('loss','Loss change (nats/token)'),('sparsity','Sparsity change (pp)'),('latency_us','Latency change (µs)')]):
            ax=axes[i,j]
            for s in ['4','7']:
                ps=[p for p in rows if p['scope']==s]; c,marker,_=SCOPE[s]
                ax.plot(range(5),[p[metric] for p in ps],color=c,marker=marker,ls=LINES[pressure],ms=5,lw=1)
            ax.axhline(0,color='#777777',lw=.7); ax.set_ylabel(label)
            ax.set_title(f"({chr(97+3*i+j)}) OL1({pressure}) − no pressure",loc='left',fontsize=9)
            categorical(ax); ax.margins(y=.17)
    fig.legend(handles=[Line2D([],[],color=SCOPE[s][0],marker=SCOPE[s][1],label=SCOPE[s][2]) for s in ['4','7']],
               loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.52,.005))
    return save(fig,'A2-pressure-versus-none.pdf')|{'pairs':pairs}

def operation_changes(data):
    palette=['#4477AA','#66CCEE','#228833','#CCBB44','#EE6677','#AA3377']
    labels=['QKV projection','FFN-up','FFN-down','Attention-output','QK','PV']
    fig,axes=plt.subplots(1,2,figsize=(10.6,4.5))
    fig.subplots_adjust(left=.08,right=.985,bottom=.30,top=.90,wspace=.25)
    records=[]
    for ax,k,letter in zip(axes,[.05,.5],'ab'):
        for x,(s,p,q) in enumerate([('4','h','none'),('4','all','h'),('7','h','none'),('7','all','h')]):
            a,b=pick(data,'14M',s,p,k),pick(data,'14M',s,q,k)
            assert a['counts']['model_product_count']==b['counts']['model_product_count']
            den=a['counts']['model_product_count']; positive=negative=0.; components={}
            for op,c in zip(OPS,palette):
                delta=a['counts']['per_operation'][op]['zero_product_count']-b['counts']['per_operation'][op]['zero_product_count']
                value=100*delta/den; components[op]={'zero_product_difference':delta,'model_denominator':den,'contribution_pp':value}
                ax.bar(x,value,bottom=positive if value>=0 else negative,color=c,width=.6,edgecolor='white',lw=.3)
                if value>=0: positive+=value
                else: negative+=value
            net=sum(v['contribution_pp'] for v in components.values())
            assert abs(net-(a['sparsity']-b['sparsity']))<1e-12
            ax.scatter([x],[net],color='black',marker='D',s=24,zorder=5)
            records.append({'kappa':k,'scope':s,'treatment':p,'reference':q,'treatment_key':a['checkpoint_key'],
                            'reference_key':b['checkpoint_key'],'components':components,'net_pp':net})
        ax.set_xticks(range(4),[r'$T_4:P_h-P_0$',r'$T_4:P_4-P_h$',r'$T_7:P_h-P_0$',r'$T_7:P_7-P_h$'],rotation=15)
        ax.axhline(0,color='#555555',lw=.75); ax.set_title(f'({letter}) 14M, '+rf'$\kappa={k:g}$',loc='left')
        # Bar sticky edges can clip the extremum of a diverging stack; derive limits explicitly.
        stacks=[r['components'] for r in records if r['kappa']==k]
        lo=min(sum(min(0,c['contribution_pp']) for c in stack.values()) for stack in stacks)
        hi=max(sum(max(0,c['contribution_pp']) for c in stack.values()) for stack in stacks)
        pad=.12*(hi-lo)
        ax.set_ylim(lo-pad,hi+pad); polish(ax)
    axes[0].set_ylabel('Change in contribution to\nmodel-wide sparsity (pp)')
    handles=[plt.Rectangle((0,0),1,1,color=c,label=l) for c,l in zip(palette,labels)]
    handles+=[Line2D([],[],color='black',marker='D',ls='none',label='Net change')]
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=8,bbox_to_anchor=(.53,.01))
    return save(fig,'04-operation-sparsity-changes.pdf')|{'contrasts':records}

def table2(data):
    records=[]
    for k in KAPPAS:
        row={'kappa':k}
        for m in ['14M','70M']:
            a,b=pick(data,m,'7','h',k),pick(data,m,'4','h',k)
            row[m]={'seven_key':a['checkpoint_key'],'four_key':b['checkpoint_key'],
                    'sparsity_pp':a['sparsity']-b['sparsity'],'loss_nats':a['loss']-b['loss']}
        records.append(row)
    output=['# Table 2: Threshold placement at fixed h-only pressure','',
            'Seven-site minus four-site, with OL1(h) held fixed. One final checkpoint per trained setting.', '',
            '| kappa | 14M: ΔS (pp) | 14M: ΔL (nats/token) | 70M: ΔS (pp) | 70M: ΔL (nats/token) |',
            '|---|---:|---:|---:|---:|']
    for r in records:
        output.append(f"| {r['kappa']:g} | {r['14M']['sparsity_pp']:+.4f} | {r['14M']['loss_nats']:+.4f} | {r['70M']['sparsity_pp']:+.4f} | {r['70M']['loss_nats']:+.4f} |")
    output+=['','Sources and exact checkpoint pairs: `data/paper-derived.json`. Positive loss changes are worse; positive sparsity changes mean more logical zero products. These five settings are not independent seeds.']
    (HERE/'TABLE-2-threshold-placement.md').write_text('\n'.join(output)+'\n',encoding='utf-8',newline='\n')
    return records
