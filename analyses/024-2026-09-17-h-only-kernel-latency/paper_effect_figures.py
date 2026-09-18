"""Explicit threshold contrasts and count-based operation contributions."""
from paper_style import *

def difference(data,size,scope,k,treatment,reference):
    a,b=pick(data,size,scope,treatment,k),pick(data,size,scope,reference,k)
    return {'model':size,'scope':scope,'kappa':k,'treatment_key':a['checkpoint_key'],
            'reference_key':b['checkpoint_key'],'loss':a['loss']-b['loss'],
            'sparsity':a['sparsity']-b['sparsity'],'latency_us':1000*(a['latency_ms']-b['latency_ms']),
            'sessions':[a['timing_session'],b['timing_session']]}

def pressure_scope(data):
    # All-site minus h-only pressure, matched within size, topology, and kappa.
    styles=[('4','#3679AD'),('7','#C27539')]
    models=['14M','70M']
    pairs=[difference(data,m,s,k,'all','h')|{'treatment':'all','reference':'h'}
           for m in models for s,_ in styles for k in KAPPAS]
    title=r'Global ($P_{\mathrm{all}}$) vs. Local ($P_h$) Pressure Paired Analysis'
    fig,axes=plt.subplots(2,3,figsize=(11.52,5.8),sharex=True)
    # Keep comparable loss/sparsity scales; latency needs a separate scale per size.
    for j in [0,1]: axes[1,j].sharey(axes[0,j])
    fig.subplots_adjust(left=.058/.9,right=1-.015/.9,bottom=.16,top=.87,hspace=.55,wspace=.34)
    fig.suptitle(title,fontsize=14,y=.985)
    series=[]
    for s,color in styles:
        label=rf'$T_{s}/P_{{\mathrm{{all}}}} - T_{s}/P_h$'
        for i,m in enumerate(models):
            rows=[r for r in pairs if (r['model'],r['scope'])==(m,s)]
            for ax,metric in zip(axes[i],['loss','sparsity','latency_us']):
                ax.plot(range(5),[r[metric] for r in rows],color=color,ls='-',lw=1.6,
                        marker='o',ms=5.8,mfc=color,mec='white',mew=.7,zorder=4,label=label)
        series.append({'scope':s,'treatment':'all','reference':'h','label':label,
                       'color':color,'linestyle':'-','marker':'o'})
    for i,m in enumerate(models):
        for j,(metric,label,comparison) in enumerate([
                ('loss',r'Validation loss change $\Delta L$','paired loss difference'),
                ('sparsity',r'Sparsity change $\Delta S_{\mathrm{model}}$ (pp)','paired sparsity difference'),
                ('latency_us',r'Latency change $\Delta t$ ($\mu$s)','paired latency difference')]):
            ax=axes[i,j]
            vals=[p[metric] for p in pairs if metric!='latency_us' or p['model']==m]+[0]
            lo,hi=min(vals),max(vals); pad=.12*(hi-lo)
            ax.set_ylim(lo-pad,hi+pad); ax.axhline(0,color='#777777',lw=.7)
            if metric=='loss': ax.set_yticks([-.1,0,.1,.2,.3])
            if metric=='latency_us': ax.set_yticks([0,20,40,60] if m=='14M' else [0,250,500,750,1000])
            ax.set_title(f'({chr(97+3*i+j)}) {m}: {comparison}',loc='left',pad=11,fontsize=11.5)
            ax.set_ylabel(label,fontsize=11); categorical(ax)
            ax.set_xlabel(r'Threshold $\kappa$',fontsize=11)
            ax.tick_params(labelsize=10,length=3,width=.65)
            for spine in ax.spines.values(): spine.set_linewidth(.65)
            ax.grid(axis='y',color='#E8EAED',lw=.6)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,
               fontsize=11,handlelength=2.3,columnspacing=2.3,
               bbox_to_anchor=(.53,.018))
    return save(fig,'03-pressure-scope-threshold.pdf')|{
        'pairs':pairs,'series':series,'title':title,
        'layout':{'rows':2,'columns':3,'width_inches':11.52,'height_inches':5.8},
        'typography_pt':{'title':14,'panel_titles':11.5,'axes':11,'ticks':10,'legend':11},
        'panel_nomenclature':'Paired loss difference / paired sparsity difference / paired latency difference',
        'latency_estimand':'Difference of retained full-model geometric mean latencies, in microseconds',
        'y_scales':{'loss':'shared across sizes','sparsity':'shared across sizes','latency_us':'independent per size'},
        'reference':'OL1(h) at the same model size, threshold topology, and kappa',
        'style_reference':'14_plot_14m_quality_latency.py',
        'models':models}

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
    # Same operation order and muted palette as the manuscript's site-structure figure.
    operations=[('qkv_projection','QKV','#9eaec2'),('qk_scores','QK','#df985c'),
                ('probability_value','PV','#f0c68b'),
                ('attention_output_projection','Attention output','#76969c'),
                ('mlp_w1','FFN up','#817ca9'),('mlp_w2','FFN down','#b5acd0')]
    title='Operation Contributions to Model-wide Sparsity on Pythia-14M'
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.6),sharey=True)
    fig.subplots_adjust(left=.09,right=.985,bottom=.23,top=.81,wspace=.20)
    fig.suptitle(title,fontsize=14,y=.975)
    recipes=[(s,p) for s in ['4','7'] for p in ['none','h','all']]
    positions=[0,1,2,3.6,4.6,5.6]
    pressure_labels={'none':'0','h':'h','all':r'\mathrm{all}'}
    labels=[rf'$T_{s}/P_{{{pressure_labels[p]}}}$' for s,p in recipes]
    records=[]
    for ax,k,letter in zip(axes,[.05,.5],'ab'):
        for x,(s,p) in zip(positions,recipes):
            row=pick(data,'14M',s,p,k)
            den=row['counts']['model_product_count']; bottom=0.; components={}
            for op,_,color in operations:
                count=row['counts']['per_operation'][op]['zero_product_count']
                value=100*count/den
                components[op]={'zero_product_count':count,'model_denominator':den,'contribution_pp':value}
                ax.bar(x,value,bottom=bottom,color=color,width=.72,edgecolor='white',lw=.4,zorder=3)
                bottom+=value
            assert sum(c['zero_product_count'] for c in components.values())==row['counts']['block_zero_product_count']
            assert abs(bottom-row['sparsity'])<1e-12
            records.append({'model':'14M','kappa':k,'scope':s,'pressure':p,
                            'checkpoint_key':row['checkpoint_key'],'components':components,'total_pp':bottom})
        ax.set_xticks(positions,labels)
        ax.set_xlim(-.65,6.25)
        ax.set_title(f'({letter}) '+rf'Threshold $\kappa={k:g}$',loc='left',pad=11,fontsize=11.5)
        ax.set_xlabel('Threshold / pressure recipe',fontsize=11,labelpad=9)
        ax.tick_params(labelsize=10,length=3,width=.65)
        for spine in ax.spines.values(): spine.set_linewidth(.65)
        ax.grid(axis='y',color='#E8EAED',lw=.6,zorder=0)
    axes[0].set_ylim(0,30)
    axes[0].set_yticks(range(0,31,5))
    axes[0].set_ylabel('Contribution to model-wide\nsparsity (pp)',fontsize=11)
    handles=[plt.Rectangle((0,0),1,1,color=c,label=label) for _,label,c in operations]
    fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,fontsize=11,
               handlelength=1.2,columnspacing=1.5,bbox_to_anchor=(.53,.015))
    return save(fig,'04-operation-sparsity-changes.pdf')|{
        'checkpoints':records,'title':title,
        'estimand':'100 * pooled operation zero-product count / full-model product count; absolute contribution in percentage points',
        'layout':{'rows':1,'columns':2,'width_inches':10.8,'height_inches':4.6,'shared_y_limits':[0,30]},
        'typography_pt':{'title':14,'panel_titles':11.5,'axes':11,'ticks':10,'legend':11},
        'palette':[{'operation':op,'label':label,'color':c} for op,label,c in operations],
        'palette_reference':'manuscript/draft/figures/pressure-scope/05-site-structure.pdf',
        'style_reference':'03-pressure-scope-threshold.pdf'}

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
