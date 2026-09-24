"""Plot only the approved Base/T2/Ph/T7/Ph scale comparison, without fitting."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter,NullLocator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'analyses/037-2026-09-24-14m-31m-70m-latency-quality'
OLD_HASH='5d6d88d24d32ff9ebadc76a2653fed1b517becedf06beda1cecbdd84bc27f55a'
TITLE='Quality\u2013latency trade-offs across model scales'
STYLE_REFERENCE=ROOT/'analyses/034-2026-09-21-70m-latency-quality-frontier/figures/02-14m-70m-latency-quality.pdf'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    old_path=OLD/'data/combined-figure.json'
    assert sha(old_path)==OLD_HASH
    old=json.loads(old_path.read_text())
    old_pdf=OLD/old['output'];assert sha(old_pdf)==old['output_sha256']
    new_path=HERE/'data/31m-t7-results.json';new=json.loads(new_path.read_text())
    for name,digest in new['sources_sha256'].items():assert sha(ROOT/name)==digest,name
    families=[('hz','h'),('7','h')]
    cohorts={s:[p for p in rows if (p['scope'],p['pressure']) in families] for s,rows in old['cohorts'].items()}
    cohorts['31M']+=new['points']
    fields={'14M':'latency_ms','31M':'latency_ms','70M':'displayed_latency_ms'}
    refs=old['base_references']
    executions=[]
    for scale,rows in cohorts.items():
        for family in families:
            group=[p for p in rows if (p['scope'],p['pressure'])==family]
            assert sorted(p['kappa'] for p in group)==[0.,.01,.05,.1,.5]
        for p in rows:
            executions.append(dict(model=scale,scope=p['scope'],pressure=p['pressure'],kappa=p['kappa'],
                loss=p['loss'],latency_ms=p[fields[scale]],checkpoint_key=p['checkpoint_key'],backend='kernel'))
    executions += [dict(p,scope='0',pressure='none',kappa=None) for p in refs]
    assert len(executions)==36 and len({p['checkpoint_key'] for p in executions})==33
    frontier=sorted([p for p in executions if not any(q['loss']<=p['loss'] and q['latency_ms']<=p['latency_ms']
        and (q['loss']<p['loss'] or q['latency_ms']<p['latency_ms']) for q in executions)],key=lambda p:p['loss'])
    markers={'14M':'o','31M':'^','70M':'s'}
    colors={'hz':'#173FAD','7':'#C27539'}
    linestyles={'hz':(0,(3,2)),'7':'-'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':12,
        'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,ax=plt.subplots(figsize=(8.6,5.2));fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.90)
    for scale,marker in markers.items():
        for scope,_ in families:
            points=sorted([p for p in executions if p['model']==scale and p['scope']==scope],key=lambda p:p['kappa'])
            x=[p['loss'] for p in points];y=[p['latency_ms'] for p in points]
            ax.plot(x,y,color=colors[scope],ls=linestyles[scope],lw=1.6,marker=marker,ms=5.8,
                mec='white',mew=.7,zorder=4)
    base_loss_guides={}
    for scale in markers:
        losses={p['loss'] for p in refs if p['model']==scale}
        assert len(losses)==1
        base_loss_guides[scale]=losses.pop()
        ax.axvline(base_loss_guides[scale],color='#92969B',lw=.8,ls=(0,(2,3)),zorder=1)
        ax.text(base_loss_guides[scale],.018,scale,transform=ax.get_xaxis_transform(),
                ha='center',va='bottom',fontsize=9,color='#65696F',
                bbox=dict(facecolor='white',edgecolor='none',pad=1.5),zorder=8)
    for backend in ('kernel','PyTorch'):
        points=sorted([p for p in refs if p['backend']==backend],key=lambda p:p['latency_ms'])
        for p in points:
            kernel=backend=='kernel'
            ax.plot(p['loss'],p['latency_ms'],marker=markers[p['model']],ls='none',ms=10 if kernel else 5.5,
                mfc='white' if kernel else '#52565C',mec='#52565C',mew=1.5 if kernel else .7,zorder=6 if kernel else 7)
    xmin,xmax=4.02,5.86
    ymin,ymax=.43,2.85
    ax.set(xscale='log',yscale='log',xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='Validation loss',ylabel='Full-model latency (ms)')
    ax.set_xticks([4.2,4.5,4.8,5.1,5.4,5.7])
    ax.set_yticks([.5,1.,1.5,2.,2.5])
    for axis in (ax.xaxis,ax.yaxis):axis.set_major_formatter(ScalarFormatter());axis.set_minor_locator(NullLocator())
    ax.grid(axis='y',color='#E8EAED',lw=.6);ax.set_axisbelow(True);ax.tick_params(length=3,width=.65)
    ax.set_title(TITLE,loc='left',pad=12)
    family_handles=[Line2D([],[],marker='o',ls='none',ms=10,mfc='white',mec='#52565C',mew=1.5,label='Base (kernel)'),
        Line2D([],[],marker='o',ls='none',ms=5.5,color='#52565C',label='Base (PyTorch)'),
        *[Line2D([],[],color=colors[s],ls=linestyles[s],lw=1.6,label=label) for s,label in [('hz',r'$T_2/P_h$'),('7',r'$T_7/P_h$')]]]
    scale_handles=[Line2D([],[],marker=m,ls='none',color='#52565C',ms=5.8,label=s) for s,m in markers.items()]
    fig.legend(handles=family_handles,loc='lower center',bbox_to_anchor=(.54,.080),ncol=4,frameon=False,fontsize=9.5,handlelength=2.,columnspacing=1.4)
    fig.legend(handles=scale_handles,loc='lower center',bbox_to_anchor=(.54,.018),ncol=3,frameon=False,fontsize=9.5,columnspacing=2.2)
    assert all(xmin<=p['loss']<=xmax and ymin<=p['latency_ms']<=ymax for p in executions)
    output=HERE/'figures/01-base-t2-t7-scale-frontier.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches='tight',pad_inches=.04,metadata={'Title':TITLE,'CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert sha(old_pdf)==old['output_sha256']
    result=dict(executions=executions,measured_pareto_frontier=frontier,counts=dict(checkpoints=33,execution_points=36),
        source_sha256={old_path.relative_to(ROOT).as_posix():OLD_HASH,new_path.relative_to(ROOT).as_posix():sha(new_path)},
        output=output.relative_to(HERE).as_posix(),output_sha256=sha(output),historical_coordinates_unchanged=True,
        presentation=dict(title=TITLE,model_labels=list(markers),base_loss_guides=base_loss_guides,base_guide_labels=True,
            style_reference=STYLE_REFERENCE.relative_to(ROOT).as_posix(),style_reference_sha256=sha(STYLE_REFERENCE),
            kappa_annotations=False,workload_subtitle=False,legend_footer=False),
        interpretation='Absolute session-specific measurements. Colored segments connect threshold settings within each recipe and scale; vertical guides mark Base loss. No scaling law fitted.')
    (HERE/'data/scale-figure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(output)

if __name__=='__main__':main()
