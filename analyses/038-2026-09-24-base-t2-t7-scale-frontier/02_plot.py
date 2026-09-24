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
    colors={'hz':'#0072B2','7':'#D55E00'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,ax=plt.subplots(figsize=(9.0,6.0));fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.87)
    for scale,marker in markers.items():
        for scope,_ in families:
            points=sorted([p for p in executions if p['model']==scale and p['scope']==scope],key=lambda p:p['kappa'])
            x=[p['loss'] for p in points];y=[p['latency_ms'] for p in points]
            ax.plot(x,y,color=colors[scope],lw=1.5,marker=marker,ms=6.8,mec='white',mew=.65,zorder=4)
            for p,label in [(points[0],'0'),(points[-1],'0.5')]:
                ax.annotate(label,(p['loss'],p['latency_ms']),xytext=(5,6),textcoords='offset points',
                    color=colors[scope],fontsize=7.5,zorder=7)
    for backend in ('kernel','PyTorch'):
        points=sorted([p for p in refs if p['backend']==backend],key=lambda p:p['latency_ms'])
        # Connect only actual Base scales; this is a visual guide, not a fit.
        ax.plot([p['loss'] for p in points],[p['latency_ms'] for p in points],color='#74797E',lw=.75,
            ls=(0,(3,4)) if backend=='kernel' else (0,(1,3)),zorder=1)
        for p in points:
            kernel=backend=='kernel'
            ax.plot(p['loss'],p['latency_ms'],marker=markers[p['model']],ls='none',ms=10 if kernel else 5.5,
                mfc='white' if kernel else '#42474D',mec='#42474D',mew=1.5 if kernel else .6,zorder=6 if kernel else 7)
    xmin=min(p['loss'] for p in executions)*.985;xmax=max(p['loss'] for p in executions)*1.025
    ymin=min(p['latency_ms'] for p in executions)*.92;ymax=max(p['latency_ms'] for p in executions)*1.09
    ax.set(xscale='log',yscale='log',xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='Validation loss',ylabel='Full-model latency (ms)')
    ax.set_xticks([x for x in (4.1,4.3,4.5,4.7,4.9,5.1,5.3,5.5,5.7,5.9,6.1) if xmin<x<xmax])
    ax.set_yticks([x for x in (.5,.7,1.,1.5,2.,2.5,3.) if ymin<x<ymax])
    for axis in (ax.xaxis,ax.yaxis):axis.set_major_formatter(ScalarFormatter());axis.set_minor_locator(NullLocator())
    ax.grid(axis='y',color='#E7E9EC',lw=.6);ax.set_axisbelow(True);ax.tick_params(length=3,width=.65,labelsize=9.5)
    ax.set_title('Loss and latency across Pythia model scales',loc='left',fontsize=14,pad=26)
    ax.text(0,1.025,'RTX 5090  ·  BF16  ·  batch 1  ·  2,048 tokens  ·  full logits',transform=ax.transAxes,color='#676D73',fontsize=9)
    family_handles=[Line2D([],[],marker='o',ls='none',ms=9,mfc='white',mec='#42474D',mew=1.4,label='Base (kernel)'),
        Line2D([],[],marker='o',ls='none',ms=5.5,color='#42474D',label='Base (PyTorch)'),
        *[Line2D([],[],color=colors[s],lw=1.6,label=label) for s,label in [('hz',r'$T_2/P_h$'),('7',r'$T_7/P_h$')]]]
    scale_handles=[Line2D([],[],marker=m,ls='none',color='#5E6369',ms=7,label='30M (Pythia-31M)' if s=='31M' else s) for s,m in markers.items()]
    fig.legend(handles=family_handles,loc='lower center',bbox_to_anchor=(.54,.105),ncol=4,frameon=False,fontsize=10,handlelength=1.8,columnspacing=1.7)
    fig.legend(handles=scale_handles,loc='lower center',bbox_to_anchor=(.54,.053),ncol=3,frameon=False,fontsize=9.5,columnspacing=2.2)
    fig.text(.54,.017,r'Each colored curve sweeps $\kappa\in\{0,0.01,0.05,0.1,0.5\}$; endpoint labels show $\kappa$.',ha='center',fontsize=8.5,color='#676D73')
    assert all(xmin<=p['loss']<=xmax and ymin<=p['latency_ms']<=ymax for p in executions)
    output=HERE/'figures/01-base-t2-t7-scale-frontier.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches='tight',pad_inches=.06,metadata={'Title':'Base, T2/Ph and T7/Ph across Pythia model scales','CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert sha(old_pdf)==old['output_sha256']
    result=dict(executions=executions,measured_pareto_frontier=frontier,counts=dict(checkpoints=33,execution_points=36),
        source_sha256={old_path.relative_to(ROOT).as_posix():OLD_HASH,new_path.relative_to(ROOT).as_posix():sha(new_path)},
        output=output.relative_to(HERE).as_posix(),output_sha256=sha(output),historical_coordinates_unchanged=True,
        interpretation='Absolute session-specific measurements. Connecting segments guide the eye; no scaling law fitted.')
    (HERE/'data/scale-figure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(output)

if __name__=='__main__':main()
