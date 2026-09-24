"""Extend Analysis034 Figure02 without moving or removing any historical point."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'analyses/034-2026-09-21-70m-latency-quality-frontier'
OLD_HASH = 'ad3d09de96dda3bb7d67d6e4aaf75bd593a52a42226a04a4febb5dcb71ed7954'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sha(OLD/'data/combined-figure.json')==OLD_HASH
    old = json.loads((OLD/'data/combined-figure.json').read_text())
    assert sha(OLD/old['output'])==old['output_sha256']
    new_path = HERE/'data/31m-results.json';new = json.loads(new_path.read_text())
    for name,digest in new['sources_sha256'].items():
        assert sha(ROOT/name)==digest,name
    palette = json.loads((ROOT/'analyses/027-2026-09-20-run044-manuscript/data/figure-data.json').read_text())
    cohorts = {'14M':old['cohorts']['14M'],'31M':new['points'],'70M':old['cohorts']['70M']}
    metric = {'14M':'latency_ms','31M':'latency_ms','70M':'displayed_latency_ms'}
    markers = {'14M':'o','31M':'^','70M':'s'}
    highlights = [('hz','h'),('7','all')]
    refs = list(old['base_references'])
    base = next(p for p in new['points'] if p['scope']=='0')
    refs += [dict(model='31M',backend=backend,loss=base['loss'],latency_ms=base['implementation_latency_ms'][mode],
                  checkpoint_key=base['checkpoint_key'],timing_session='Run054',
                  implementation='opt073-31m-v1' if backend=='kernel' else 'native_graph')
             for backend,mode in [('kernel','kernel_graph'),('PyTorch','native_graph')]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':12,
        'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.65,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,ax = plt.subplots(figsize=(9.1,5.9));fig.subplots_adjust(left=.10,right=.985,top=.90,bottom=.27)
    handles = {};drawn=[]
    styles = sorted(palette['series'],key=lambda s:(s['scope'],s['pressure']) in highlights)
    for style in styles:
        key = (style['scope'],style['pressure'])
        if key==('0','none'):continue
        highlight = key in highlights
        for size,rows in cohorts.items():
            points = sorted([p for p in rows if (p['scope'],p['pressure'])==key],
                key=lambda p:(p.get('dose') if size=='14M' else p['kappa']) or 0)
            if not points:continue
            drawn += [p['checkpoint_key'] for p in points]
            color = style['color'] if highlight else '#9DA3AB'
            line = style['linestyle'];line=(line[0],tuple(line[1])) if isinstance(line,list) else line
            width,alpha = (1.6,1.) if highlight else (.85,.65)
            marker = dict(marker=markers[size],ms=6.4 if size=='31M' else (5.8 if highlight else 4.2),
                mfc=color,mec='white',mew=.7 if highlight else .5)
            x=[p['loss'] for p in points];y=[p[metric[size]] for p in points]
            ax.plot(x,y,color=color,ls=line,lw=width,alpha=alpha,zorder=4 if highlight else 2)
            ax.plot(x,y,color=color,ls='none',**marker,zorder=5 if highlight else 3)
            if highlight:handles[(key,size)] = Line2D([],[],color=color,ls=line,lw=width,**marker,label=f"{size} {style['label']}")
    bases = {}
    for ref in refs:
        size,backend=ref['model'],ref['backend'];kernel=backend=='kernel'
        marker=dict(marker=markers[size],ms=10 if kernel else 5.5,mfc='white' if kernel else '#52565C',
                    mec='#52565C',mew=1.5 if kernel else .7)
        ax.plot(ref['loss'],ref['latency_ms'],color='#52565C',ls='none',**marker,zorder=6 if kernel else 7)
        bases[(backend,size)]=Line2D([],[],color='#52565C',ls='none',**marker,label=f'{size} Base ({backend})')
    for size,rows in cohorts.items():
        p=next(p for p in rows if p['scope']=='0')
        ax.axvline(p['loss'],color='#92969B',lw=.8,ls=(0,(2,3)),zorder=1)
        ax.text(p['loss']+.025,.98,f'{size} Base loss',transform=ax.get_xaxis_transform(),
                color='#747A81',fontsize=8.5,ha='left',va='top')
    coords=[(p['loss'],p[metric[size]]) for size,rows in cohorts.items() for p in rows]
    coords += [(p['loss'],p['latency_ms']) for p in refs]
    xlim=(min(4.02,min(x for x,y in coords)*.98),max(6.10,max(x for x,y in coords)*1.01))
    ylim=(min(.40,min(y for x,y in coords)*.95),max(2.85,max(y for x,y in coords)*1.04))
    ax.set(xlim=xlim,ylim=ylim,xlabel='Validation loss',ylabel='Full-model latency (ms)',xscale='log',yscale='log')
    ax.set_xticks([4.2,4.5,4.8,5.1,5.4,5.7,6.0]);ax.set_yticks([.5,1.,1.5,2.,2.5])
    for axis in (ax.xaxis,ax.yaxis):
        axis.set_major_formatter(ScalarFormatter());axis.set_minor_locator(NullLocator());axis.set_minor_formatter(NullFormatter())
    ax.set_title('Pythia-14M, Pythia-31M and Pythia-70M',loc='left',pad=12)
    ax.tick_params(length=3,width=.65);ax.grid(axis='y',color='#E8EAED',lw=.6);ax.set_axisbelow(True)
    legend=[bases[(backend,size)] for backend in ('kernel','PyTorch') for size in cohorts]
    legend += [handles[(key,size)] for key in highlights for size in cohorts if (key,size) in handles]
    fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,fontsize=9.2,handlelength=2.,
        columnspacing=1.4,labelspacing=.8,bbox_to_anchor=(.54,.008))
    assert len(drawn)==len(set(drawn))==69 and len(refs)==6
    assert all(xlim[0]<=x<=xlim[1] and ylim[0]<=y<=ylim[1] for x,y in coords)
    output=HERE/'figures/01-14m-31m-70m-latency-quality.pdf';output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,bbox_inches='tight',pad_inches=.04,metadata={'Title':'Pythia-14M, Pythia-31M and Pythia-70M: loss and latency','CreationDate':None,'ModDate':None})
    plt.close(fig)
    assert sha(OLD/old['output'])==old['output_sha256']
    result=dict(cohorts=cohorts,base_references=refs,latency_fields=metric,markers=markers,xlim=xlim,ylim=ylim,
        counts=dict(checkpoints=72,interventions=69,base_executions=6,execution_points=75),
        sources_sha256={str((OLD/'data/combined-figure.json').relative_to(ROOT)):OLD_HASH,str(new_path.relative_to(ROOT)):sha(new_path)},
        output=str(output.relative_to(HERE)),output_sha256=sha(output),historical_pdf_unchanged=True,
        interpretation='Absolute session-specific measurements; no rescaling or fitted scaling law.')
    (HERE/'data/combined-figure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':main()
