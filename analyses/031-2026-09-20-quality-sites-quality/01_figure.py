"""Introduction: matched quality panels around Table 2's conditional savings."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE/'figures/01-quality-sites-quality.pdf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {}

    def read(relative):
        path = ROOT/relative
        sources[relative] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    small = read('analyses/027-2026-09-20-run044-manuscript/data/figure-data.json')
    large = read('analyses/029-2026-09-20-70m-t2-figure/data/70m-quality-sparsity-native-latency.json')
    table = read('analyses/024-2026-09-17-h-only-kernel-latency/data/operation-latency.json')
    for parent in (small, large, table):
        for relative, digest in parent.get('sources_sha256', parent.get('source_sha256', {})).items():
            assert sha(ROOT/relative) == digest, relative
    raw = read('runs/037-2026-09-19-pythia14m-operation-latency/results/operation-latency.json')
    assert raw['status'] == 'qualified'
    assert table['verification']['qualified_processes'] == 30
    for row in table['rows']:
        mode = 'without-'+row['operation']
        expected = 1000*(raw['latencies_ms'][mode]-raw['latencies_ms']['full'])
        low, high = raw['process_ranges_ms'][mode]
        full_low, full_high = raw['process_ranges_ms']['full']
        span = [1000*(low-full_high),1000*(high-full_low)]
        assert math.isclose(expected,row['saved_microseconds'],abs_tol=1e-10)
        assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(span,row['cross_process_difference_span_us']))
    source_table = ROOT/'manuscript/draft/tables/operation-latency.tex'
    sources[source_table.relative_to(ROOT).as_posix()] = sha(source_table)
    for row in table['rows']:
        assert f"${row['saved_microseconds']:+.1f}$" in source_table.read_text()

    groups = lambda rows: {(r['scope'],r['pressure']) for r in rows}
    common = groups(small['trained_points']) & groups(large['trained_points'])
    order = [('0','none'),('1','none'),('hz','h'),('4','h'),('7','h'),('4','all'),('7','all')]
    assert common == set(order)
    styles = {(s['scope'],s['pressure']):s for s in small['series']}
    panels = {}
    for name, source in [('14M',small),('70M',large)]:
        rows = [r for r in source['trained_points'] if (r['scope'],r['pressure']) in common]
        assert len(rows) == 27 and len({r['checkpoint_key'] for r in rows}) == 27
        for key in order[2:]:
            assert sorted(r['kappa'] for r in rows if (r['scope'],r['pressure'])==key) == [0,.01,.05,.1,.5]
        panels[name] = dict(points=rows,clipping=source['clipping_points'],ceilings=source['ceilings'])
        assert len(panels[name]['clipping']) == 20

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,
        'axes.titlesize':12,'xtick.labelsize':10,'ytick.labelsize':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    fig,axes = plt.subplots(1,3,figsize=(12.2,4.5),gridspec_kw={'width_ratios':[1,.92,1]})
    fig.subplots_adjust(left=.06,right=.989,top=.82,bottom=.32,wspace=.36)
    handles = []
    for size,ax,letter in [('14M',axes[0],'a'),('70M',axes[2],'c')]:
        panel = panels[size]
        for key in order:
            style = styles[key]
            rows = sorted((r for r in panel['points'] if (r['scope'],r['pressure'])==key),key=lambda r:r['kappa'] or 0)
            control = key[0] in ('0','1')
            color = style['color']
            ls = tuple(style['linestyle']) if isinstance(style['linestyle'],list) else style['linestyle']
            line = dict(color=color,ls=ls,lw=1.7,marker='o',ms=8 if control else 5.5,
                mfc='white' if key[0]=='0' else color,mec=color if key[0]=='0' else 'white',
                mew=1.4 if key[0]=='0' else .7)
            ax.plot([r['sparsity'] for r in rows],[r['loss'] for r in rows],**line,zorder=5 if control else 4)
            if size=='14M': handles.append(Line2D([],[],**line,label=style['label']))
        for scope in ('0','1'):
            rows = sorted((r for r in panel['clipping'] if r['scope']==scope),key=lambda r:r['target'])
            ax.plot([r['sparsity'] for r in rows],[r['loss'] for r in rows],
                color=styles[(scope,'none')]['color'],ls=':',lw=1.25,zorder=2)
        limits = (-.75,31,5.06,6.19) if size=='14M' else (-1.25,51,4,5.58)
        ax.set(xlim=limits[:2],ylim=limits[2:],xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)',ylabel='Validation loss')
        ax.set_xticks([0,10,20,30] if size=='14M' else [0,10,20,30,40,50])
        ax.set_title(f'({letter}) Pythia-{size}',loc='left',pad=31)
        ax.text(0,1.045,'Quality-sparsity trade-off',transform=ax.transAxes,fontsize=10.5,color='#555D66')
        ax.grid(axis='y',color='#E8EAED',lw=.6); ax.set_axisbelow(True)
        assert all(limits[0]<=r['sparsity']<=limits[1] and limits[2]<=r['loss']<=limits[3] for r in panel['points'])

    ax = axes[1]
    rows = table['rows']
    assert [r['operation'] for r in rows] == ['a','m','h','z','qk','pv']
    values = [r['saved_microseconds'] for r in rows]
    errors = [[v-r['cross_process_difference_span_us'][0] for v,r in zip(values,rows)],
              [r['cross_process_difference_span_us'][1]-v for v,r in zip(values,rows)]]
    ax.bar(range(6),values,width=.64,color=['#9DA5AE','#9DA5AE','#485667','#485667','#9DA5AE','#9DA5AE'],
        yerr=errors,capsize=3,error_kw={'elinewidth':1,'capthick':1,'ecolor':'#343B44'},zorder=3)
    for i,row in enumerate(rows):
        v=row['saved_microseconds'];low,high=row['cross_process_difference_span_us']
        ax.text(i,high+5 if v>0 else -15,f'{v:+.1f}',ha='center',va='bottom' if v>0 else 'top',fontsize=10)
    ax.axhline(0,color='#515861',lw=.8,zorder=2)
    ax.set(ylim=(-30,177),ylabel=r'Saved time ($\mu$s)',xlabel='Sparsification site')
    ax.set_xticks(range(6),[r'$a$',r'$m$',r'$h$',r'$z$',r'$q,k$',r'$v$'])
    ax.set_yticks([0,50,100,150]);ax.grid(axis='y',color='#E8EAED',lw=.6);ax.set_axisbelow(True)
    ax.set_title('(b) Conditional time savings',loc='left',pad=31)
    ax.text(0,1.045,r'14M $T_7/P_{\mathrm{all}}$, $\kappa=0.5$',transform=ax.transAxes,fontsize=10.5,color='#555D66')
    for ax in axes:ax.tick_params(length=3,width=.65)
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=11,
        handlelength=2.2,columnspacing=1.7,labelspacing=.65,bbox_to_anchor=(.52,.075))
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT,bbox_inches='tight',pad_inches=.04,metadata={'Title':'Matched quality-sparsity trade-offs and conditional site savings',
        'CreationDate':None,'ModDate':None})
    plt.close(fig)
    record=dict(output=OUTPUT.relative_to(HERE).as_posix(),output_sha256=sha(OUTPUT),
        script=Path(__file__).name,script_sha256=sha(Path(__file__)),sources_sha256=sources,
        panels=panels,bar_chart=table,shared_recipes=[dict(scope=k[0],pressure=k[1],style=styles[k]) for k in order],
        scope='Approved introduction figure. Canonical quality, not kernel qualification loss.',
        coverage=dict(validation_blocks=338,validation_documents=500,excluded_tail_tokens=1444),
        note='Panel b uses one 14M checkpoint; bars are not additive site allocations or confidence intervals.')
    (HERE/'data').mkdir(exist_ok=True)
    (HERE/'data/figure-data.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(pdf=str(OUTPUT),points_per_quality_panel=27,shared_recipes=7,bars=6)))


if __name__=='__main__':main()
