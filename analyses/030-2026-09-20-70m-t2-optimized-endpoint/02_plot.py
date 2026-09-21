"""Focused manuscript plots; historical counts never accompany new-kernel timings."""
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analyses/024-2026-09-17-h-only-kernel-latency'
STYLE = ROOT / 'analyses/027-2026-09-20-run044-manuscript/data/figure-data.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name, evidence):
    path = HERE / 'figures' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)
    record = dict(output=path.relative_to(HERE).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  script=Path(__file__).name, evidence=evidence)
    (HERE/'data'/f'{path.stem}.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')


def main_70m(data, styles):
    rows = [r for r in data['trained_points'] if r['model']=='70M']
    assert len(rows)==27
    clips = [r for r in data['clipping_points'] if r['model']=='70M' and r['scope'] in ('0','1')]
    order = [('0','none'),('1','none'),('hz','h'),('4','h'),('7','h'),('4','all'),('7','all')]
    native = data['matched_summary']['native_base_ms']
    latencies = [r['displayed_latency_ms'] for r in rows if r['displayed_latency_ms'] is not None]
    ylim = (math.floor((min(latencies)-.08)*10)/10, math.ceil((max(latencies)+.08)*10)/10)
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.85),sharex=True)
    fig.subplots_adjust(left=.08,right=.985,top=.84,bottom=.26,wspace=.26)
    fig.suptitle('Quality, sparsity and latency on Pythia-70M',fontsize=12.6,y=.975)
    handles=[]
    for key in order:
        s=styles[key]
        group=sorted((r for r in rows if (r['scope'],r['pressure'])==key),key=lambda r:r['kappa'] or 0)
        control=key[0] in ('0','1')
        line=dict(color=s['color'],ls=tuple(s['linestyle']) if isinstance(s['linestyle'],list) else s['linestyle'],
                  lw=1.6,marker='o',ms=9 if control else 5.8,mfc='white' if key[0]=='0' else s['color'],
                  mec=s['color'] if key[0]=='0' else 'white',mew=1.5 if key[0]=='0' else .7)
        for ax,metric in zip(axes,('loss','displayed_latency_ms')):
            ax.plot([r['sparsity'] for r in group],[r[metric] for r in group],**line,zorder=5 if control else 4)
        handles.append(Line2D([],[],**line,label=s['label']))
    extra, = [r for r in rows if r.get('run047_id')=='c26']
    for scope in ('0','1'):
        group=sorted((r for r in clips if r['scope']==scope),key=lambda r:r['target'])
        axes[0].plot([r['sparsity'] for r in group],[r['loss'] for r in group],
                     color=styles[(scope,'none')]['color'],ls=':',lw=1.3,zorder=2)
    for scope in ('hz','4','7'):
        ceiling=data['ceilings']['70M'][scope]['R_model_max_percent']
        axes[0].axvline(ceiling,color='#92969B',lw=.8,ls=(0,(2,3)),alpha=.8,zorder=1)
        axes[0].text(ceiling+.5 if scope=='hz' else ceiling-.25,.98,
                     rf'$T_{{{"2" if scope=="hz" else scope}}}$ ceiling',
                     transform=axes[0].get_xaxis_transform(),ha='left' if scope=='hz' else 'right',
                     va='top',fontsize=8.55,color='#6C7177')
    axes[1].axhline(native,color='#52565C',lw=.9,ls=':',zorder=1)
    axes[1].text(2,native+.018,f'PyTorch base: {native:.3f} ms',color='#52565C',fontsize=8.55)
    for ax,title,ylabel,limits in zip(axes,('(a) Quality-sparsity trade-off','(b) Full-model latency'),
            ('Validation loss','Full-model latency (ms)'),((4,5.58),ylim)):
        ax.set(xlim=(-1.25,51),ylim=limits,xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)',ylabel=ylabel)
        ax.set_title(title,loc='left',pad=11)
        ax.set_xticks([0,10,20,30,40,50]);ax.grid(axis='y',color='#E8EAED',lw=.6);ax.set_axisbelow(True)
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9.18,
               handlelength=2.1,columnspacing=1.5,labelspacing=.7,bbox_to_anchor=(.53,.005))
    save(fig,'23-70m-quality-sparsity-native-latency.pdf',dict(
        source='data/full-trained-results.json',checkpoint_keys=[r['checkpoint_key'] for r in rows],
        native_base_ms=native,base_marker_backend='native_graph',other_markers_backend='candidate_graph',
        clipping_panel='a only; unchanged historical quality',latency_limits=ylim,
        timing_note=data['timing_note'], later_endpoint_key=extra['checkpoint_key']))


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.9,'axes.titlesize':9.9,
        'axes.labelsize':9.9,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'mathtext.fontset':'dejavusans'})
    styles={(s['scope'],s['pressure']):s for s in read(STYLE)['series']}
    main_70m(read(HERE/'data/full-trained-results.json'),styles)
