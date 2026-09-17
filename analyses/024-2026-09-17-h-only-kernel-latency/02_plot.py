"""Clean, annotation-free full-model speedup and absolute-latency figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FAMILIES = [
    ('A4','A4','#2878B5','-','o','white'),
    ('A4+OL1@4','A4 + OL1(all)','#2878B5','--','o','#2878B5'),
    ('A4+OL1@h','A4 + OL1(h)','#2878B5',':','s','#2878B5'),
    ('A7','A7','#C96024','-','^','white'),
    ('A7+OL1@7','A7 + OL1(all)','#C96024','--','^','#C96024'),
    ('A7+OL1@h','A7 + OL1(h)','#C96024',':','D','#C96024'),
]


def draw(ax, points, key):
    controls = [p for p in points if p['family'].startswith('A1-H')]
    ax.scatter([p['sparsity_percent'] for p in controls], [p[key] for p in controls],
               s=17, marker='x', color='.62', linewidths=.7, zorder=2)
    baseline, = [p for p in points if p['family']=='A0']
    ax.scatter([baseline['sparsity_percent']], [baseline[key]], s=28, marker='*',
               color='.2', zorder=3)
    for family, label, color, linestyle, marker, face in FAMILIES:
        rows = sorted([p for p in points if p['family']==family], key=lambda p:p['kappa_or_lambda'])
        assert [p['kappa_or_lambda'] for p in rows] == [0,.01,.05,.1,.5]
        assert all(p['qualified'] for p in rows), 'Do not plot a failed numerical qualification as a gain'
        x, y = [p['sparsity_percent'] for p in rows], [p[key] for p in rows]
        ax.plot(x, y, color=color, ls=linestyle, lw=1.1, marker=marker,
                ms=4, mfc=face, mec=color, mew=.8, label=label, zorder=4)
        if key=='speedup':
            ax.errorbar(x, y,
                yerr=[[p[key]-p['process_speedup_min'] for p in rows],
                      [p['process_speedup_max']-p[key] for p in rows]],
                fmt='none', ecolor=color, elinewidth=.7, capsize=1.5, zorder=3)
    ax.set_xlabel(r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)')
    ax.set_xlim(-.6, 30)
    ax.set_xticks(range(0,31,5))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='.9', lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)


def legend(fig):
    handles = [Line2D([],[],color=c,ls=ls,marker=m,ms=4,lw=1.1,mfc=f,mec=c,
                      mew=.8,label=label) for _,label,c,ls,m,f in FAMILIES]
    handles = [handles[i] for i in [0,3,1,4,2,5]]
    handles += [Line2D([],[],color='.2',ls='none',marker='*',ms=5,label='A0'),
                Line2D([],[],color='.62',ls='none',marker='x',ms=4,label='A1-H controls')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               fontsize=7.5, columnspacing=1.4, handlelength=2.3,
               bbox_to_anchor=(.5,.015))


def main():
    data=json.loads((HERE/'data/results.json').read_text())
    points=data['points']
    assert len(points)==40 and all(p['qualified'] for p in points)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,
                         'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.linewidth':.6,'savefig.transparent':False})
    target=HERE/'figures';target.mkdir(exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.2,3.55))
    draw(ax,points,'speedup')
    ax.axhline(1,color='.45',lw=.7,ls='--',zorder=1)
    ax.set_ylabel('Full-model speedup (native / K050)')
    ax.set_ylim(min(.95,min(p['process_speedup_min'] for p in points)-.025),
                max(p['process_speedup_max'] for p in points)+.045)
    legend(fig);fig.subplots_adjust(left=.10,right=.985,bottom=.27,top=.97)
    fig.savefig(target/'01-14m-k050-sparsity-speedup.pdf',metadata={
        'Title':'Model-wide sparsity and final K050 speedup, including h-only OL1',
        'Creator':'Analysis 024 / 02_plot.py','CreationDate':None})
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.55),sharex=True,sharey=True)
    for ax,key,title in zip(axes,['native_gm_ms','k050_gm_ms'],['Native PyTorch SDPA','Final K050']):
        draw(ax,points,key);ax.set_title(title,pad=7)
    axes[0].set_ylabel('Full-model latency (ms)')
    axes[0].set_ylim(0,1.07*max(p['native_gm_ms'] for p in points))
    legend(fig);fig.subplots_adjust(left=.08,right=.985,bottom=.27,top=.9,wspace=.14)
    fig.savefig(target/'02-14m-k050-sparsity-latency.pdf',metadata={
        'Title':'Absolute native and final K050 latency, including h-only OL1',
        'Creator':'Analysis 024 / 02_plot.py','CreationDate':None})
    plt.close(fig)
    print('Two PDF figures; 40 checkpoints each; only Run033 newly timed')


if __name__=='__main__':main()
