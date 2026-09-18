"""Qualified 70M final port latency; same topology display as Figure03."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=HERE.parents[1]/'runs/035-2026-09-18-pythia70m-k050-port'
GROUPS=[
    ('A0','Baseline (A0)','#444A52','*',72,1),
    ('A1-H','1-site (A1-H)','#19856B','o',30,1),
    ('A4','4-sites (A4*)','#2878B5','D',34,10),
    ('A7','7-sites (A7)','#C96024','^',42,10),
]
LABELS={
    'A4+OL1@h':(-2,(-9,-1),'right'),
    'A4+OL1@all':(-2,(8,4),'left'),
    'A7+OL1@h':(-1,(0,-8),'center'),
    'A7+OL1@all':(-1,(6,-10),'left'),
}


def main():
    source=RUN/'results/70m-final-kernel.json'
    data=json.loads(source.read_text())
    assert data['status']=='complete_verified_reduction'
    points=data['points']
    assert len(points)==22 and all(p['qualified'] for p in points)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
        'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.6,3.8))
    connections=[];annotations=[]
    for prefix,label,color,marker,size,expected in GROUPS:
        selected=[p for p in points if p['family'].startswith(prefix)]
        assert len(selected)==expected
        for p in selected:
            c=p['canonical_counts']
            assert math.isclose(p['sparsity_percent'],100*c['block_zero_product_count']/c['model_product_count'],abs_tol=1e-12)
        for family in sorted({p['family'] for p in selected}):
            ordered=sorted([p for p in selected if p['family']==family],key=lambda p:p['kappa'] or 0)
            if len(ordered)==1:continue
            assert [p['kappa'] for p in ordered]==[0,.01,.05,.1,.5]
            ax.plot([p['sparsity_percent'] for p in ordered],[p['candidate_gm_ms'] for p in ordered],
                color=color,linestyle='--',linewidth=1,alpha=.8,zorder=2)
            connections.append({'family':family,'conditions':[p['condition'] for p in ordered],
                'order':'increasing kappa','kappa':[p['kappa'] for p in ordered]})
            # Small labels beside their own curves, clear of nearby families.
            index,offset,alignment=LABELS[family]
            p=ordered[index];text='OL1(h)' if family.endswith('@h') else 'OL1(all)'
            ax.annotate(text,(p['sparsity_percent'],p['candidate_gm_ms']),xytext=offset,
                textcoords='offset points',color=color,fontsize=6.8,alpha=.85,ha=alignment,va='center')
            annotations.append({'family':family,'text':text,'condition':p['condition'],'offset_points':offset,'alignment':alignment})
        ax.scatter([p['sparsity_percent'] for p in selected],[p['candidate_gm_ms'] for p in selected],
            label=label,color=color,marker=marker,s=size,linewidths=.5,edgecolors='white',zorder=3)
    ys=[p['candidate_gm_ms'] for p in points];xs=[p['sparsity_percent'] for p in points]
    pad=.08*(max(ys)-min(ys));ylim=(math.floor((min(ys)-pad)*100)/100,math.ceil((max(ys)+pad)*100)/100)
    xmax=5*math.ceil((max(xs)+4)/5)
    ax.set(xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
        ylabel='Full-model latency (ms)',xlim=(-.6,xmax),ylim=ylim)
    ax.set_xticks(range(0,xmax+1,5));ax.grid(axis='y',color='#E7E9ED',linewidth=.6)
    ax.set_axisbelow(True);ax.tick_params(direction='out',length=3,width=.6)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,
        fontsize=8.2,handletextpad=.4,columnspacing=1.35,bbox_to_anchor=(.53,.025))
    fig.subplots_adjust(left=.11,right=.985,bottom=.23,top=.965)
    assert sum(len(c.get_offsets()) for c in ax.collections)==22 and len(ax.lines)==4
    assert all(ax.get_xlim()[0]<p['sparsity_percent']<ax.get_xlim()[1] and ylim[0]<p['candidate_gm_ms']<ylim[1] for p in points)
    output=HERE/'figures/04-70m-k050-port-sparsity-latency-topology.pdf'
    fig.savefig(output,metadata={'Title':'Pythia-70M: final K050-derived port latency by topology',
        'Creator':'Analysis024 / 04_plot_70m_final_latency.py','CreationDate':None,'ModDate':None})
    plt.close(fig)
    record={'source':source.relative_to(HERE.parents[1]).as_posix(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'candidate':data['candidate'],'protocol':data['protocol'],'comparison_limits':data['comparison_limits'],
        'latency_definition':data['latency_definition'],'plotted_checkpoints':22,'y_limits_ms':ylim,
        'connections':connections,'annotations':annotations,'points':points}
    (HERE/'data/70m-final-latency-topology.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('22 qualified checkpoints; groups1/1/10/10; four pressure-family curves')


if __name__=='__main__':main()
