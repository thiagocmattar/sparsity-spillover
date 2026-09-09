"""Revised manuscript visuals from frozen records; no evaluation or old writes."""
from pathlib import Path
import gzip
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'analyses/018-2026-09-08-results-materials'
sys.path.insert(0, str(PRIOR))
import plots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

OUT = HERE / 'figures'
DATA = json.loads((PRIOR/'figure_data.json').read_text(encoding='utf-8'))
RUNTIME = json.loads((HERE/'runtime-audit.json').read_text())
spec = importlib.util.spec_from_file_location('density018', PRIOR/'02_activation_density_v3.py')
density018 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(density018)


def save(fig, name):
    fig.savefig(OUT/name, metadata={'Creator':'Analysis 019; audited existing records',
                                  'CreationDate':None, 'ModDate':None})
    plt.close(fig)


def revised_original(fig, name):
    if name == '01-v2-14m-overview.pdf':
        fig._suptitle.set_text('Quality–sparsity trade-offs for Pythia-14M')
        ax=fig.axes[0]
        endpoint=next(r for r in DATA['trained'] if r['id']=='14M:A7-OL1:0.5')
        ax.annotate('A7-OL1, κ=0.5\nΔ loss vs A0: +0.6208',
                    xy=(100*endpoint['R_model'],endpoint['loss']),xytext=(17.3,6.07),
                    fontsize=8,ha='left',va='top',arrowprops={'arrowstyle':'-','color':'.4','lw':.6})
        ax.annotate('A0',xy=(0,5.2085730123),xytext=(.8,5.27),fontsize=8,
                    arrowprops={'arrowstyle':'-','color':'.4','lw':.6})
    elif name == '03-scale-transfer-and-ceilings.pdf':
        fig._suptitle.set_text('Quality–sparsity trade-offs across Pythia sizes')
        fig.axes[4].set_xlabel(r'Block-only sparsity $\mathcal{S}_{\mathrm{block}}$ (%)')
        for legend in fig.legends:
            for label in legend.get_texts():
                label.set_text(label.get_text().replace('ceiling','reach'))
        name='03-scale-transfer-and-reach.pdf'
    elif name == '07-kernel-realization.pdf':
        fig._suptitle.set_text('Matched retrospective search and native-relative association')
        fig.axes[1].set_title('(b) Selected K050')
        name='08-search-history-and-association.pdf'
    save(fig,name)


def density_figure():
    paths=sorted((ROOT/'manuscript/draft/supplementary-data/histograms').glob('*.json.gz'))
    assert len(paths)==7
    rows={}
    for p in paths:
        d=json.loads(gzip.decompress(p.read_bytes()))
        rows[(d['source']['family'],d['source']['training_parameter'])]=d
    verified=json.loads((PRIOR/'activation-density-v3-data.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(5.5,6.9),sharex='col',sharey='col')
    fig.subplots_adjust(left=.12,right=.98,bottom=.12,top=.83,wspace=.27,hspace=.87)
    styles=('-', '--', '-.')
    for i,kappa in enumerate(density018.KAPPAS):
        for j,group in enumerate(density018.GROUPS):
            ax=axes[i,j]
            panel=next(p for p in verified['panels'] if p['kappa']==kappa and p['group']==group)
            for family,color,style in zip(density018.FAMILIES,density018.COLORS,styles):
                d=rows[(family,None if family=='A0' else kappa)]
                g=d['groups'][group]
                edges,values,counts=density018.density(g,d['grid'])
                assert abs(g['exact_zero_count']/g['total']-panel['recipes'][family]['zero_fraction'])<1e-14
                ax.stairs(values,edges,color=color,lw=.9,ls=style,zorder=3)
            zeros=[panel['recipes'][f]['zero_label'] for f in density018.FAMILIES]
            ax.text(0,1.06,f'Exact zeros: A0 {zeros[0]}\nA4-OL1 {zeros[1]}; A7-OL1 {zeros[2]}',
                    transform=ax.transAxes,fontsize=8,va='bottom',linespacing=1.3)
            ax.set_title(f'κ={kappa:g}: '+('FFN (h, m)' if j==0 else 'Attention (q, k, v)'),
                         fontsize=9,y=1.42,pad=0)
            for threshold in ((kappa,) if j==0 or kappa==0 else (-kappa,kappa)):
                ax.axvline(threshold,color='.45',ls=':',lw=.8)
            ax.set_yscale('symlog',linthresh=.01,linscale=.5)
            ax.set(xlim=density018.XLIMS[j],ylim=(0,60))
            ax.set_yticks([0,.01,.1,1,10],['0','.01','.1','1','10'])
            ax.set_xticks([-2,-1,0,1,2,3] if j==0 else [-4,-2,0,2,4])
            ax.grid(color='.9',lw=.4)
            if j==0:ax.set_ylabel('Density')
            if i==2:ax.set_xlabel('Activation x')
    fig.suptitle('Exact-zero mass and nonzero activation distributions',fontsize=11,y=.99)
    legend=[Line2D([],[],color=c,ls=s,label=f) for f,c,s in zip(density018.FAMILIES,density018.COLORS,styles)]
    legend.append(Line2D([],[],color='.45',ls=':',label='Threshold'))
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.01),ncol=4,frameon=False,fontsize=8)
    save(fig,'05-v4-activation-density-grid.pdf')


def runtime_figure():
    fig,axes=plt.subplots(1,2,figsize=(5.5,3.5),gridspec_kw={'width_ratios':(.85,1.2)})
    fig.subplots_adjust(left=.19,right=.98,bottom=.28,top=.79,wspace=.52)
    a,b=axes
    for y,key,color in [(2,'k050-no-skip','#888888'),(1,'k050','#333333'),(0,'k050-attention-dense','#009E73')]:
        x=RUNTIME['candidate_summary'][key]['geomean_native_relative']
        a.plot([1,x],[y,y],color=color,lw=1.5)
        a.scatter([x],[y],c=color,s=28)
        a.text(x+.007,y,f'{x:.4f}×',fontsize=8,va='center')
    a.set(yticks=[0,1,2],yticklabels=['Attention dense','K050','All skips off'],xlim=(.98,1.36),ylim=(-.7,2.7),
          xlabel='GM speedup vs native',title='(a) Cohort means')
    a.set_xticks([1,1.1,1.2,1.3]);a.axvline(1,color='.65',ls=':',lw=.8)
    for family in plots.FAMILIES:
        rows=[r for r in RUNTIME['rows'] if r['family']==family]
        color,marker=plots.STYLE[family]
        b.scatter([100*r['canonical_FP16_S_model'] for r in rows],
                  [r['sparse_path_factor'] for r in rows],color=color,marker=marker,s=25)
    fit=RUNTIME['association_sensitivity']['all']['sparse_path'];xx=np.array([0,.30])
    b.plot(100*xx,fit['intercept']+fit['slope']*xx,color='.5',ls='--',lw=.8)
    c30=next(r for r in RUNTIME['rows'] if r['condition']=='c30')
    b.scatter([100*c30['canonical_FP16_S_model']],[c30['sparse_path_factor']],facecolors='none',edgecolors='black',s=85,lw=.7)
    b.text(.04,.95,f"GM {RUNTIME['sparse_path_geomean']:.4f}×; helps 14/30\nDescriptive OLS $R^2$={fit['r_squared']:.3f}",
           transform=b.transAxes,fontsize=8,va='top')
    b.set(xlim=(-1,31),ylim=(.86,1.49),xlabel=plots.S_LABEL,ylabel='K050 / all-skips-off speed factor',title='(b) Sparse-path increment')
    b.axhline(1,color='.65',ls=':',lw=.8);b.grid(axis='y',color='.92',lw=.6)
    fig.legend(handles=plots.handles(plots.FAMILIES),loc='lower center',bbox_to_anchor=(.56,.015),ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Fusion gains and conditional benefit from sparse execution',fontsize=11,y=.98)
    save(fig,'07-kernel-attribution.pdf')


def main():
    OUT.mkdir(exist_ok=True)
    plots.configure()
    plots.save=revised_original
    plots.overview(DATA,all_variants=True)
    plots.scaling(DATA)
    plots.kernels(DATA)
    density_figure()
    runtime_figure()
    sources=[PRIOR/'figure_data.json',PRIOR/'activation-density-v3-data.json',PRIOR/'plots.py',
             PRIOR/'02_activation_density_v3.py',HERE/'runtime-audit.json',
             *sorted((ROOT/'manuscript/draft/supplementary-data/histograms').glob('*.json.gz'))]
    (OUT/'SOURCES.json').write_text(json.dumps({str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print('Five PDF figures regenerated from frozen numerical records.')


if __name__=='__main__':
    main()
