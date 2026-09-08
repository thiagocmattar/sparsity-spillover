"""Figure 05-v3 from the complete, retained Run 031 signed histograms."""
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]/'runs/031-2026-09-08-signed-activation-density'
FAMILIES = ('A0','A4-OL1','A7-OL1')
COLORS = ('#444444','#0072B2','#D55E00')
GROUPS = ('FFN activations','Attention activations')
KAPPAS = (0.,.05,.5)
REBIN = 10  # Sum native integer counts into nominal .01-wide display bins.
XLIMS = ((-2.5,3.),(-4.,4.))


def load_data():
    manifest = json.loads((RUN/'results/manifest.json').read_text())
    assert manifest['status']=='completed' and not manifest['smoke']
    assert manifest['completed_blocks']==2366 and manifest['blocks_per_checkpoint']==338
    rows, identities = {}, {}
    for path in sorted((RUN/'results/histograms').glob('*.json.gz')):
        data = json.loads(gzip.decompress(path.read_bytes()))
        assert data['coverage']['complete_block_coverage']
        assert data['coverage']['input_tokens']==692224
        assert data['coverage']['excluded_tail_tokens']==1444
        source = data['source']
        key = (source['family'],source['training_parameter'])
        assert key not in rows
        rows[key] = data
        identities[path.relative_to(HERE.parents[1]).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(rows)=={('A0',None)} | {(family,k) for family in FAMILIES[1:] for k in KAPPAS}
    return rows, identities


def density(group, grid):
    counts = np.asarray(group['histogram'],dtype=np.int64)
    assert len(counts)==grid['bins'] and len(counts)%REBIN==0
    assert int(counts.sum())+group['underflow']+group['overflow']+group['exact_zero_count']==group['total']
    native = np.linspace(grid['lower'],grid['upper'],grid['bins']+1,dtype=np.float32)
    edges = native[::REBIN].astype(np.float64)
    pooled = counts.reshape(-1,REBIN).sum(axis=1)
    values = pooled/(group['total']*np.diff(edges))
    assert np.isclose(np.sum(values*np.diff(edges)),counts.sum()/group['total'])
    return edges, values, pooled


def main():
    rows, identities = load_data()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.axisbelow':True,'axes.linewidth':.6})
    fig, axes = plt.subplots(3,2,figsize=(5.5,6.1),sharex='col',sharey='col')
    fig.subplots_adjust(left=.13,right=.98,bottom=.17,top=.89,wspace=.28,hspace=.30)
    report = {'source_sha256':identities,'native_bin_width':.001,'display_bin_width':.01,
              'zero_mass':'separate point mass; excluded from histogram',
              'normalization':'count / (all activation elements * bin width)',
              'density_axis':'symlog, linear threshold .01','panels':[]}
    for i,kappa in enumerate(KAPPAS):
        for j,name in enumerate(GROUPS):
            ax = axes[i,j]
            panel = {'kappa':kappa,'group':name,'xlim':XLIMS[j],'recipes':{}}
            for family,color in zip(FAMILIES,COLORS):
                data = rows[(family,None if family=='A0' else kappa)]
                group = data['groups'][name]
                edges, values, counts = density(group,data['grid'])
                assert values.max()<60, 'Increase the shared density range to include every peak'
                ax.stairs(values,edges,color=color,lw=.9,label=family,zorder=3)
                ax.stairs(values,edges,color=color,fill=True,alpha=.10,lw=0,zorder=2)
                zero = group['exact_zero_count']/group['total']
                zero_label = '0%' if zero==0 else '<0.01%' if 100*zero<.01 else f'{100*zero:.2f}%'
                outside = int(counts[(edges[:-1]<XLIMS[j][0])|(edges[1:]>XLIMS[j][1])].sum())
                outside += group['underflow']+group['overflow']
                panel['recipes'][family] = {'zero_fraction':zero,
                    'zero_label':zero_label,'display_outside_fraction':outside/group['total'],
                    'histogram_outside_fraction':(group['underflow']+group['overflow'])/group['total'],
                    'total':group['total']}
            for threshold in ((kappa,) if j==0 or kappa==0 else (-kappa,kappa)):
                ax.axvline(threshold,color='.3',lw=.8,ls='--',zorder=4)
            ax.grid(axis='both',color='.88',lw=.4)
            ax.set_yscale('symlog',linthresh=.01,linscale=.5)
            ax.set_xlim(XLIMS[j])
            ax.set_ylim(0,60)
            ax.set_yticks([0,.01,.1,1,10],['0','.01','.1','1','10'])
            ax.set_xticks([-2,-1,0,1,2,3] if j==0 else [-4,-2,0,2,4])
            if i==0:
                ax.set_title(name,pad=10)
            if j==0:
                ax.set_ylabel(r'$\kappa='+f'{kappa:g}'+r'$'+'\nDensity')
            if i==2:
                ax.set_xlabel(r'Activation $x$')
            report['panels'].append(panel)
    fig.suptitle('How interventions reshape activation distributions (Pythia-14M)',fontsize=11,y=.985)
    legend = [Line2D([],[],color=c,lw=1.2,label=f) for f,c in zip(FAMILIES,COLORS)]
    legend += [Line2D([],[],color='.3',lw=.8,ls='--',label=r'Gate threshold $\kappa$')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.54,.035),ncol=4,
               frameon=False,fontsize=7.5,columnspacing=1.0,handlelength=1.6,handletextpad=.4)
    (HERE/'figures').mkdir(exist_ok=True)
    fig.savefig(HERE/'figures/05-v3-activation-density-grid.pdf',
                metadata={'Creator':'Analysis 018; Run 031 measurements','CreationDate':None,'ModDate':None})
    plt.close(fig)
    (HERE/'activation-density-v3-data.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    artifact = HERE/'figures/05-v3-activation-density-grid.pdf'
    inventory = json.loads((HERE/'artifact_inventory.json').read_text())
    inventory[artifact.relative_to(HERE).as_posix()] = {
        'bytes':artifact.stat().st_size,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
    (HERE/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
