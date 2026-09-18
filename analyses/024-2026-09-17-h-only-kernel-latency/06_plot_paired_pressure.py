"""Matched OL1(all)-minus-OL1(h) effects within four- and seven-site gates."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = [('4-sites', 'a4', '#2878B5'), ('7-sites', 'a7', '#C96024')]
METRICS = [
    ('loss_nats', 'Validation loss', r'$\Delta$ loss (nats/token)'),
    ('sparsity_pp', 'Model-wide sparsity', r'$\Delta\mathcal{S}_{\mathrm{model}}$ (percentage points)'),
    ('latency_us', 'K050 latency', r'$\Delta$ latency ($\mu$s)'),
]


def collect():
    # Figure05 already retains all 20 required endpoints, exact counts and identities.
    source = HERE / 'data/paired-topology-effects.json'
    old = json.loads(source.read_text(encoding='utf-8'))
    for name, sha in old['sources_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    lookup = {(p['recipe'], p['kappa']): p for p in old['pairs']}
    assert len(lookup) == len(old['pairs']) == 15
    pairs = []
    for label, topology, _ in GROUPS:
        for kappa in [0, .01, .05, .1, .5]:
            h = lookup['OL1(h)', kappa][topology]
            all_sites = lookup['OL1(all)', kappa][topology]
            assert all_sites['model_product_count'] == h['model_product_count']
            delta = {key: all_sites[key] - h[key] for key, _, _ in METRICS}
            assert math.isclose(delta['sparsity_pp'],
                100 * (all_sites['block_zero_product_count'] - h['block_zero_product_count'])
                / h['model_product_count'], abs_tol=1e-12)
            assert all(math.isfinite(v) for v in delta.values())
            pairs.append({'topology': label, 'kappa': kappa, 'h_only': h,
                          'all_sites': all_sites, 'delta': delta,
                          'latency_ratio': all_sites['latency_us'] / h['latency_us']})
    return {
        'contrast': 'OL1(all) minus OL1(h) at matched gate topology and kappa',
        'sources_sha256': {**old['sources_sha256'],
            source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()},
        **{key: old[key] for key in ['coverage', 'seed', 'matched_initialization_and_schedule_sha256',
                                    'timing_protocol', 'loss_definition', 'sparsity_definition',
                                    'latency_definition', 'box_definition']},
        'limits': [
            'One shared training seed; boxes describe variation over five fixed kappa settings, not confidence intervals.',
            'Gate topology and kappa stay fixed; pressure target composition and equal-tensor objective normalization change.',
            'Four-site timings share Run029; seven-site all/h timings span Run029/Run033 physical GPU/host sessions.',
        ],
        'pairs': pairs,
    }


def draw(data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.spines.left': False, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15), sharey=True)
    summaries = []
    for panel, (ax, (key, title, xlabel)) in enumerate(zip(axes, METRICS)):
        ax.axvline(0, color='#78818C', linestyle='--', linewidth=.8, zorder=1)
        for row, (label, _, color) in enumerate(GROUPS):
            values = [p['delta'][key] for p in data['pairs'] if p['topology'] == label]
            assert len(values) == 5
            low, q1, median, q3, high = np.quantile(values, [0, .25, .5, .75, 1])
            stats = dict(whislo=float(low), q1=float(q1), med=float(median),
                         q3=float(q3), whishi=float(high), fliers=[])
            y = 1 - row
            ax.bxp([stats], positions=[y], widths=.30, orientation='horizontal',
                   showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops={'facecolor': color + '20', 'edgecolor': color, 'linewidth': 1.1},
                   medianprops={'color': color, 'linewidth': 1.6},
                   whiskerprops={'color': color, 'linewidth': 1.1},
                   capprops={'color': color, 'linewidth': 1.1}, zorder=2)
            ax.scatter(values, y + np.array([-.07, .07, -.035, .035, 0]),
                       s=19, color=color, edgecolors='white', linewidths=.45, zorder=3)
            summaries.append({'topology': label, 'metric': key, 'n': len(values),
                              **{k: v for k, v in stats.items() if k != 'fliers'},
                              'mean': float(np.mean(values))})
        ax.set_title(f'({chr(97 + panel)}) {title}', loc='left', pad=13)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_yticks([1, 0], [g[0] for g in GROUPS])
        ax.set_ylim(-.55, 1.55)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', length=3, width=.6, labelsize=8)
        ax.grid(axis='x', color='#E7E9ED', linewidth=.55)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-.025, .35)
    axes[0].set_xticks([0, .1, .2, .3], ['0', '0.10', '0.20', '0.30'])
    axes[1].set_xlim(-2.2, 12)
    axes[1].set_xticks([0, 5, 10])
    axes[2].set_xlim(-7, 65)
    axes[2].set_xticks([0, 20, 40, 60])
    fig.text(.535, .97, 'OL1(all) minus OL1(h) at matched ' + r'$\kappa$',
             ha='center', va='top', fontsize=10.5)
    fig.text(.535, .115, 'Box: middle 50%   |   Bar: median   |   Whiskers: full range   |   Dots: five ' + r'$\kappa$' + ' values',
             ha='center', fontsize=8, color='#46505B')
    fig.text(.535, .047, 'One seed; gate topology is fixed; seven-site timings span GPU sessions.',
             ha='center', fontsize=7.5, color='#626B75')
    fig.subplots_adjust(left=.115, right=.985, bottom=.34, top=.77, wspace=.28)
    for ax, (key, _, _) in zip(axes, METRICS):
        assert len(ax.collections) == 2
        assert sum(len(c.get_offsets()) for c in ax.collections) == 10
        assert all(ax.get_xlim()[0] < p['delta'][key] < ax.get_xlim()[1] for p in data['pairs'])
    output = HERE / 'figures/06-14m-paired-pressure-effects.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M: matched all-site minus h-only OL1 effects',
        'Creator': 'Analysis024 / 06_plot_paired_pressure.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    data['summaries'] = summaries
    data['geometric_mean_latency_change_percent'] = {
        label: 100 * math.expm1(np.mean([math.log(p['latency_ratio'])
               for p in data['pairs'] if p['topology'] == label])) for label, _, _ in GROUPS}
    return output


def main():
    data = collect()
    output = draw(data)
    (HERE / 'data/paired-pressure-effects.json').write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'10 matched pairs; 3 panels; 5 contrasts per topology: {output.name}')


if __name__ == '__main__':
    main()
