"""All-site and h-only OL1 minus no pressure: a two-row paired-effect figure."""
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
PRESSURES = ['OL1(all)', 'OL1(h)']
GROUPS = [('4-sites', 'a4', '#2878B5'), ('7-sites', 'a7', '#C96024')]
METRICS = [
    ('loss_nats', 'Validation loss', r'$\Delta$ loss (nats/token)'),
    ('sparsity_pp', 'Model-wide sparsity', r'$\Delta\mathcal{S}_{\mathrm{model}}$ (percentage points)'),
    ('latency_us', 'K050 latency', r'$\Delta$ latency ($\mu$s)'),
]


def collect():
    source = HERE / 'data/paired-topology-effects.json'
    old = json.loads(source.read_text(encoding='utf-8'))
    for name, sha in old['sources_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    lookup = {(p['recipe'], p['kappa']): p for p in old['pairs']}
    assert len(lookup) == len(old['pairs']) == 15
    pairs = []
    for pressure in PRESSURES:
        for label, topology, _ in GROUPS:
            for kappa in [0, .01, .05, .1, .5]:
                baseline = lookup['No pressure', kappa][topology]
                intervention = lookup[pressure, kappa][topology]
                assert intervention['model_product_count'] == baseline['model_product_count']
                delta = {key: intervention[key] - baseline[key] for key, _, _ in METRICS}
                assert math.isclose(delta['sparsity_pp'],
                    100 * (intervention['block_zero_product_count'] - baseline['block_zero_product_count'])
                    / baseline['model_product_count'], abs_tol=1e-12)
                assert all(math.isfinite(v) for v in delta.values())
                pairs.append({'pressure': pressure, 'topology': label, 'kappa': kappa,
                              'no_pressure': baseline, 'with_pressure': intervention,
                              'delta': delta, 'latency_ratio': intervention['latency_us'] / baseline['latency_us']})
    return {
        'contrast': 'Pressure minus no pressure at matched gate topology and kappa',
        'row_order': PRESSURES, 'topology_order': [g[0] for g in GROUPS],
        'sources_sha256': {**old['sources_sha256'],
            source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()},
        **{key: old[key] for key in ['coverage', 'seed', 'matched_initialization_and_schedule_sha256',
                                    'timing_protocol', 'loss_definition', 'sparsity_definition',
                                    'latency_definition', 'box_definition']},
        'limits': [
            'One shared training seed; boxes describe variation across five fixed kappa settings, not confidence intervals.',
            'The two pressure contrasts reuse the same no-pressure reference within each topology and kappa.',
            'Gate topology and kappa stay fixed; pressure is added at all gated sites or at h only.',
            'Only seven-site h-only latency contrasts span Run033/Run029 physical GPU/host sessions; all other timing pairs share Run029.',
        ],
        'pairs': pairs,
    }


def draw(data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.spines.left': False, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.7), sharex='col', sharey=True)
    summaries = []
    for row, pressure in enumerate(PRESSURES):
        for col, (key, title, xlabel) in enumerate(METRICS):
            ax = axes[row, col]
            ax.axvline(0, color='#78818C', linestyle='--', linewidth=.8, zorder=1)
            for group, (label, _, color) in enumerate(GROUPS):
                selected = [p for p in data['pairs'] if p['pressure'] == pressure and p['topology'] == label]
                values = [p['delta'][key] for p in selected]
                assert len(values) == 5
                low, q1, median, q3, high = np.quantile(values, [0, .25, .5, .75, 1])
                stats = dict(whislo=float(low), q1=float(q1), med=float(median),
                             q3=float(q3), whishi=float(high), fliers=[])
                y = 1 - group
                ax.bxp([stats], positions=[y], widths=.30, orientation='horizontal',
                       showfliers=False, patch_artist=True, manage_ticks=False,
                       boxprops={'facecolor': color + '20', 'edgecolor': color, 'linewidth': 1.1},
                       medianprops={'color': color, 'linewidth': 1.6},
                       whiskerprops={'color': color, 'linewidth': 1.1},
                       capprops={'color': color, 'linewidth': 1.1}, zorder=2)
                ax.scatter(values, y + np.array([-.07, .07, -.035, .035, 0]),
                           s=19, color=color, edgecolors='white', linewidths=.45, zorder=3)
                summaries.append({'pressure': pressure, 'topology': label, 'metric': key, 'n': len(values),
                                  **{k: v for k, v in stats.items() if k != 'fliers'},
                                  'mean': float(np.mean(values))})
            ax.set_title(f'({chr(97 + row * 3 + col)}) {title}', loc='left', pad=10)
            if row == 1:
                ax.set_xlabel(xlabel, labelpad=8)
            ax.set_yticks([1, 0], [g[0] for g in GROUPS])
            ax.set_ylim(-.55, 1.55)
            ax.tick_params(axis='y', length=0, pad=8)
            ax.tick_params(axis='x', length=3, width=.6, labelsize=8, labelbottom=True)
            ax.grid(axis='x', color='#E7E9ED', linewidth=.55)
            ax.set_axisbelow(True)
    axes[0, 0].set_xlim(-.30, .42)
    axes[0, 0].set_xticks([-.2, 0, .2, .4], ['-0.20', '0', '0.20', '0.40'])
    axes[0, 1].set_xlim(-.9, 13.2)
    axes[0, 1].set_xticks([0, 5, 10])
    axes[0, 2].set_xlim(-75, 8)
    axes[0, 2].set_xticks([-60, -40, -20, 0])
    for y, pressure in zip([.96, .55], PRESSURES):
        fig.text(.115, y, pressure + ' minus no pressure', ha='left', va='top', fontsize=11,
                 fontweight='bold')
    fig.text(.535, .075, 'Box: middle 50%   |   Bar: median   |   Whiskers: full range   |   Dots: five matched ' + r'$\kappa$' + ' values',
             ha='center', fontsize=8, color='#46505B')
    fig.text(.535, .034, 'One seed; fixed gate topology; only 7-site OL1(h) timings span GPU sessions.',
             ha='center', fontsize=7.5, color='#626B75')
    fig.subplots_adjust(left=.115, right=.985, bottom=.20, top=.85, wspace=.28, hspace=.65)
    for row, pressure in enumerate(PRESSURES):
        for ax, (key, _, _) in zip(axes[row], METRICS):
            assert len(ax.collections) == 2
            assert sum(len(c.get_offsets()) for c in ax.collections) == 10
            assert all(ax.get_xlim()[0] < p['delta'][key] < ax.get_xlim()[1]
                       for p in data['pairs'] if p['pressure'] == pressure)
    output = HERE / 'figures/07-14m-pressure-vs-none-effects.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M: all-site and h-only OL1 versus no pressure',
        'Creator': 'Analysis024 / 07_plot_pressure_vs_none.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    data['summaries'] = summaries
    return output


def main():
    data = collect()
    output = draw(data)
    (HERE / 'data/pressure-vs-none-effects.json').write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'20 matched pairs; 2 by 3 panels; 5 contrasts per box: {output.name}')


if __name__ == '__main__':
    main()
