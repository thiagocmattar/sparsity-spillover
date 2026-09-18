"""Five matched A7-minus-A4 contrasts per recipe, shown as horizontal boxes."""
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
KAPPAS = [0, .01, .05, .1, .5]
RECIPES = [
    ('No pressure', 'A4', 'A7', 'A4', 'A7', '#53616F'),
    ('OL1(h)', 'A4-OL1-H', 'A7-OL1-H', 'A4+OL1@h', 'A7+OL1@h', '#19856B'),
    ('OL1(all)', 'A4-OL1', 'A7-OL1', 'A4+OL1@4', 'A7+OL1@7', '#7560A7'),
]
METRICS = [
    ('loss_nats', 'Validation loss', r'$\Delta$ loss (nats/token)'),
    ('sparsity_pp', 'Model-wide sparsity', r'$\Delta\mathcal{S}_{\mathrm{model}}$ (percentage points)'),
    ('latency_us', 'K050 latency', r'$\Delta$ latency ($\mu$s)'),
]


def collect():
    quality_path = ROOT / 'analyses/023-2026-09-17-14m-pressure-targets-paper-table/results.json'
    timing_path = HERE / 'data/results.json'
    quality = json.loads(quality_path.read_text(encoding='utf-8'))
    timing = json.loads(timing_path.read_text(encoding='utf-8'))
    q_index = {(p['family'], p['kappa']): p for p in quality['rows']}
    t_index = {(p['family'], p['kappa_or_lambda']): p for p in timing['points']}
    assert len(q_index) == len(quality['rows']) == 30
    assert len(t_index) == len(timing['points']) == 40
    assert quality['coverage']['sequences'] == timing['protocol']['validation_blocks'] == 338
    assert quality['coverage']['documents'] == timing['protocol']['validation_documents'] == 500
    assert quality['coverage']['excluded_tail_tokens'] == 1444
    pairs = []
    for label, q4, q7, t4, t7, _ in RECIPES:
        for kappa in KAPPAS:
            endpoints = []
            for q_family, t_family in [(q4, t4), (q7, t7)]:
                q, t = q_index[q_family, kappa], t_index[t_family, kappa]
                counts = t['canonical_counts']
                assert t['qualified'] and all(r['qualified'] for r in t['replicates'])
                assert counts['block_zero_product_count'] == q['block_zero_product_count']
                assert counts['model_product_count'] == q['model_product_count']
                weight = next(f for f in t['checkpoint_files'] if f['path'].endswith('/model.safetensors'))
                assert weight['path'].startswith(q['source_attempt'] + '/')
                endpoints.append({
                    'family': q_family, 'source_attempt': q['source_attempt'],
                    'checkpoint_content_sha256': q['checkpoint_content_sha256'],
                    'timing_condition': t['condition'], 'timing_session': t['session'],
                    'block_zero_product_count': counts['block_zero_product_count'],
                    'model_product_count': counts['model_product_count'],
                    'loss_nats': q['ordinary_final_validation_loss'],
                    'sparsity_pp': 100 * counts['block_zero_product_count'] / counts['model_product_count'],
                    'latency_us': 1000 * t['k050_gm_ms'],
                })
            a4, a7 = endpoints
            assert a4['model_product_count'] == a7['model_product_count']
            assert t_index[t4, kappa]['timing_block_indices'] == t_index[t7, kappa]['timing_block_indices']
            delta = {key: a7[key] - a4[key] for key, _, _ in METRICS}
            assert all(math.isfinite(value) for value in delta.values())
            pairs.append({'recipe': label, 'kappa': kappa, 'a4': a4, 'a7': a7, 'delta': delta})
    return {
        'contrast': '7-sites minus 4-sites at matched pressure recipe and kappa',
        'sources_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [quality_path, timing_path]},
        'coverage': quality['coverage'], 'seed': quality['seed'],
        'matched_initialization_and_schedule_sha256': quality['matched_initialization_and_schedule_sha256'],
        'timing_protocol': timing['protocol'],
        'loss_definition': 'Ordinary final validation for every endpoint; not the mixed-pass archival table.',
        'sparsity_definition': '100 times pooled zero-product counts / model-product counts; differences in percentage points.',
        'latency_definition': 'Difference of geometric means of 1344 K050 full-logit host timings per checkpoint, converted from ms to microseconds.',
        'box_definition': '25th and 75th percentiles (numpy linear quantiles); median bar; min/max whiskers; all five kappa contrasts as dots.',
        'limits': [
            'One shared training seed; boxes describe variation across fixed kappa settings, not confidence intervals.',
            'OL1(all) expands both gate and pressure scope from four to seven sites.',
            'OL1(h) absolute latency comparisons span different physical GPU/host sessions.',
        ],
        'pairs': pairs,
    }


def draw(data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.spines.left': False, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.45), sharey=True)
    summaries = []
    for panel, (ax, (key, title, xlabel)) in enumerate(zip(axes, METRICS)):
        ax.axvline(0, color='#78818C', linestyle='--', linewidth=.8, zorder=1)
        for row, (label, _, _, _, _, color) in enumerate(RECIPES):
            values = [p['delta'][key] for p in data['pairs'] if p['recipe'] == label]
            assert len(values) == 5
            low, q1, median, q3, high = np.quantile(values, [0, .25, .5, .75, 1])
            stats = dict(whislo=float(low), q1=float(q1), med=float(median),
                         q3=float(q3), whishi=float(high), fliers=[])
            y = 2 - row
            ax.bxp([stats], positions=[y], widths=.30, orientation='horizontal',
                   showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops={'facecolor': color + '20', 'edgecolor': color, 'linewidth': 1.1},
                   medianprops={'color': color, 'linewidth': 1.6},
                   whiskerprops={'color': color, 'linewidth': 1.1},
                   capprops={'color': color, 'linewidth': 1.1}, zorder=2)
            # Fixed small offsets keep close contrasts visible; y has no numeric meaning.
            ax.scatter(values, y + np.array([-.07, .07, -.035, .035, 0]),
                       s=19, color=color, edgecolors='white', linewidths=.45, zorder=3)
            summaries.append({'recipe': label, 'metric': key, 'n': len(values),
                              **{k: v for k, v in stats.items() if k != 'fliers'},
                              'mean': float(np.mean(values))})
        ax.set_title(f'({chr(97 + panel)}) {title}', loc='left', pad=13)
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_yticks([2, 1, 0], [r[0] for r in RECIPES])
        ax.set_ylim(-.55, 2.55)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.tick_params(axis='x', length=3, width=.6, labelsize=8)
        ax.grid(axis='x', color='#E7E9ED', linewidth=.55)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-.235, .065)
    axes[0].set_xticks([-.2, -.1, 0, .05], ['-0.20', '-0.10', '0', '0.05'])
    axes[1].set_xlim(-1.8, 16.3)
    axes[1].set_xticks([0, 5, 10, 15])
    axes[2].set_xlim(-4, 68)
    axes[2].set_xticks([0, 20, 40, 60])
    fig.text(.535, .97, '7-sites minus 4-sites at matched ' + r'$\kappa$',
             ha='center', va='top', fontsize=10.5)
    fig.text(.535, .115, 'Box: middle 50%   |   Bar: median   |   Whiskers: full range   |   Dots: five ' + r'$\kappa$' + ' values',
             ha='center', fontsize=8, color='#46505B')
    fig.text(.535, .047, 'One seed; h-only timings span GPU sessions; OL1(all) expands pressure targets.',
             ha='center', fontsize=7.5, color='#626B75')
    fig.subplots_adjust(left=.115, right=.985, bottom=.32, top=.79, wspace=.28)
    for ax, (key, _, _) in zip(axes, METRICS):
        assert len(ax.collections) == 3
        assert sum(len(c.get_offsets()) for c in ax.collections) == 15
        assert all(ax.get_xlim()[0] < p['delta'][key] < ax.get_xlim()[1] for p in data['pairs'])
    output = HERE / 'figures/05-14m-paired-topology-effects.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M: matched seven-minus-four-site contrasts',
        'Creator': 'Analysis024 / 05_plot_paired_topology.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    data['summaries'] = summaries
    return output


def main():
    data = collect()
    output = draw(data)
    (HERE / 'data/paired-topology-effects.json').write_text(
        json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'15 matched pairs; 3 panels; 5 contrasts per recipe: {output.name}')


if __name__ == '__main__':
    main()
