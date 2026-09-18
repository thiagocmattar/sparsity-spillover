"""Final K050 latency versus model-wide sparsity, grouped by topology."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 72, 1),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 30, 5),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 34, 15),
    ('A7', '7-sites (A7)', '#C96024', '^', 42, 15),
]
# Small labels sit beside their curves without boxes or leader arrows.
LABELS = [
    ('A1-H+OL1', 'OL1(h)', (4.4, .5985), 'left'),
    ('A4', 'no pressure', (9.65, .488), 'right'),
    ('A4+OL1@h', 'OL1(h)', (9.8, .4525), 'right'),
    ('A4+OL1@4', 'OL1(all)', (13.2, .456), 'left'),
    ('A7', 'no pressure', (13.65, .528), 'left'),
    ('A7+OL1@h', 'OL1(h)', (17.15, .472), 'left'),
    ('A7+OL1@7', 'OL1(all)', (21.2, .536), 'left'),
]


def main():
    source = HERE / 'data/results.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert len(data['points']) == 40 and all(p['qualified'] for p in data['points'])
    points = [p for p in data['points'] if p['family'] != 'A1-H+L1']
    assert len(points) == 36
    records, connections, annotations = [], [], []
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for prefix, label, color, marker, size, expected_count in GROUPS:
        selected = [p for p in points if p['family'].startswith(prefix)]
        assert len(selected) == expected_count
        for point in selected:
            counts = point['canonical_counts']
            sparsity = 100 * counts['block_zero_product_count'] / counts['model_product_count']
            assert math.isclose(sparsity, point['sparsity_percent'], rel_tol=0, abs_tol=1e-12)
            assert math.isfinite(point['k050_gm_ms']) and point['k050_gm_ms'] > 0
            records.append({
                'condition': point['condition'], 'family': point['family'],
                'group': label, 'session': point['session'],
                'sparsity_percent': sparsity, 'k050_gm_ms': point['k050_gm_ms'],
            })
        for family in sorted({p['family'] for p in selected}):
            family_points = [p for p in selected if p['family'] == family]
            if len(family_points) == 1:
                continue
            ordered = sorted(family_points, key=lambda p: p['kappa_or_lambda'])
            ax.plot([p['sparsity_percent'] for p in ordered],
                    [p['k050_gm_ms'] for p in ordered],
                    color=color, linestyle='--', linewidth=1.0, alpha=.8, zorder=2)
            connections.append({'group': label, 'family': family,
                                'order': 'increasing kappa or lambda within this recipe',
                                'kappa_or_lambda': [p['kappa_or_lambda'] for p in ordered],
                                'conditions': [{'session': p['session'], 'condition': p['condition']}
                                               for p in ordered]})
        ax.scatter([p['sparsity_percent'] for p in selected],
                   [p['k050_gm_ms'] for p in selected],
                   label=label, color=color, marker=marker, s=size,
                   linewidths=.5, edgecolors='white', zorder=3)
    assert len(records) == 36
    assert len({(r['session'], r['condition']) for r in records}) == 36
    latencies = [r['k050_gm_ms'] for r in records]
    padding = .08 * (max(latencies) - min(latencies))
    y_limits = (math.floor(100 * (min(latencies) - padding)) / 100,
                math.ceil(100 * (max(latencies) + padding)) / 100)
    ax.set(xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel='Full-model latency (ms)', xlim=(-.6, 30), ylim=y_limits)
    ax.set_xticks(range(0, 31, 5))
    ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)
    for family, text, position, alignment in LABELS:
        color = next(g[2] for g in GROUPS if family.startswith(g[0]))
        ax.text(*position, text, color=color, fontsize=6.8, alpha=.85,
                ha=alignment, va='center', zorder=5)
        annotations.append({'family': family, 'text': text,
                            'text_position': position})
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4,
               frameon=False, fontsize=8.2, handletextpad=.4,
               columnspacing=1.35, bbox_to_anchor=(.53, .025))
    fig.subplots_adjust(left=.11, right=.985, bottom=.23, top=.965)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 36
    assert len(ax.lines) == 7 and sum(len(line.get_xdata()) for line in ax.lines) == 34
    assert all(line.get_linestyle() == '--' for line in ax.lines)
    assert len(annotations) == 7
    assert {a['family'] for a in annotations} == {c['family'] for c in connections}
    assert all(ax.get_xlim()[0] < r['sparsity_percent'] < ax.get_xlim()[1]
               and ax.get_ylim()[0] < r['k050_gm_ms'] < ax.get_ylim()[1] for r in records)
    output = HERE / 'figures/03-14m-k050-sparsity-latency-topology.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M: final K050 latency by topology',
        'Creator': 'Analysis024 / 03_plot_final_latency.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    reduction = {
        'source': 'data/results.json',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'protocol': data['protocol'], 'latency_definition': data['latency_definition'],
        'cross_session_limit': data['cross_session_limit'],
        'display': 'Final K050 only; four topology labels; dashed recipe-family curves with small pressure labels, no arrows or boxes; no cross-recipe connections or fits; latency axis fitted to data with padding.',
        'source_checkpoint_count': 40,
        'excluded_families': ['A1-H+L1'],
        'plotted_checkpoint_count': 36,
        'y_limits_ms': y_limits,
        'annotations': annotations,
        'connections': connections,
        'points': records,
    }
    (HERE / 'data/final-latency-topology.json').write_text(
        json.dumps(reduction, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('36 qualified checkpoints; groups 1/5/15/15; single-site naive L1 excluded')


if __name__ == '__main__':
    main()
