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
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 30, 9),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 34, 15),
    ('A7', '7-sites (A7)', '#C96024', '^', 42, 15),
]


def main():
    source = HERE / 'data/results.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    points = data['points']
    assert len(points) == 40 and all(p['qualified'] for p in points)
    records, connections = [], []
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
        ordered = sorted(selected, key=lambda p: (p['sparsity_percent'], p['family'], p['condition']))
        if len(ordered) > 1:
            ax.plot([p['sparsity_percent'] for p in ordered],
                    [p['k050_gm_ms'] for p in ordered],
                    color=color, linewidth=1.0, alpha=.8, zorder=2)
            connections.append({'group': label, 'order': 'increasing model-wide sparsity',
                                'conditions': [{'session': p['session'], 'condition': p['condition']}
                                               for p in ordered]})
        ax.scatter([p['sparsity_percent'] for p in selected],
                   [p['k050_gm_ms'] for p in selected],
                   label=label, color=color, marker=marker, s=size,
                   linewidths=.5, edgecolors='white', zorder=3)
    assert len(records) == 40
    assert len({(r['session'], r['condition']) for r in records}) == 40
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
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4,
               frameon=False, fontsize=8.2, handletextpad=.4,
               columnspacing=1.35, bbox_to_anchor=(.53, .025))
    fig.subplots_adjust(left=.11, right=.985, bottom=.23, top=.965)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 40
    assert len(ax.lines) == 3 and sum(len(line.get_xdata()) for line in ax.lines) == 39
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
        'display': 'Final K050 only; four topology labels; markers in each legend group connected in increasing sparsity order; no fits or point annotations; latency axis fitted to data with padding.',
        'y_limits_ms': y_limits,
        'connections': connections,
        'points': records,
    }
    (HERE / 'data/final-latency-topology.json').write_text(
        json.dumps(reduction, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('40 qualified checkpoints; groups 1/9/15/15; final K050 latency only')


if __name__ == '__main__':
    main()
