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
# Direct labels: recipe, text, text position, alignment, optional final-segment fraction.
LABELS = [
    ('A1-H', 'no pressure', (2.4, .653), 'center', None),
    ('A1-H+L1', 'L1', (1.2, .573), 'center', None),
    ('A1-H+OL1', 'OL1(h)', (5.3, .582), 'left', None),
    ('A4', 'no pressure', (6.3, .501), 'right', None),
    ('A4+OL1@h', 'OL1(h)', (6.5, .452), 'left', None),
    ('A4+OL1@4', 'OL1(all)', (13.6, .449), 'left', None),
    ('A7', 'no pressure', (17.4, .516), 'left', .72),
    ('A7+OL1@h', 'OL1(h)', (18.6, .466), 'left', None),
    ('A7+OL1@7', 'OL1(all)', (22.0, .551), 'left', .55),
]


def main():
    source = HERE / 'data/results.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    points = data['points']
    assert len(points) == 40 and all(p['qualified'] for p in points)
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
    for family, text, position, alignment, fraction in LABELS:
        rows = sorted([p for p in points if p['family'] == family],
                      key=lambda p: -1 if p['kappa_or_lambda'] is None else p['kappa_or_lambda'])
        anchor = (rows[-1]['sparsity_percent'], rows[-1]['k050_gm_ms'])
        if fraction is not None:
            previous = (rows[-2]['sparsity_percent'], rows[-2]['k050_gm_ms'])
            anchor = tuple(a + fraction * (b - a) for a, b in zip(previous, anchor))
        color = next(g[2] for g in GROUPS if family.startswith(g[0]))
        ax.annotate(text, xy=anchor, xytext=position, color=color,
                    fontsize=8.1, ha=alignment, va='center', zorder=5,
                    bbox={'boxstyle': 'square,pad=.12', 'fc': 'white', 'ec': 'none', 'alpha': .95},
                    arrowprops={'arrowstyle': '->', 'color': color, 'lw': .65,
                                'shrinkA': 3, 'shrinkB': 4, 'mutation_scale': 6})
        annotations.append({'family': family, 'text': text, 'anchor': anchor,
                            'text_position': position})
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', ncol=4,
               frameon=False, fontsize=8.2, handletextpad=.4,
               columnspacing=1.35, bbox_to_anchor=(.53, .025))
    fig.subplots_adjust(left=.11, right=.985, bottom=.23, top=.965)
    assert sum(len(c.get_offsets()) for c in ax.collections) == 40
    assert len(ax.lines) == 8 and sum(len(line.get_xdata()) for line in ax.lines) == 38
    assert all(line.get_linestyle() == '--' for line in ax.lines)
    assert len(annotations) == 9
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
        'display': 'Final K050 only; four topology labels; dashed recipe-family curves in increasing kappa or lambda order with direct pressure labels; no cross-recipe connections or fits; latency axis fitted to data with padding.',
        'y_limits_ms': y_limits,
        'annotations': annotations,
        'connections': connections,
        'points': records,
    }
    (HERE / 'data/final-latency-topology.json').write_text(
        json.dumps(reduction, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('40 qualified checkpoints; groups 1/9/15/15; final K050 latency only')


if __name__ == '__main__':
    main()
