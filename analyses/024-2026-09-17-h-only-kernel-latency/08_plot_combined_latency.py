"""Combine the verified Figure03/04 cohorts without new measurements."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 64),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 27),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 30),
    ('A7', '7-sites (A7)', '#C96024', '^', 37),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [HERE/'data/final-latency-topology.json',
               HERE/'data/70m-final-latency-topology.json']
    data = [json.loads(p.read_text()) for p in sources]
    upstream = [HERE/data[0]['source'], ROOT/data[1]['source']]
    for row, source, latency in zip(data, upstream, ['k050_gm_ms', 'candidate_gm_ms']):
        assert row['source_sha256'] == sha(source)
        original = json.loads(source.read_text())['points']
        assert all(p['qualified'] for p in original)
        lookup = {(p.get('session'), p['condition']): p for p in original}
        for point in row['points']:
            raw = lookup[(point.get('session'), point['condition'])]
            assert point['family'] == raw['family']
            assert point[latency] == raw[latency] and math.isfinite(point[latency])
            counts = raw['canonical_counts']
            expected = 100*counts['block_zero_product_count']/counts['model_product_count']
            assert math.isclose(point['sparsity_percent'], expected, rel_tol=0, abs_tol=1e-12)
    assert [len(d['points']) for d in data] == [36, 22]
    assert [len(d['connections']) for d in data] == [7, 4]
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.2))
    panels = []
    for ax, title, record, latency_key, xmax, expected in zip(
            axes, ['14M', '70M'], data, ['k050_gm_ms', 'candidate_gm_ms'],
            [30, 45], [[1, 5, 15, 15], [1, 1, 10, 10]]):
        points = record['points']
        def identity(p):
            return (p.get('session'), p['condition'])
        by_id = {identity(p): p for p in points}
        assert len(by_id) == len(points)
        for family in record['connections']:
            ids = [identity(p) if isinstance(p, dict) else (None, p)
                   for p in family['conditions']]
            ordered = [by_id[i] for i in ids]
            assert all(p['family'] == family['family'] for p in ordered)
            color = next(g[2] for g in GROUPS if family['family'].startswith(g[0]))
            ax.plot([p['sparsity_percent'] for p in ordered],
                    [p[latency_key] for p in ordered], color=color,
                    linestyle='--', linewidth=1, alpha=.8, zorder=2)
        for (prefix, label, color, marker, size), count in zip(GROUPS, expected):
            selected = [p for p in points if p['family'].startswith(prefix)]
            assert len(selected) == count
            ax.scatter([p['sparsity_percent'] for p in selected],
                       [p[latency_key] for p in selected], label=label,
                       color=color, marker=marker, s=size,
                       linewidths=.5, edgecolors='white', zorder=3)
        for a in record['annotations']:
            color = next(g[2] for g in GROUPS if a['family'].startswith(g[0]))
            if 'text_position' in a:
                alignment = 'right' if a['family'] in {'A4', 'A4+OL1@h'} else 'left'
                ax.text(*a['text_position'], a['text'], color=color, fontsize=6.5,
                        alpha=.85, ha=alignment, va='center')
            else:
                p = by_id[(None, a['condition'])]
                ax.annotate(a['text'], (p['sparsity_percent'], p[latency_key]),
                            xytext=a['offset_points'], textcoords='offset points',
                            color=color, fontsize=6.5, alpha=.85,
                            ha=a['alignment'], va='center')
        ax.set(xlim=(-.6, xmax), ylim=record['y_limits_ms'],
               ylabel='Full-model latency (ms)')
        ax.set_title(title, fontsize=11, pad=11)
        ax.set_xticks(range(0, xmax+1, 5))
        ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out', length=3, width=.6)
        assert sum(len(c.get_offsets()) for c in ax.collections) == len(points)
        assert len(ax.lines) == len(record['connections'])
        assert all(ax.get_xlim()[0] < p['sparsity_percent'] < ax.get_xlim()[1]
                   and ax.get_ylim()[0] < p[latency_key] < ax.get_ylim()[1] for p in points)
        panels.append({'model': title, 'x_limits_percent': list(ax.get_xlim()),
                       'y_limits_ms': list(ax.get_ylim()),
                       'points': [{'condition': p['condition'], 'session': p.get('session'),
                                   'family': p['family'], 'sparsity_percent': p['sparsity_percent'],
                                   'latency_ms': p[latency_key]} for p in points],
                       'connections': record['connections'], 'annotations': record['annotations']})
    fig.supxlabel(r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
                  fontsize=11, x=.53, y=.125)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=4,
               frameon=False, fontsize=8.5, handletextpad=.4,
               columnspacing=1.8, bbox_to_anchor=(.53, .01))
    fig.subplots_adjust(left=.072, right=.985, bottom=.25, top=.89, wspace=.27)
    output = HERE/'figures/08-14m-70m-final-sparsity-latency.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M and Pythia-70M: final kernel latency versus sparsity',
        'Creator': 'Analysis024 / 08_plot_combined_latency.py',
        'CreationDate': None, 'ModDate': None,
    })
    plt.close(fig)
    result = {
        'sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sources],
        'upstream_sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in upstream],
        'display': 'Two linear panels with independently fitted axes; common units, colors and markers; separate dashed recipe families.',
        'checkpoint_count': 58, 'panels': panels,
        'comparison_limits': '14M uses K050;70M uses the qualified k050-70m-v2 port. Matched workload/timing definition, different physical GPU/host sessions and shape-specific implementation; no equal optimization-budget claim.',
        'pdf_sha256': sha(output),
    }
    (HERE/'data/combined-final-latency.json').write_text(
        json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('58 checkpoints;36 at14M and22 at70M;7+4 separate family curves')


if __name__ == '__main__':
    main()
