"""Re-express Figure09 latency as speedup over the size-matched candidate A0."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 72),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 32),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 32),
    ('A7', '7-sites (A7)', '#C96024', '^', 42),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE/'data/matched-combined-latency.json'
    retained = json.loads(source.read_text())
    for item in retained['sources'] + retained['verified_against']:
        assert sha(ROOT/item['path']) == item['sha256']
    assert sha(HERE/'figures/09-14m-70m-matched-sparsity-latency.pdf') == retained['pdf_sha256']
    points = retained['points']
    assert len(points) == 44 and all(p['qualified'] for p in points)
    baselines = {}
    keys = []
    for model in ['14M', '70M']:
        cohort = [p for p in points if p['model'] == model]
        a0 = [p for p in cohort if p['family'] == 'A0']
        assert len(cohort) == 22 and len(a0) == 1
        keys.append({(p['family'], p['kappa']) for p in cohort})
        baseline = a0[0]
        baselines[model] = {k: baseline[k] for k in ['condition', 'session', 'latency_ms']}
        for point in cohort:
            assert math.isfinite(point['latency_ms']) and point['latency_ms'] > 0
            point['baseline_latency_ms'] = baseline['latency_ms']
            point['speedup_vs_a0'] = baseline['latency_ms']/point['latency_ms']
        assert baseline['speedup_vs_a0'] == 1.0
    assert keys[0] == keys[1] and len(keys[0]) == 22
    by_id = {(p['model'], p['condition']): p for p in points}
    assert len(by_id) == 44
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11,
        'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for curve in retained['connections']:
        ordered = [by_id[(curve['model'], c)] for c in curve['conditions']]
        assert [p['kappa'] for p in ordered] == retained['kappas']
        assert all(p['family'] == curve['family'] for p in ordered)
        color = next(g[2] for g in GROUPS if curve['family'].startswith(g[0]))
        ax.plot([p['sparsity_percent'] for p in ordered],
                [p['speedup_vs_a0'] for p in ordered], color=color,
                linestyle=(0, curve['dash_pattern_points']), linewidth=1.15,
                alpha=.85, zorder=2)
    for model in ['14M', '70M']:
        for prefix, _, color, marker, size in GROUPS:
            selected = [p for p in points if p['model'] == model
                        and p['family'].split('+')[0] == prefix]
            assert len(selected) == (1 if prefix in ['A0', 'A1-H'] else 10)
            # The two A0 values coincide at 1x: retain both using nested stars.
            if prefix == 'A0':
                size = 110 if model == '14M' else 38
            ax.scatter([p['sparsity_percent'] for p in selected],
                       [p['speedup_vs_a0'] for p in selected],
                       facecolors='white' if model == '14M' else color,
                       edgecolors=color, marker=marker, s=size,
                       linewidths=1.05 if model == '14M' else .7, zorder=3)
    ymin = min(p['speedup_vs_a0'] for p in points)
    ymax = max(p['speedup_vs_a0'] for p in points)
    padding = (ymax/ymin)**.07
    ax.set(xlim=(-.8, 42.5), yscale='log', ylim=(ymin/padding, ymax*padding),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel='Full-model speedup vs. A0 (log scale)')
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks([1, 1.2, 1.5, 2])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'+r'$\times$'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
    ax.axhline(1, color='#A7ADB5', linewidth=.7, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)
    topology_handles = [Line2D([], [], color=color, marker=marker,
                              markersize=math.sqrt(size), linestyle='none', label=label)
                        for _, label, color, marker, size in GROUPS]
    style_handles = [
        Line2D([], [], color='#444A52', marker='o', markersize=5.5,
               markerfacecolor='white', linestyle='none', label='14M'),
        Line2D([], [], color='#444A52', marker='o', markersize=5.5,
               linestyle='none', label='70M'),
        Line2D([], [], color='#444A52', linestyle=(0, (2, 2)), label='OL1(h)'),
        Line2D([], [], color='#444A52', linestyle=(0, (6, 3)), label='OL1(all)'),
    ]
    fig.legend(handles=topology_handles, loc='lower center', ncol=4,
               frameon=False, fontsize=9, handletextpad=.4, columnspacing=1.6,
               bbox_to_anchor=(.55, .054))
    fig.legend(handles=style_handles, loc='lower center', ncol=4,
               frameon=False, fontsize=9, handletextpad=.6, columnspacing=2.2,
               bbox_to_anchor=(.55, .005))
    fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.97)
    assert len(fig.axes) == 1 and len(ax.lines) == 9  # Eight curves plus the A0 guide.
    assert ax.get_yscale() == 'log'
    assert sum(len(c.get_offsets()) for c in ax.collections) == 44
    assert all(ax.get_xlim()[0] < p['sparsity_percent'] < ax.get_xlim()[1]
               and ax.get_ylim()[0] < p['speedup_vs_a0'] < ax.get_ylim()[1] for p in points)
    output = HERE/'figures/10-14m-70m-matched-sparsity-a0-speedup.pdf'
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M and Pythia-70M speedup over size-matched candidate A0',
        'Creator': 'Analysis024 / 10_plot_a0_normalized_speedup.py',
        'CreationDate': None, 'ModDate': None,
    })
    record = {
        'source': {'path': source.relative_to(ROOT).as_posix(), 'sha256': sha(source)},
        'upstream_sources': retained['sources'],
        'definition': 'candidate A0 geometric-mean latency / candidate checkpoint geometric-mean latency, separately within each model size',
        'baseline_kind': 'final candidate kernel on A0, not the checkpoint-specific native reference',
        'baselines': baselines, 'checkpoint_count': 44, 'checkpoint_count_per_model': 22,
        'kappas': retained['kappas'], 'points': points, 'connections': retained['connections'],
        'y_scale': 'log', 'y_unit': 'x', 'y_ticks': list(ax.get_yticks()),
        'x_limits_percent': list(ax.get_xlim()), 'y_limits_speedup': list(ax.get_ylim()),
        'baseline_marker_areas_pt2': {'14M': 110, '70M': 38},
        'comparison_limits': retained['comparison_limits'] + ' Shared A0 normalization is descriptive, not a paired native-relative kernel speedup or an equal-quality comparison; 14M A7 h-only points span a different session from A0.',
        'pdf_sha256': sha(output),
    }
    plt.close(fig)
    (HERE/'data/matched-a0-normalized-speedup.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('44 points; eight curves; both A0 controls equal 1x; separate size-matched baselines.')
    for model in baselines:
        print(model, baselines[model], 'maximum speedup:',
              max(p['speedup_vs_a0'] for p in points if p['model'] == model))


if __name__ == '__main__':
    main()
