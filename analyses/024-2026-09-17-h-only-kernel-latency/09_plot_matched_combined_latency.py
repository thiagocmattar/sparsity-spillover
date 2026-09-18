"""Plot exactly the 22 shared recipe/kappa conditions at each model size."""
import argparse
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
KAPPAS = [0, .01, .05, .1, .5]
RECIPES = ['A4+OL1@h', 'A4+OL1@all', 'A7+OL1@h', 'A7+OL1@all']
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 72),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 32),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 32),
    ('A7', '7-sites (A7)', '#C96024', '^', 42),
]
DASHES = {'h': (2, 2), 'all': (6, 3)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_points():
    sources = [HERE/'data/results.json',
               ROOT/'runs/035-2026-09-18-pythia70m-k050-port/results/70m-final-kernel.json']
    references = [HERE/'data/final-latency-topology.json',
                  HERE/'data/70m-final-latency-topology.json']
    expected = {('A0', None), ('A1-H', None)} | {
        (recipe, kappa) for recipe in RECIPES for kappa in KAPPAS}
    points = []
    for model, source, reference, latency_key, kappa_key in zip(
            ['14M', '70M'], sources, references,
            ['k050_gm_ms', 'candidate_gm_ms'], ['kappa_or_lambda', 'kappa']):
        assert json.loads(reference.read_text())['source_sha256'] == sha(source)
        record = json.loads(source.read_text())
        if model == '70M':
            assert record['status'] == 'complete_verified_reduction'
            assert record['candidate'] == 'k050-70m-v2'
        selected = []
        for p in record['points']:
            family = {'A4+OL1@4': 'A4+OL1@all', 'A7+OL1@7': 'A7+OL1@all'}.get(
                p['family'], p['family'])
            if family not in ['A0', 'A1-H', *RECIPES]:
                continue
            assert p['qualified'] is True
            counts = p['canonical_counts']
            numerator = counts['block_zero_product_count']
            denominator = counts['model_product_count']
            assert type(numerator) is int and type(denominator) is int
            assert 0 <= numerator <= denominator and denominator > 0
            x, y = p['sparsity_percent'], p[latency_key]
            assert math.isclose(x, 100*numerator/denominator, rel_tol=0, abs_tol=1e-12)
            assert math.isfinite(y) and y > 0
            selected.append({
                'model': model, 'session': p.get('session', 'Run035'),
                'condition': p['condition'], 'source_family': p['family'],
                'family': family, 'kappa': p[kappa_key], 'qualified': True,
                'block_zero_product_count': numerator, 'model_product_count': denominator,
                'sparsity_percent': x, 'latency_ms': y,
            })
        assert len(selected) == 22
        assert {(p['family'], p['kappa']) for p in selected} == expected
        assert len({(p['session'], p['condition']) for p in selected}) == 22
        points.extend(selected)
    return sources, references, points


def main(log_y=False):
    sources, references, points = load_points()
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelsize': 11,
        'axes.linewidth': .65, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    connections = []
    for model in ['14M', '70M']:
        for family in RECIPES:
            ordered = sorted((p for p in points if p['model'] == model
                              and p['family'] == family), key=lambda p: p['kappa'])
            assert [p['kappa'] for p in ordered] == KAPPAS
            color = next(g[2] for g in GROUPS if family.startswith(g[0]))
            pressure = family.split('@')[1]
            ax.plot([p['sparsity_percent'] for p in ordered],
                    [p['latency_ms'] for p in ordered], color=color,
                    linestyle=(0, DASHES[pressure]), linewidth=1.15, alpha=.85, zorder=2)
            connections.append({'model': model, 'family': family,
                                'dash_pattern_points': DASHES[pressure],
                                'conditions': [p['condition'] for p in ordered],
                                'kappas': KAPPAS})
        for prefix, _, color, marker, size in GROUPS:
            selected = [p for p in points if p['model'] == model
                        and p['family'].split('+')[0] == prefix]
            assert len(selected) == (1 if prefix in ['A0', 'A1-H'] else 10)
            ax.scatter([p['sparsity_percent'] for p in selected],
                       [p['latency_ms'] for p in selected],
                       facecolors='white' if model == '14M' else color,
                       edgecolors=color, marker=marker, s=size,
                       linewidths=1.05 if model == '14M' else .7, zorder=3)
    ymin, ymax = min(p['latency_ms'] for p in points), max(p['latency_ms'] for p in points)
    padding = .06*(ymax-ymin)
    ax.set(xlim=(-.8, 42.5), ylim=(ymin-padding, ymax+padding),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel='Full-model latency (ms)')
    if log_y:
        factor = (ymax/ymin)**.06
        ax.set(yscale='log', ylim=(ymin/factor, ymax*factor),
               ylabel='Full-model latency (ms, log scale)')
        ax.set_yticks([.5, .75, 1, 1.5, 2, 3])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    assert ax.get_yscale() == ('log' if log_y else 'linear')
    ax.set_xticks(range(0, 41, 5))
    ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
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
        Line2D([], [], color='#444A52', linestyle=(0, DASHES['h']), label='OL1(h)'),
        Line2D([], [], color='#444A52', linestyle=(0, DASHES['all']), label='OL1(all)'),
    ]
    fig.legend(handles=topology_handles, loc='lower center', ncol=4,
               frameon=False, fontsize=9, handletextpad=.4, columnspacing=1.6,
               bbox_to_anchor=(.55, .054))
    fig.legend(handles=style_handles, loc='lower center', ncol=4,
               frameon=False, fontsize=9, handletextpad=.6, columnspacing=2.2,
               bbox_to_anchor=(.55, .005))
    fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.97)
    assert len(fig.axes) == 1 and len(ax.lines) == 8
    assert sum(len(c.get_offsets()) for c in ax.collections) == 44
    assert all(ax.get_xlim()[0] < p['sparsity_percent'] < ax.get_xlim()[1]
               and ax.get_ylim()[0] < p['latency_ms'] < ax.get_ylim()[1] for p in points)
    suffix = '-log-y' if log_y else ''
    output = HERE/f'figures/09-14m-70m-matched-sparsity-latency{suffix}.pdf'
    fig.savefig(output, metadata={
        'Title': 'Matched Pythia-14M and Pythia-70M sparsity versus final-kernel latency',
        'Creator': 'Analysis024 / 09_plot_matched_combined_latency.py',
        'CreationDate': None, 'ModDate': None,
    })
    record = {
        'sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in sources],
        'verified_against': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in references],
        'matching_key': ['family', 'kappa'], 'kappas': KAPPAS,
        'checkpoint_count': 44, 'checkpoint_count_per_model': 22,
        'display': 'Single panel; linear shared axes; open 14M and filled 70M markers; short-dashed OL1(h) and long-dashed OL1(all).',
        'x_limits_percent': list(ax.get_xlim()), 'y_limits_ms': list(ax.get_ylim()),
        'points': points, 'connections': connections,
        'comparison_limits': 'Matched recipe/kappa conditions and measurement definition, not identical implementations or physical GPU/host sessions. 14M uses K050; 70M uses the qualified k050-70m-v2 port. No equal optimization-budget or causal scaling claim.',
        'pdf_sha256': sha(output),
    }
    if log_y:
        linear_source = HERE/'data/matched-combined-latency.json'
        linear = json.loads(linear_source.read_text())
        assert record['points'] == linear['points']
        assert json.loads(json.dumps(connections)) == linear['connections']
        record['display'] = record['display'].replace('linear shared axes', 'linear x and logarithmic y shared axes')
        record['y_scale'] = 'log'
        record['y_ticks_ms'] = list(ax.get_yticks())
        record['linear_display_source'] = {'path': linear_source.relative_to(ROOT).as_posix(),
                                           'sha256': sha(linear_source)}
    plt.close(fig)
    (HERE/f'data/matched-combined-latency{suffix}.json').write_text(
        json.dumps(record, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('Single panel; 44 checkpoints; 22 matching recipe/kappa conditions per model; 8 separate curves.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-y', action='store_true', help='Save a separate logarithmic latency variant.')
    main(log_y=parser.parse_args().log_y)
