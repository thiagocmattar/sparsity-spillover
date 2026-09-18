"""Validation loss versus logical sparsity for the matched 14M/70M recipes."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 72),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 32),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 32),
    ('A7', '7-sites (A7)', '#C96024', '^', 42),
]
QUALITY_FAMILIES = {
    '14M': {'A4+OL1@h': 'A4-OL1-H', 'A4+OL1@all': 'A4-OL1',
            'A7+OL1@h': 'A7-OL1-H', 'A7+OL1@all': 'A7-OL1'},
    '70M': {'A0': 'Baseline (GeLU)', 'A1-H': 'ReLU',
            'A4+OL1@h': 'A4 + OL1(h)', 'A4+OL1@all': 'A4 + OL1(all)',
            'A7+OL1@h': 'A7 + OL1(h)', 'A7+OL1@all': 'A7 + OL1(all)'},
}
COVERAGE = {'sequences': 338, 'input_tokens': 692224, 'source_tokens': 693668,
            'excluded_tail_tokens': 1444, 'complete_block_coverage': True}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    cohort = read(HERE/'data/matched-combined-latency.json')
    q14 = read(ROOT/'analyses/023-2026-09-17-14m-pressure-targets-paper-table/results.json')
    q70 = read(ROOT/'analyses/025-2026-09-17-70m-quality-sparsity/data/70m-quality-sparsity.json')
    quality = {'14M': q14['rows'], '70M': q70['points']}
    quality_hashes = {'14M': q14['sources_sha256'], '70M': q70['source_sha256']}
    timing = {}
    for model, source in zip(['14M', '70M'], cohort['sources']):
        assert sha(ROOT/source['path']) == source['sha256']
        timing[model] = {p['condition']: p for p in read(ROOT/source['path'])['points']}
    points = []
    for point in cohort['points']:
        model, family = point['model'], point['family']
        timed = timing[model][point['condition']]
        assert timed['qualified'] and timed['family'] == point['source_family']
        weight = next(f for f in timed['checkpoint_files'] if f['path'].endswith('/model.safetensors'))
        attempt = Path(weight['path']).parents[2]
        metrics_path = ROOT/attempt/'metrics.json'
        logical_path = ROOT/attempt/'diagnostics/logical_products.json'
        metrics, logical = read(metrics_path), read(logical_path)
        final = metrics['validation']['final']
        for measured in [final, logical['coverage']]:
            assert all(measured[k] == v for k, v in COVERAGE.items())
        for key in ['block_zero_product_count', 'model_product_count']:
            assert type(point[key]) is int
            assert logical['measured'][key] == point[key] == timed['canonical_counts'][key]
        expected = 100*point['block_zero_product_count']/point['model_product_count']
        assert math.isclose(point['sparsity_percent'], expected, rel_tol=0, abs_tol=1e-12)
        loss = final['loss']
        assert math.isfinite(loss)
        if family in QUALITY_FAMILIES[model]:
            q = next(q for q in quality[model] if q['family'] == QUALITY_FAMILIES[model][family]
                     and q['kappa'] == point['kappa'])
            expected_attempt = q['source_attempt'] if model == '14M' else (
                f"runs/{q['run']}/artifacts/attempts/{q['attempt_id']}")
            assert attempt.as_posix() == expected_attempt
            expected_loss = q['ordinary_final_validation_loss'] if model == '14M' else q['final_validation_loss']
            assert loss == expected_loss
            for path in [metrics_path, logical_path]:
                assert sha(path) == quality_hashes[model][path.relative_to(ROOT).as_posix()]
            assert q['block_zero_product_count'] == point['block_zero_product_count']
            assert q['model_product_count'] == point['model_product_count']
        else:
            assert model == '14M' and family in ['A0', 'A1-H']
            assert metrics['condition']['activation'] == ('gelu' if family == 'A0' else 'relu')
            assert metrics['condition']['is_control'] and metrics['condition']['pressure_method'] == 'none'
        points.append({k: point[k] for k in ['model', 'condition', 'family', 'kappa',
                                           'sparsity_percent', 'block_zero_product_count',
                                           'model_product_count']} | {
            'validation_loss': loss, 'logical_pass_loss_for_audit_only': logical['coverage']['loss'],
            'source_attempt': attempt.as_posix(), 'checkpoint_weight': weight,
            'loss_source': metrics_path.relative_to(ROOT).as_posix(),
        })
    assert len(points) == 44
    keys = [{(p['family'], p['kappa']) for p in points if p['model'] == m} for m in ['14M', '70M']]
    assert keys[0] == keys[1] and len(keys[0]) == 22
    speedup = read(HERE/'data/matched-a0-normalized-speedup.json')
    assert {(p['model'], p['condition']) for p in points if p['family'] != 'A1-H'} == {
        (p['model'], p['condition']) for p in speedup['points']}
    assert cohort['connections'] == speedup['connections']
    return {'points': points, 'connections': cohort['connections'], 'kappas': cohort['kappas'],
            'sources_sha256': sources, 'coverage': {'validation_documents': 500, **COVERAGE},
            'loss_definition': 'Ordinary reloaded final-checkpoint metrics.validation.final.loss for every point.',
            'sparsity_definition': '100 * pooled block_zero_product_count / model_product_count, including the dense LM-head denominator.',
            'checkpoint_count': 44, 'checkpoint_count_per_model': 22}


def main():
    data = collect()
    points = data['points']
    by_id = {(p['model'], p['condition']): p for p in points}
    assert len(by_id) == 44
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.linewidth': .65,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for curve in data['connections']:
        ordered = [by_id[(curve['model'], c)] for c in curve['conditions']]
        assert [p['kappa'] for p in ordered] == data['kappas']
        assert all(p['family'] == curve['family'] for p in ordered)
        color = next(g[2] for g in GROUPS if curve['family'].startswith(g[0]))
        ax.plot([p['sparsity_percent'] for p in ordered], [p['validation_loss'] for p in ordered],
                color=color, linestyle=(0, curve['dash_pattern_points']), linewidth=1.15,
                alpha=.85, zorder=2)
    for model in ['14M', '70M']:
        for prefix, _, color, marker, size in GROUPS:
            selected = [p for p in points if p['model'] == model and p['family'].split('+')[0] == prefix]
            assert len(selected) == (1 if prefix in ['A0', 'A1-H'] else 10)
            ax.scatter([p['sparsity_percent'] for p in selected], [p['validation_loss'] for p in selected],
                       facecolors='white' if model == '14M' else color, edgecolors=color,
                       marker=marker, s=size, linewidths=1.05 if model == '14M' else .7, zorder=3)
    ymin, ymax = min(p['validation_loss'] for p in points), max(p['validation_loss'] for p in points)
    span = ymax-ymin
    ax.set(xlim=(-.8, 42.5), ylim=(ymin-.055*span, ymax+.12*span),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel='Validation loss (nats)')
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks([4, 4.5, 5, 5.5, 6])
    annotations = []
    for model, x in [('14M', 12), ('70M', 33)]:
        y = max(p['validation_loss'] for p in points if p['model'] == model) + .05*span
        ax.text(x, y, model, color='#555C64', fontsize=10, ha='center', va='bottom')
        annotations.append({'text': model, 'position': [x, y], 'coordinates': 'data'})
    ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)
    topology_handles = [Line2D([], [], color=c, marker=m, markersize=math.sqrt(s),
                              linestyle='none', label=label) for _, label, c, m, s in GROUPS]
    style_handles = [
        Line2D([], [], color='#444A52', marker='o', markersize=5.5,
               markerfacecolor='white', linestyle='none', label='14M'),
        Line2D([], [], color='#444A52', marker='o', markersize=5.5, linestyle='none', label='70M'),
        Line2D([], [], color='#444A52', linestyle=(0, (2, 2)), label='OL1(h)'),
        Line2D([], [], color='#444A52', linestyle=(0, (6, 3)), label='OL1(all)'),
    ]
    fig.legend(handles=topology_handles, loc='lower center', ncol=4, frameon=False, fontsize=9,
               handletextpad=.4, columnspacing=1.6, bbox_to_anchor=(.55, .054))
    fig.legend(handles=style_handles, loc='lower center', ncol=4, frameon=False, fontsize=9,
               handletextpad=.6, columnspacing=2.2, bbox_to_anchor=(.55, .005))
    fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.97)
    assert len(fig.axes) == 1 and len(ax.lines) == 8
    assert ax.get_xscale() == ax.get_yscale() == 'linear'
    assert sum(len(c.get_offsets()) for c in ax.collections) == 44
    assert all(ax.get_xlim()[0] < p['sparsity_percent'] < ax.get_xlim()[1]
               and ax.get_ylim()[0] < p['validation_loss'] < ax.get_ylim()[1] for p in points)
    output = HERE/'figures/11-14m-70m-matched-quality-sparsity.pdf'
    fig.savefig(output, metadata={'Title': 'Matched Pythia-14M and Pythia-70M quality versus model-wide sparsity',
                                 'Creator': 'Analysis024 / 11_plot_matched_quality_sparsity.py',
                                 'CreationDate': None, 'ModDate': None})
    data.update({'x_scale': 'linear', 'y_scale': 'linear', 'x_limits_percent': list(ax.get_xlim()),
                 'y_limits_loss': list(ax.get_ylim()), 'annotations': annotations,
                 'pdf_sha256': sha(output)})
    plt.close(fig)
    (HERE/'data/matched-quality-sparsity.json').write_text(
        json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(f"44 checkpoints; eight curves; uniform ordinary final validation; {len(data['sources_sha256'])} source hashes.")


if __name__ == '__main__':
    main()
