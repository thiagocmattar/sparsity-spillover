"""Add four complete A0/A1-H post-hoc clipping sweeps to Figure11."""
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
RUN = ROOT/'runs/030-2026-09-08-all-models-posthoc-clipping'
GROUPS = [
    ('A0', 'Baseline (A0)', '#444A52', '*', 72),
    ('A1-H', '1-site (A1-H)', '#19856B', 'o', 32),
    ('A4', '4-sites (A4*)', '#2878B5', 'D', 32),
    ('A7', '7-sites (A7)', '#C96024', '^', 42),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    trained = read(HERE/'data/matched-quality-sparsity.json')
    for path, digest in trained['sources_sha256'].items():
        assert sha(ROOT/path) == digest
    assert sha(HERE/'figures/11-14m-70m-matched-quality-sparsity.pdf') == trained['pdf_sha256']
    clipping = read(RUN/'results/clipping-points.json')
    verified = read(RUN/'results/verification.json')
    inputs = read(RUN/'input-manifest.json')
    assert verified['status'] == clipping['status'] == 'complete_verified'
    assert verified['all_targets_and_checkpoint_hashes_verified']
    assert sha(RUN/'results/clipping-points.json') == verified['output_hashes']['results/clipping-points.json']
    assert clipping['protocol']['clipping_sites'] == ['a', 'm', 'h', 'z']
    targets = [i/10 for i in range(10)]
    assert clipping['protocol']['targets'] == targets
    selected = [p for p in clipping['points'] if p['scale'] in ['14M', '70M']
                and p['family'] in ['A0', 'A1-H']]
    assert len(selected) == 40 and len({p['id'] for p in selected}) == 40
    raw = {}
    for path in sorted({p['source'] for p in selected}):
        assert sha(ROOT/path) == clipping['sources'][path]
        record = read(ROOT/path)
        for p in record.get('conditions', record.get('points', [])):
            raw[p['checkpoint_content_sha256'], p['target_sparsity']] = p
    clipped, curves, audits = [], [], []
    for model in ['14M', '70M']:
        for family in ['A0', 'A1-H']:
            group = sorted((p for p in selected if p['scale'] == model and p['family'] == family),
                           key=lambda p: p['dose'])
            assert [p['dose'] for p in group] == targets
            control = next(p for p in trained['points'] if p['model'] == model and p['family'] == family)
            identity = next(p for p in inputs['checkpoints'] if p['id'] == group[0]['source_checkpoint_id'])
            assert identity['source'] == control['source_attempt']
            weight = next(f for f in identity['files'] if f['name'] == 'model.safetensors')
            assert weight['sha256'] == control['checkpoint_weight']['sha256']
            assert weight['bytes'] == control['checkpoint_weight']['bytes']
            assert identity['checkpoint']+'/model.safetensors' == control['checkpoint_weight']['path']
            for p in group:
                assert p['checkpoint_content_sha256'] == identity['checkpoint_content_sha256']
                r = raw[p['checkpoint_content_sha256'], p['dose']]
                assert r['validation'] == p['coverage'] and r['validation']['loss'] == p['loss']
                assert r['logical_products'] == p['counts']
                assert r['thresholds_by_site_layer'] == p['thresholds_by_site_layer']
                assert {k.split('.layer_')[0] for k in p['thresholds_by_site_layer']} == {'a', 'm', 'h', 'z'}
                assert len(p['thresholds_by_site_layer']) == 24
                assert all(p['coverage'][k] == trained['coverage'][k] for k in
                           ['sequences', 'input_tokens', 'source_tokens', 'excluded_tail_tokens', 'complete_block_coverage'])
                counts = p['counts']
                assert sum(v['zero_product_count'] for v in counts['per_operation'].values()) == counts['block_zero_product_count']
                assert counts['model_product_count'] == counts['block_product_count']+counts['lm_head_product_count']
                assert p['R_model'] == counts['block_zero_product_count']/counts['model_product_count']
                assert math.isfinite(p['loss'])
                assert p['delta_loss_from_p0'] == p['loss']-group[0]['loss']
                clipped.append({
                    'id': p['id'], 'model': model, 'family': family, 'target_p': p['dose'],
                    'sparsity_percent': 100*counts['block_zero_product_count']/counts['model_product_count'],
                    'validation_loss': p['loss'], 'delta_loss_from_p0': p['delta_loss_from_p0'],
                    'block_zero_product_count': counts['block_zero_product_count'],
                    'model_product_count': counts['model_product_count'],
                    'source_checkpoint_id': p['source_checkpoint_id'],
                    'checkpoint_content_sha256': p['checkpoint_content_sha256'],
                    'source': p['source'], 'nondominated_within_checkpoint': p['nondominated_within_checkpoint'],
                })
            assert all(p['nondominated_within_checkpoint'] for p in group)
            zero = group[0]
            difference = zero['loss']-control['validation_loss']
            assert abs(difference) <= 5e-4
            audits.append({'model': model, 'family': family, 'trained_loss': control['validation_loss'],
                           'clipping_p0_loss': zero['loss'], 'loss_difference': difference,
                           'sparsity_difference_pp': 100*zero['R_model']-control['sparsity_percent']})
            curves.append({'model': model, 'family': family, 'point_ids': [p['id'] for p in group],
                           'targets': targets})
    return {'trained_points': trained['points'], 'trained_connections': trained['connections'],
            'clipping_points': clipped, 'clipping_connections': curves, 'p0_audit': audits,
            'sources_sha256': sources, 'coverage': trained['coverage'],
            'clipping_protocol': {'targets': targets, 'sites': ['a', 'm', 'h', 'z'],
                                  'calibration': 'Per-site/layer absolute-value order-statistic thresholds on the first ten complete source-order training blocks.',
                                  'rule': 'Evaluation-only abs(x) <= threshold, after existing trained gates.',
                                  'evaluation': 'FP32 parameters, FP16 CUDA autocast, eager uncached attention, batch one, sequence length 2048.'},
            'loss_definition': 'Trained points retain ordinary final-checkpoint loss; clipping points retain measured FP16 eager batch-one sweep loss, including each independent p=0. No offset adjustment.',
            'trained_count': 44, 'clipping_count': 40, 'checkpoint_count': 44,
            'scope': 'Figure11 unchanged plus A0/A1-H evaluation-only clipping at both sizes; all ten targets and the full observed loss range.'}


def main():
    data = collect()
    trained, clipped = data['trained_points'], data['clipping_points']
    by_id = {(p['model'], p['condition']): p for p in trained}
    clip_by_id = {p['id']: p for p in clipped}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 11, 'axes.linewidth': .65,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    for curve in data['clipping_connections']:
        ordered = [clip_by_id[i] for i in curve['point_ids']]
        _, _, color, marker, _ = next(g for g in GROUPS if g[0] == curve['family'])
        ax.plot([p['sparsity_percent'] for p in ordered], [p['validation_loss'] for p in ordered],
                color=color, linestyle=':', linewidth=1.05, alpha=.8, marker=marker,
                markersize=4.5 if marker == '*' else 3.2, markeredgewidth=.65,
                markerfacecolor='white' if curve['model'] == '14M' else color, zorder=2)
    for curve in data['trained_connections']:
        ordered = [by_id[curve['model'], c] for c in curve['conditions']]
        color = next(g[2] for g in GROUPS if curve['family'].startswith(g[0]))
        ax.plot([p['sparsity_percent'] for p in ordered], [p['validation_loss'] for p in ordered],
                color=color, linestyle=(0, curve['dash_pattern_points']), linewidth=1.15, alpha=.85, zorder=2)
    for model in ['14M', '70M']:
        for prefix, _, color, marker, size in GROUPS:
            selected = [p for p in trained if p['model'] == model and p['family'].split('+')[0] == prefix]
            ax.scatter([p['sparsity_percent'] for p in selected], [p['validation_loss'] for p in selected],
                       facecolors='white' if model == '14M' else color, edgecolors=color,
                       marker=marker, s=size, linewidths=1.05 if model == '14M' else .7, zorder=3)
    all_points = trained+clipped
    ymin, ymax = min(p['validation_loss'] for p in all_points), max(p['validation_loss'] for p in all_points)
    span = ymax-ymin
    ax.set(xlim=(-.8, 42.5), ylim=(ymin-.035*span, ymax+.10*span),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)', ylabel='Validation loss (nats)')
    ax.set_xticks(range(0, 41, 5))
    ax.set_yticks([4, 5, 6, 7, 8, 9])
    annotations = []
    for model, x in [('14M', 12), ('70M', 33)]:
        y = max(p['validation_loss'] for p in all_points if p['model'] == model) + .035*span
        ax.text(x, y, model, color='#555C64', fontsize=10, ha='center', va='bottom')
        annotations.append({'text': model, 'position': [x, y], 'coordinates': 'data'})
    ax.grid(axis='y', color='#E7E9ED', linewidth=.6)
    ax.set_axisbelow(True)
    ax.tick_params(direction='out', length=3, width=.6)
    topology_handles = [Line2D([], [], color=c, marker=m, markersize=math.sqrt(s), linestyle='none', label=label)
                        for _, label, c, m, s in GROUPS]
    style_handles = [
        Line2D([], [], color='#444A52', marker='o', markersize=5.5, markerfacecolor='white', linestyle='none', label='14M'),
        Line2D([], [], color='#444A52', marker='o', markersize=5.5, linestyle='none', label='70M'),
        Line2D([], [], color='#444A52', linestyle=(0, (2, 2)), label='OL1(h)'),
        Line2D([], [], color='#444A52', linestyle=(0, (6, 3)), label='OL1(all)'),
        Line2D([], [], color='#444A52', linestyle=':', label='Post-hoc clipping'),
    ]
    fig.legend(handles=topology_handles, loc='lower center', ncol=4, frameon=False, fontsize=9,
               handletextpad=.4, columnspacing=1.6, bbox_to_anchor=(.55, .054))
    fig.legend(handles=style_handles, loc='lower center', ncol=5, frameon=False, fontsize=9,
               handletextpad=.5, columnspacing=1.5, bbox_to_anchor=(.55, .005))
    fig.subplots_adjust(left=.10, right=.985, bottom=.25, top=.97)
    assert len(fig.axes) == 1 and len(ax.lines) == 12
    assert ax.get_xscale() == ax.get_yscale() == 'linear'
    assert sum(len(c.get_offsets()) for c in ax.collections) == 44
    assert sum(len(line.get_xdata()) for line in ax.lines[:4]) == 40
    assert all(ax.get_xlim()[0] < p['sparsity_percent'] < ax.get_xlim()[1]
               and ax.get_ylim()[0] < p['validation_loss'] < ax.get_ylim()[1] for p in all_points)
    output = HERE/'figures/12-14m-70m-quality-sparsity-clipping.pdf'
    fig.savefig(output, metadata={'Title': 'Matched 14M/70M quality-sparsity with baseline and one-site clipping',
                                 'Creator': 'Analysis024 / 12_plot_quality_sparsity_clipping.py',
                                 'CreationDate': None, 'ModDate': None})
    data.update({'x_scale': 'linear', 'y_scale': 'linear', 'x_limits_percent': list(ax.get_xlim()),
                 'y_limits_loss': list(ax.get_ylim()), 'annotations': annotations, 'pdf_sha256': sha(output)})
    plt.close(fig)
    (HERE/'data/quality-sparsity-clipping.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('44 unchanged trained points; 40 clipping measurements; four complete sweeps; full-range linear axes.')
    print('Maximum absolute clipping p=0 versus ordinary control loss difference:', max(abs(p['loss_difference']) for p in data['p0_audit']))


if __name__ == '__main__':
    main()
