"""All executed paper recipes, with uniform final loss and pooled logical counts."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / 'figures/11-all-model-quality-sparsity.pdf'
EVIDENCE = HERE / 'data/all-model-quality-sparsity.json'
TITLE = 'Quality-sparsity trade-offs across model sizes'
KAPPAS = [0, .01, .05, .1, .5]
SIZES = ['14M', '70M', '410M']
LIMITS = {'14M': [-.8, 31], '70M': [-1.3, 51], '410M': [-2.3, 91]}
LOSS_LIMITS = [4.0, 6.20]
STYLES = [
    ('0', 'none', r'$T_0/P_0$ (Base)', '#52565C', 'none'),
    ('1', 'none', r'$T_1/P_0$ (ReLU)', '#718052', 'none'),
    ('1', 'L1', r'$T_1/P_1$ (L1)', '#95664E', '-.'),
    ('1', 'h', r'$T_1/P_1$ (OL1)', '#B9568B', '--'),
    ('4', 'none', r'$T_4/P_0$', '#BA5256', '-.'),
    ('4', 'h', r'$T_4/P_h$', '#008B87', (0, (3, 2))),
    ('4', 'all', r'$T_4/P_{\mathrm{all}}$', '#3679AD', '-'),
    ('7', 'none', r'$T_7/P_0$', '#B19A39', '-.'),
    ('7', 'h', r'$T_7/P_h$', '#8A669C', (0, (3, 2))),
    ('7', 'all', r'$T_7/P_{\mathrm{all}}$', '#C27539', '-'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    paper = read(HERE / 'data/paper-checkpoints.json')
    historical = read(ROOT / 'analyses/018-2026-09-08-results-materials/figure_data.json')
    clipping = read(ROOT / 'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json')
    inputs = [{k: r[k] for k in ['model', 'family', 'scope', 'pressure', 'kappa',
              'local_pressure_weight', 'source_attempt', 'loss', 'counts', 'checkpoint_key',
              'latency_ms', 'timing_session', 'kernel']} for r in paper['checkpoints']]
    for r in historical['trained']:
        if r['scale'] != '410M':
            continue
        scope = {'A0': '0', 'A1-H': '1', 'A4-OL1': '4', 'A7-OL1': '7'}[r['family']]
        inputs.append({'model': '410M', 'family': r['family'], 'scope': scope,
                       'pressure': 'all' if scope in ['4', '7'] else 'none',
                       'kappa': r['dose'], 'local_pressure_weight': None,
                       'source_attempt': r['source'], 'counts': r['counts']})
    rows = []
    coverage = {'sequences': 338, 'input_tokens': 692224, 'source_tokens': 693668,
                'excluded_tail_tokens': 1444, 'complete_block_coverage': True}
    for r in inputs:
        attempt = ROOT / r['source_attempt']
        metrics = read(attempt / 'metrics.json')
        logical = read(attempt / 'diagnostics/logical_products.json')
        manifest = read(attempt / 'manifest.json')
        final = metrics['validation']['final']
        for c in [final, logical['coverage']]:
            assert all(c[k] == v for k, v in coverage.items())
        assert r['counts'] == logical['measured']
        counts = r.pop('counts')
        assert sum(p['zero_product_count'] for p in counts['per_operation'].values()) == counts['block_zero_product_count']
        assert counts['model_product_count'] == counts['block_product_count'] + counts['lm_head_product_count']
        assert metrics['training']['completed_steps'] == 712
        assert metrics['training']['optimizer_step_count'] == 712
        assert metrics['training']['input_tokens'] == 1493172224
        assert manifest['seeds'] == {'data_order': 1234, 'model': 1234}
        if 'loss' in r:
            assert r['loss'] == final['loss']
        r.update(loss=final['loss'], sparsity=100 * counts['block_zero_product_count'] / counts['model_product_count'],
                 zero_product_count=counts['block_zero_product_count'], model_product_count=counts['model_product_count'],
                 coverage=coverage, final_checkpoint_content_sha256=manifest['checkpoints']['final']['content_sha256'],
                 initial_parameter_sha256=manifest['initial_parameter_sha256'],
                 training_schedule_hash=manifest['training_schedule_hash'],
                 training_steps=712, training_tokens=1493172224,
                 final_learning_rate=metrics['training']['learning_rate_final'],
                 logical_pass_loss=logical['coverage']['loss'])
        rows.append(r)
    assert len(rows) == len({r['source_attempt'] for r in rows}) == 74
    for size, count in zip(SIZES, [40, 22, 12]):
        group = [r for r in rows if r['model'] == size]
        assert len(group) == count
        assert len({r['initial_parameter_sha256'] for r in group}) == 1
        assert len({r['training_schedule_hash'] for r in group}) == 1
    clips = []
    for p in clipping['points']:
        if p['family'] not in ['A0', 'A1-H']:
            continue
        assert all(p['coverage'][k] == v for k, v in coverage.items())
        r = next(r for r in rows if r['model'] == p['scale'] and r['family'] == p['family'])
        assert r['final_checkpoint_content_sha256'] == p['checkpoint_content_sha256']
        counts = p['counts']
        clips.append({'id': p['id'], 'model': p['scale'], 'scope': r['scope'], 'target': p['dose'],
                      'loss': p['loss'], 'sparsity': 100 * counts['block_zero_product_count'] / counts['model_product_count'],
                      'zero_product_count': counts['block_zero_product_count'],
                      'model_product_count': counts['model_product_count'],
                      'checkpoint_content_sha256': p['checkpoint_content_sha256'], 'coverage': coverage})
    assert len(clips) == 60
    ceilings = dict(paper['coverage'])
    ceilings['410M'] = {s: next(r['ceiling'] for r in historical['trained']
                              if r['scale'] == '410M' and r['family'] == f'A{s}-OL1') for s in ['4', '7']}
    for size in SIZES:
        for scope in ['0', '1']:
            assert sorted(p['target'] for p in clips if p['model'] == size and p['scope'] == scope) == [p / 10 for p in range(10)]
    baselines = {s: next(r for r in rows if r['model'] == s and r['scope'] == '0') for s in SIZES}
    extremes = []
    for size in SIZES:
        r = next(r for r in rows if r['model'] == size and r['scope'] == '7' and r['pressure'] == 'all' and r['kappa'] == .5)
        extremes.append({'model': size, 'sparsity': r['sparsity'], 'loss': r['loss'],
                         'base_loss': baselines[size]['loss'], 'loss_increase': r['loss'] - baselines[size]['loss'],
                         'fraction_of_ceiling': r['sparsity'] / ceilings[size]['7']['R_model_max_percent']})
    speedups = [{'model': r['model'], 'scope': r['scope'], 'pressure': r['pressure'],
                 'sparsity': r['sparsity'], 'base_latency_ms': baselines[r['model']]['latency_ms'],
                 'latency_ms': r['latency_ms'],
                 'speedup_vs_optimized_base': baselines[r['model']]['latency_ms'] / r['latency_ms'],
                 'timing_session': r['timing_session'], 'base_session': baselines[r['model']]['timing_session'],
                 'kernel': r['kernel']} for r in rows if r['model'] in ['14M', '70M'] and r['scope'] in ['4', '7']
                and r['pressure'] in ['h', 'all'] and r['kappa'] == .5]
    return {'trained_points': rows, 'clipping_points': clips, 'ceilings': ceilings,
            'sources_sha256': sources, 'high_threshold_T7_all': extremes, 'high_threshold_speedups': speedups,
            'loss_convention': 'Ordinary reloaded final-checkpoint validation for all 74 trained models; retained FP16 clipping loss.',
            'coverage': coverage, 'training_steps': 712, 'training_tokens': 1493172224,
            'cross_size_protocol': historical['exposure']}


def main():
    preserved = {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data = build()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5,
                        'axes.titlesize': 12.5, 'axes.labelsize': 12.5,
                        'xtick.labelsize': 11, 'ytick.labelsize': 11,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.55), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.34, wspace=.20)
    fig.suptitle(TITLE, fontsize=15, y=.98)
    series, panels = [], []
    for ax, size, letter in zip(axes, SIZES, 'abc'):
        rows = [r for r in data['trained_points'] if r['model'] == size]
        for scope, pressure, label, color, ls in STYLES:
            group = [r for r in rows if (r['scope'], r['pressure']) == (scope, pressure)]
            if not group:
                continue
            dose = 'local_pressure_weight' if scope == '1' and pressure != 'none' else 'kappa'
            group.sort(key=lambda r: -1 if r[dose] is None else r[dose])
            if scope in ['4', '7']:
                assert [r['kappa'] for r in group] == KAPPAS
            elif pressure != 'none':
                assert [r[dose] for r in group] == [.05, .1, .5, 1]
            else:
                assert len(group) == 1
            control = scope in ['0', '1'] and pressure == 'none'
            ax.plot([r['sparsity'] for r in group], [r['loss'] for r in group],
                    color=color, ls=ls, marker='o', lw=1.55, ms=7.5 if control else 4.8,
                    mfc='white' if scope == '0' else color,
                    mec=color if scope == '0' else 'white', mew=1.25 if scope == '0' else .55,
                    zorder=5 if control else 4)
            series.append({'model': size, 'scope': scope, 'pressure': pressure, 'label': label,
                           'color': color, 'linestyle': ls, 'dose_field': dose,
                           'source_attempts': [r['source_attempt'] for r in group]})
        clips = [p for p in data['clipping_points'] if p['model'] == size]
        for scope, color in [('0', STYLES[0][3]), ('1', STYLES[1][3])]:
            group = sorted([p for p in clips if p['scope'] == scope], key=lambda p: p['target'])
            ax.plot([p['sparsity'] for p in group], [p['loss'] for p in group],
                    color=color, ls=':', lw=1.25, zorder=2)
        for scope in ['4', '7']:
            ceiling = data['ceilings'][size][scope]['R_model_max_percent']
            ax.axvline(ceiling, color='#92969B', lw=.75, ls=(0, (2, 3)), alpha=.8, zorder=1)
            right_of_line = size == '14M' and scope == '4'
            ax.text(ceiling + .5 if right_of_line else ceiling - .4,
                    .87 if size == '410M' and scope == '7' else .98,
                    rf'$T_{scope}$ ceiling', transform=ax.get_xaxis_transform(),
                    ha='left' if right_of_line else 'right', va='top', fontsize=9.5, color='#6C7177')
        x, y, rotation = {'14M': (3.4, 5.54, 76), '70M': (14.8, 4.80, 63),
                          '410M': (34.0, 5.03, 23)}[size]
        ax.text(x, y, 'Post-hoc', rotation=rotation, color='#6C7177', fontsize=10,
                rotation_mode='anchor', ha='left', va='bottom')
        ax.set(xlim=LIMITS[size], ylim=LOSS_LIMITS, xlabel=r'$S_{\mathrm{model}}$ (%)')
        ax.set_title(f'({letter}) Pythia-{size}', loc='left', pad=10)
        ax.set_xticks({'14M': [0, 10, 20, 30], '70M': [0, 20, 40], '410M': [0, 30, 60, 90]}[size])
        ax.set_yticks([4.0, 4.5, 5.0, 5.5, 6.0])
        ax.tick_params(length=3, width=.65, labelleft=True)
        ax.grid(axis='y', color='#E8EAED', lw=.6, zorder=0)
        assert all(LOSS_LIMITS[0] <= r['loss'] <= LOSS_LIMITS[1] for r in rows)
        panels.append({'model': size, 'trained_count': len(rows), 'clipping_count': len(clips),
                       'clipping_outside_y': [p['id'] for p in clips if not LOSS_LIMITS[0] <= p['loss'] <= LOSS_LIMITS[1]]})
    axes[0].set_ylabel('Validation loss')
    handles = [Line2D([], [], color=c, ls=ls, marker='o', ms=6,
                      mfc='white' if s == '0' else c, mec=c if s == '0' else 'white',
                      mew=1 if s == '0' else .5, lw=1.55, label=label)
               for s, p, label, c, ls in STYLES]
    # Fill the shared legend across rows, preserving the recipe ordering.
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
    fig.legend(handles=[handles[i] for i in order], loc='lower center', ncol=5,
               frameon=False, fontsize=11, handlelength=2.0, columnspacing=1.35,
               labelspacing=.9, bbox_to_anchor=(.52, .035))
    fig.savefig(OUTPUT, metadata={'Title': TITLE, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert preserved == {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data.update(title=TITLE, output=OUTPUT.relative_to(HERE).as_posix(), output_sha256=sha(OUTPUT),
                script=Path(__file__).name, script_sha256=sha(Path(__file__)), series=series, panels=panels,
                layout={'rows': 1, 'columns': 3, 'width_inches': 10.8, 'height_inches': 4.55},
                x_limits=LIMITS, shared_y_limits=LOSS_LIMITS, preserved_pdf_sha256=preserved)
    EVIDENCE.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(OUTPUT), 'panels': panels, 'extremes': data['high_threshold_T7_all']}))


if __name__ == '__main__':
    main()
