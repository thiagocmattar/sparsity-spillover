"""Join the exact 36-model panel-(a) cohort to Run041; plot retained measurements."""
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analyses/024-2026-09-17-h-only-kernel-latency'
NEW = ROOT / 'runs/041-2026-09-20-pythia14m-hz-h-only-ol1'
FIGURE = HERE / 'figures/01-14m-quality-sparsity-latency-with-hz.pdf'
COVERAGE = dict(sequences=338, input_tokens=692224, source_tokens=693668,
                excluded_tail_tokens=1444, complete_block_coverage=True)
STYLES = [
    ('0', 'none', r'$T_0/P_0$ (Base)', '#52565C', 'none'),
    ('1', 'none', r'$T_1/P_0$ (ReLU)', '#718052', 'none'),
    ('1', 'h', r'$T_1/P_1$', '#B9568B', '--'),
    ('hz', 'h', r'$T_{hz}/P_h$ (new)', '#173FAD', (0, (1.5, 1.3))),
    ('4', 'none', r'$T_4/P_0$', '#BA5256', '-.'),
    ('4', 'h', r'$T_4/P_h$', '#008B87', (0, (3, 2))),
    ('4', 'all', r'$T_4/P_{\mathrm{all}}$', '#3679AD', '-'),
    ('7', 'none', r'$T_7/P_0$', '#B19A39', '-.'),
    ('7', 'h', r'$T_7/P_h$', '#8A669C', (0, (3, 2))),
    ('7', 'all', r'$T_7/P_{\mathrm{all}}$', '#C27539', '-'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gm(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(statistics.mean(map(math.log, values)))


def collect():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    selection = read(OLD / 'data/quality-main-appendix.json')
    view = next(v for v in selection['outputs'] if v['name'] == '14-14m-70m-quality-sparsity.pdf')
    rows = [dict(r) for r in view['trained_points'] if r['model'] == '14M']
    clips = [p for p in view['clipping_points'] if p['model'] == '14M']
    assert len(rows) == 36 and len(clips) == 20
    assert all(r['pressure'] != 'L1' for r in rows)
    paper = read(OLD / 'data/paper-checkpoints.json')
    paper_by_key = {r['checkpoint_key']: r for r in paper['checkpoints']}
    for row in rows:
        p = paper_by_key[row['checkpoint_key']]
        for k in ['loss', 'sparsity', 'latency_ms', 'source_attempt', 'timing_session']:
            assert row[k] == p[k]
        row.update(timing_device_uuid=p['timing_device_uuid'],
                   timing_workload=p['timing_workload'], process_latency_ms=p['process_latency_ms'])
        assert p['kernel'] == 'K050' and p['timing_workload']['paired_samples_per_checkpoint'] == 1344
        assert math.isclose(gm(p['process_latency_ms']), p['latency_ms'], rel_tol=1e-12)

    training = read(NEW / 'artifacts/verification.json')
    latency = read(NEW / 'latency/artifacts/verification.json')
    summary = read(NEW / 'artifacts/summary.json')
    assert training['status'] == 'verified' and training['condition_count'] == 4
    assert latency['all_required_artifacts_verified'] and latency['qualified_processes'] == 12
    for source in summary['sources']:
        assert sha(NEW / source['path']) == source['sha256']
    hz_ceiling = None
    for index, t in enumerate(sorted(training['conditions'], key=lambda r: r['condition']['gate_threshold'])):
        condition = t['condition']
        assert condition['active_sites'] == ['h', 'z']
        assert condition['pressure_sites'] == ['h'] and condition['pressure_weight'] == 1
        assert condition['pressure_method'] == 'orthogonal_l1'
        attempt = NEW / 'artifacts/attempts' / t['attempt_id']
        metrics = read(attempt / 'metrics.json')
        logical = read(attempt / 'diagnostics/logical_products.json')
        manifest = read(attempt / 'manifest.json')
        times, per_process = [], []
        for replicate in range(1, 4):
            folder = NEW / 'latency/artifacts/attempts' / f'scientific-c{index:02d}-r{replicate}-001'
            result = read(folder / 'result.json')
            timing = read(folder / 'timing.json')
            assert result['status'] == 'complete' and result['qualified']
            assert result['validation_blocks'] == 338
            assert result['checkpoint']['final_checkpoint_content_sha256'] == t['checkpoint_content_sha256']
            samples = [p['host_ms'] for p in timing['samples'] if p['mode'] == 'candidate_graph']
            assert len(samples) == 448
            times.extend(samples)
            per_process.append(gm(samples))
        device_ids = {p['gpu_uuid'] for p in latency['scientific_processes']}
        assert len(device_ids) == 1 and len(times) == 1344
        counts = logical['measured']
        s = next(p for p in summary['conditions'] if p['kappa'] == condition['gate_threshold'])
        assert metrics['validation']['final']['loss'] == s['training_validation_loss']
        assert math.isclose(gm(times), s['k050_geomean_host_ms'], rel_tol=1e-12)
        assert counts['R_model'] == s['R_model']
        rows.append(dict(model='14M', family='HZ-OL1-h', scope='hz', pressure='h',
                         kappa=condition['gate_threshold'], local_pressure_weight=None,
                         source_attempt=attempt.relative_to(ROOT).as_posix(),
                         checkpoint_key=t['checkpoint_content_sha256'],
                         final_checkpoint_content_sha256=t['checkpoint_content_sha256'],
                         loss=metrics['validation']['final']['loss'],
                         sparsity=100 * counts['block_zero_product_count'] / counts['model_product_count'],
                         zero_product_count=counts['block_zero_product_count'],
                         model_product_count=counts['model_product_count'], coverage=COVERAGE,
                         initial_parameter_sha256=manifest['initial_parameter_sha256'],
                         training_schedule_hash=manifest['training_schedule_hash'],
                         training_steps=712, training_tokens=1493172224,
                         latency_ms=gm(times), process_latency_ms=per_process,
                         timing_session='Run041', timing_device_uuid=next(iter(device_ids)),
                         timing_workload=paper_by_key[rows[0]['checkpoint_key']]['timing_workload'],
                         kernel='K050', qualified=True))
        if hz_ceiling is not None:
            assert hz_ceiling == logical['architecture_maximum']
        hz_ceiling = logical['architecture_maximum']

    assert len(rows) == len({r['source_attempt'] for r in rows}) == 40
    assert len({r['initial_parameter_sha256'] for r in rows}) == 1
    assert len({r['training_schedule_hash'] for r in rows}) == 1
    for row in rows:
        attempt = ROOT / row['source_attempt']
        metrics = read(attempt / 'metrics.json')
        logical = read(attempt / 'diagnostics/logical_products.json')
        manifest = read(attempt / 'manifest.json')
        counts = logical['measured']
        assert sum(p['zero_product_count'] for p in counts['per_operation'].values()) == counts['block_zero_product_count']
        assert counts['model_product_count'] == counts['block_product_count'] + counts['lm_head_product_count']
        assert row['loss'] == metrics['validation']['final']['loss']
        assert row['zero_product_count'] == logical['measured']['block_zero_product_count']
        assert row['model_product_count'] == logical['measured']['model_product_count']
        assert row['final_checkpoint_content_sha256'] == manifest['checkpoints']['final']['content_sha256']
        assert row['initial_parameter_sha256'] == manifest['initial_parameter_sha256']
        assert row['training_schedule_hash'] == manifest['training_schedule_hash']
        assert manifest['seeds'] == {'model': 1234, 'data_order': 1234}
        for coverage in [metrics['validation']['final'], logical['coverage']]:
            assert all(coverage[k] == v for k, v in COVERAGE.items())
        assert row['training_steps'] == metrics['training']['completed_steps'] == 712
        assert row['training_tokens'] == metrics['training']['input_tokens'] == 1493172224
        assert math.isclose(row['sparsity'], 100 * row['zero_product_count'] / row['model_product_count'], rel_tol=1e-12)
    for point in clips:
        assert point['coverage'] == COVERAGE
        assert math.isclose(point['sparsity'], 100 * point['zero_product_count'] / point['model_product_count'], rel_tol=1e-12)
    references = {}
    for name in ['06-14m-quality-sparsity-latency.pdf', '14-14m-70m-quality-sparsity.pdf']:
        path = ROOT / 'manuscript/draft/figures' / name
        references[path.relative_to(ROOT).as_posix()] = sha(path)
        assert sha(path) == sha(OLD / 'figures' / name)
    return dict(trained_points=rows, clipping_points=clips,
                ceilings=dict(view['ceilings']['14M'], hz=hz_ceiling),
                sources_sha256=sources, reference_pdfs_sha256=references,
                coverage=COVERAGE, excluded_naive_l1=selection['excluded_trained_points'],
                loss_convention='Ordinary final-checkpoint FP16 validation; retained FP16 pass for clipping.',
                latency_convention='Geometric mean of 1344 K050 host timings; RTX5090 BF16 B1 T2048 full logits.',
                session_caveat='Run029 (31), Run033 (5), and Run041 (4) used different physical GPU/host sessions; small latency differences are not controlled rankings.')


def draw(data):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.titlesize': 11.5, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9), sharex=True)
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.32, wspace=.28)
    fig.suptitle('Quality, sparsity and full-model latency on Pythia-14M', fontsize=15, y=.98)
    fig.text(.53, .912, '40 trained models | Latency: RTX 5090, BF16, B=1, T=2048 | three timing sessions',
             ha='center', fontsize=9.2, color='#646970')
    handles, series = [], []
    for scope, pressure, label, color, ls in STYLES:
        group = [r for r in data['trained_points'] if (r['scope'], r['pressure']) == (scope, pressure)]
        dose = 'local_pressure_weight' if scope == '1' and pressure == 'h' else 'kappa'
        group.sort(key=lambda r: -1 if r[dose] is None else r[dose])
        expected = [None] if pressure == 'none' and scope in ['0', '1'] else (
            [.05, .1, .5, 1] if scope == '1' else [0, .01, .05, .1] if scope == 'hz' else [0, .01, .05, .1, .5])
        assert [r[dose] for r in group] == expected
        control = len(group) == 1
        style = dict(color=color, ls=ls, lw=1.8 if scope == 'hz' else 1.5,
                     marker='o', ms=8 if control else 5.8 if scope == 'hz' else 4.9,
                     mfc='white' if scope == '0' else color,
                     mec=color if scope == '0' else 'white', mew=1.3 if scope == '0' else .6)
        for ax, metric in zip(axes, ['loss', 'latency_ms']):
            ax.plot([r['sparsity'] for r in group], [r[metric] for r in group],
                    **style, zorder=7 if scope == 'hz' else 5 if control else 4)
        handles.append(Line2D([], [], **style, label=label))
        series.append(dict(scope=scope, pressure=pressure, label=label,
                           source_attempts=[r['source_attempt'] for r in group], dose_field=dose))
    for scope, color in [('0', '#52565C'), ('1', '#718052')]:
        group = sorted([p for p in data['clipping_points'] if p['scope'] == scope], key=lambda p: p['target'])
        assert [p['target'] for p in group] == [i / 10 for i in range(10)]
        axes[0].plot([p['sparsity'] for p in group], [p['loss'] for p in group], color=color, ls=':', lw=1.25, zorder=2)
    axes[0].text(2.3, 5.63, 'Post-hoc', rotation=76, rotation_mode='anchor',
                 color='#6C7177', fontsize=9.5, ha='left', va='bottom')
    for scope, label in [('hz', r'$T_{hz}$'), ('4', r'$T_4$'), ('7', r'$T_7$')]:
        ceiling = data['ceilings'][scope]['R_model_max_percent']
        axes[0].axvline(ceiling, color='#92969B', lw=.7, ls=(0, (2, 3)), alpha=.75, zorder=1)
        axes[0].text(ceiling + .4 if scope != '7' else ceiling - .3, .025 if scope == 'hz' else .98,
                     label + ' ceiling', transform=axes[0].get_xaxis_transform(),
                     ha='left' if scope != '7' else 'right', va='bottom' if scope == 'hz' else 'top',
                     fontsize=9, color='#6C7177')
    limits = {'loss': [5.0, 6.2], 'latency_ms': [.438, .677]}
    for ax, metric, title in zip(axes, limits, ['(a) Quality-sparsity trade-off', '(b) Full-model latency']):
        ax.set(xlim=(-.8, 31), ylim=limits[metric], xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
        ax.set_title(title, loc='left', pad=11)
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(length=3, width=.65)
        ax.grid(axis='y', color='#E8EAED', lw=.6)
        ax.set_axisbelow(True)
        assert all(limits[metric][0] <= r[metric] <= limits[metric][1] for r in data['trained_points'])
    axes[0].set_ylabel('Validation loss')
    axes[0].set_yticks([5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2])
    axes[1].set_ylabel('Full-model latency (ms)')
    # Three columns group one-/two-site, four-site, and seven-site recipes.
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=10.8, handlelength=2.3, columnspacing=2.5, labelspacing=.6,
               bbox_to_anchor=(.53, .015))
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE, metadata={'Title': 'Pythia-14M quality, sparsity and K050 latency including h/z gates',
                                  'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return dict(series=series, limits=limits, trained_count_per_panel=40,
                clipping_count_panel_a=20, clipping_count_panel_b=0,
                clipping_outside_loss_view=[p['id'] for p in data['clipping_points'] if not 5 <= p['loss'] <= 6.2],
                clipping_latency_note='Only trained-model latency is drawn, matching reference Figure06; retained clipping latency exists in Run036.',
                figure=FIGURE.relative_to(HERE).as_posix(), figure_sha256=sha(FIGURE),
                script_sha256=sha(Path(__file__)), figsize_inches=[10.4, 4.9])


def main():
    data = collect()
    data.update(draw(data))
    for path, expected in data['reference_pdfs_sha256'].items():
        assert sha(ROOT / path) == expected
    (HERE / 'data').mkdir(exist_ok=True)
    (HERE / 'data/figure-data.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'trained_per_panel': 40, 'new_hz_points': 4, 'clipping_quality_points': 20,
                      'clipping_above_view': len(data['clipping_outside_loss_view']),
                      'sources_hashed': len(data['sources_sha256']), 'figure': str(FIGURE)}))


if __name__ == '__main__':
    main()
