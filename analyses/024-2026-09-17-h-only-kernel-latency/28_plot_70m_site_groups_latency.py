"""Single 70M panel: h/z and complementary S_model contributions versus latency."""
import hashlib
import json
import math
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'data/paper-checkpoints.json'
REFERENCE = HERE / 'data/14m-70m-quality-sparsity-latency.json'
STYLE = HERE / '14_plot_14m_quality_latency.py'
STYLES = [s for s in import_module(STYLE.stem).STYLES if s[0] in ['4', '7']]
OPERATIONS = {'h': 'mlp_w2', 'z': 'attention_output_projection',
              'a': 'qkv_projection', 'm': 'mlp_w1', 'qk': 'qk_scores', 'v': 'probability_value'}
GROUPS = {'hz': ['h', 'z'], 'complement': ['a', 'm', 'qk', 'v']}
OUTPUT = HERE / 'figures/17-70m-site-groups-sparsity-latency.pdf'
EXPORT = HERE / 'data/70m-site-groups-sparsity-latency.json'
TITLE = 'Sparsity contributions vs. latency on Pythia-70M'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    refs = {p['checkpoint_key']: p for p in reference['trained_points']
            if p['model'] == '70M' and p['scope'] in ['4', '7']}
    rows = [r for r in data['checkpoints'] if r['checkpoint_key'] in refs]
    assert len(rows) == len(refs) == 20
    assert reference['sources_sha256'][SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, REFERENCE, STYLE]}
    points = []
    for row in rows:
        original = ROOT / row['source_attempt'] / 'diagnostics/logical_products.json'
        key = original.relative_to(ROOT).as_posix()
        sources[key] = sha(original)
        assert sources[key] == data['sources_sha256'][key]
        logical = json.loads(original.read_text(encoding='utf-8'))
        counts = logical['measured']
        assert counts == row['counts']
        coverage = logical['coverage']
        assert coverage['complete_block_coverage'] and coverage['sequences'] == 338
        assert coverage['input_tokens'] == 692224 and coverage['excluded_tail_tokens'] == 1444
        operations = counts['per_operation']
        denominator = counts['model_product_count']
        assert denominator == counts['block_product_count'] + counts['lm_head_product_count']
        assert set(operations) == set(OPERATIONS.values())
        assert sum(v['product_count'] for v in operations.values()) == counts['block_product_count']
        zeros = {site: operations[op]['zero_product_count'] for site, op in OPERATIONS.items()}
        assert all(isinstance(v, int) and 0 <= v <= operations[OPERATIONS[s]]['product_count']
                   for s, v in zeros.items())
        grouped = {group: sum(zeros[site] for site in sites) for group, sites in GROUPS.items()}
        assert sum(grouped.values()) == counts['block_zero_product_count']
        contributions = {g: 100 * n / denominator for g, n in grouped.items()}
        assert math.isclose(sum(contributions.values()), row['sparsity'], abs_tol=1e-12)
        assert row['latency_ms'] == refs[row['checkpoint_key']]['latency_ms']
        assert row['kernel'] == 'k050-70m-v2' and row['timing_session'] == 'Run035'
        assert len(row['process_latency_ms']) == 3
        assert math.isclose(row['latency_ms'], math.exp(sum(map(math.log, row['process_latency_ms'])) / 3),
                            rel_tol=1e-12)
        point = {k: row[k] for k in ['checkpoint_key', 'model', 'scope', 'pressure', 'kappa',
                 'latency_ms', 'process_latency_ms', 'sparsity', 'kernel', 'timing_session',
                 'timing_device_uuid']}
        point.update(contributions_pp=contributions, group_zero_product_counts=grouped,
                     operation_zero_product_counts=zeros, model_product_count=denominator,
                     logical_source=key, coverage=coverage)
        points.append(point)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.subplots_adjust(left=.11, right=.97, top=.86, bottom=.29)
    ax.set_title(TITLE, pad=18)
    series = []
    for scope, pressure, label, color, linestyle in STYLES:
        recipe = sorted([p for p in points if (p['scope'], p['pressure']) == (scope, pressure)],
                        key=lambda p: p['kappa'])
        assert [p['kappa'] for p in recipe] == [0, .01, .05, .1, .5]
        for group in ['complement', 'hz']:
            x = [p['contributions_pp'][group] for p in recipe]
            y = [p['latency_ms'] for p in recipe]
            ax.plot(x, y, color=color, ls=linestyle, lw=1.35, zorder=2)
            is_hz = group == 'hz'
            ax.scatter(x, y, marker='o' if is_hz else 's', s=42 if is_hz else 58,
                       facecolors=color if is_hz else 'white',
                       edgecolors='white' if is_hz else color,
                       linewidths=.6 if is_hz else 1.3, zorder=4 if is_hz else 3)
            series.append({'scope': scope, 'pressure': pressure, 'label': label, 'group': group,
                           'color': color, 'linestyle': linestyle,
                           'keys': [p['checkpoint_key'] for p in recipe]})
    limits = {'x': [10.0, 26.0], 'y': [1.52, 3.28]}
    ax.set(xlim=limits['x'], ylim=limits['y'],
           xlabel=r'Contribution to $S_{\mathrm{model}}$ (pp)',
           ylabel='Full-model latency (ms)')
    ax.set_xticks([10, 15, 20, 25])
    ax.set_yticks([1.6, 2.0, 2.4, 2.8, 3.2])
    ax.tick_params(length=3, width=.65)
    ax.grid(axis='y', color='#E8EAED', lw=.6)
    ax.set_axisbelow(True)
    for p in points:
        assert all(limits['x'][0] <= x <= limits['x'][1] for x in p['contributions_pp'].values())
        assert limits['y'][0] <= p['latency_ms'] <= limits['y'][1]
    recipe_handles = [Line2D([], [], color=c, ls=ls, lw=1.6, label=label)
                      for _, _, label, c, ls in STYLES]
    fig.legend(handles=recipe_handles, loc='lower center', ncol=4, frameon=False,
               fontsize=11, columnspacing=1.7, handlelength=2.3,
               bbox_to_anchor=(.54, .105))
    group_handles = [
        Line2D([], [], ls='none', marker='o', ms=math.sqrt(42), mfc='#565B61', mec='white',
               mew=.6, label=r'$h,z$'),
        Line2D([], [], ls='none', marker='s', ms=math.sqrt(58), mfc='white', mec='#565B61',
               mew=1.3, label=r'$a,m,q,k,v$ (complement)'),
    ]
    fig.legend(handles=group_handles, loc='lower center', ncol=2, frameon=False,
               fontsize=10.5, columnspacing=2.0, handletextpad=.6,
               bbox_to_anchor=(.54, .025))
    fig.savefig(OUTPUT, metadata={'Title': TITLE, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert preserved == {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    result = {
        'question': 'Compare h/z and complementary sparsity contributions against latency in a single 70M panel.',
        'cohort': '20 trained 70M T/P settings from Figure 1; controls and post-hoc clipping omitted',
        'x_definition': '100 * selected operation zero-product counts / full-model product count',
        'x_unit': 'percentage points of S_model; denominator includes dense vocabulary head',
        'groups': GROUPS, 'operations': OPERATIONS,
        'complement_identity': 'hz contribution + complementary contribution = S_model; QK counted once',
        'y_definition': 'Run035 k050-70m-v2 full-model latency per 2048-token sequence, geometric mean of 1344 timings',
        'y_unit': 'milliseconds', 'points': points, 'series': series, 'limits': limits,
        'displayed_markers': 40, 'unique_checkpoints': 20,
        'encoding': 'Recipe color and pressure line style; filled circles h/z, open squares complement. Each checkpoint appears twice at the same latency.',
        'precision_note': 'Canonical FP16 logical counts; BF16 timings on RTX5090, batch one, full 50304 vocabulary logits.',
        'kernel_note': 'Qualified shape-specific 70M K050 port; these are full-model latencies, not separately timed site groups.',
        'timing_workload': rows[0]['timing_workload'],
        'sources_sha256': sources, 'script': Path(__file__).name,
        'script_sha256': sha(Path(__file__)), 'output': OUTPUT.relative_to(HERE).as_posix(),
        'output_sha256': sha(OUTPUT), 'preserved_pdf_sha256': preserved,
    }
    EXPORT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(OUTPUT), 'checkpoints': len(points), 'plotted_markers': 2 * len(points),
                      'ranges_pp': {g: [min(p['contributions_pp'][g] for p in points),
                                        max(p['contributions_pp'][g] for p in points)] for g in GROUPS},
                      'latency_range_ms': [min(p['latency_ms'] for p in points), max(p['latency_ms'] for p in points)]}))


if __name__ == '__main__':
    main()
