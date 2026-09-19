"""14M Figure 1 T/P recipes: h+z contribution to S_model versus final-kernel latency."""
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
OUTPUT = HERE / 'figures/15-14m-hz-sparsity-latency.pdf'
EXPORT = HERE / 'data/14m-hz-sparsity-latency.json'
TITLE = r'$h,z$ sparsity contribution vs. latency on Pythia-14M'
OPERATIONS = {'h': 'mlp_w2', 'z': 'attention_output_projection'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    refs = {p['checkpoint_key']: p for p in reference['trained_points']
            if p['model'] == '14M' and p['scope'] in ['4', '7']}
    rows = [r for r in data['checkpoints'] if r['checkpoint_key'] in refs]
    assert len(rows) == len(refs) == 20
    assert reference['sources_sha256'][SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, REFERENCE, STYLE]}
    points = []
    for row in rows:
        original = ROOT / row['source_attempt'] / 'diagnostics/logical_products.json'
        source_key = original.relative_to(ROOT).as_posix()
        sources[source_key] = sha(original)
        assert sources[source_key] == data['sources_sha256'][source_key]
        logical = json.loads(original.read_text(encoding='utf-8'))
        counts = logical['measured']
        assert counts == row['counts']
        coverage = logical['coverage']
        assert coverage['complete_block_coverage'] and coverage['sequences'] == 338
        assert coverage['input_tokens'] == 692224 and coverage['excluded_tail_tokens'] == 1444
        operations = counts['per_operation']
        denominator = counts['model_product_count']
        assert denominator == counts['block_product_count'] + counts['lm_head_product_count']
        assert sum(v['product_count'] for v in operations.values()) == counts['block_product_count']
        assert sum(v['zero_product_count'] for v in operations.values()) == counts['block_zero_product_count']
        zeros = {site: operations[op]['zero_product_count'] for site, op in OPERATIONS.items()}
        assert all(isinstance(v, int) and 0 <= v <= operations[OPERATIONS[s]]['product_count']
                   for s, v in zeros.items())
        contribution = 100 * sum(zeros.values()) / denominator
        assert 0 <= contribution <= row['sparsity']
        assert row['latency_ms'] == refs[row['checkpoint_key']]['latency_ms']
        assert row['kernel'] == 'K050' and row['timing_precision'] == 'BF16'
        assert math.isclose(row['latency_ms'], math.exp(sum(map(math.log, row['process_latency_ms'])) / 3),
                            rel_tol=1e-12)
        point = {k: row[k] for k in ['checkpoint_key', 'model', 'scope', 'pressure', 'kappa',
                                    'latency_ms', 'process_latency_ms', 'sparsity', 'kernel',
                                    'timing_session', 'timing_device_uuid', 'timing_workload']}
        point.update(hz_contribution_pp=contribution,
                     contributions_pp={s: 100 * v / denominator for s, v in zeros.items()},
                     zero_product_counts=zeros, model_product_count=denominator,
                     logical_source=source_key, coverage=coverage)
        points.append(point)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.subplots_adjust(left=.13, right=.97, top=.85, bottom=.27)
    ax.set_title(TITLE, pad=18)
    handles = []
    for scope, pressure, label, color, linestyle in STYLES:
        group = sorted([p for p in points if (p['scope'], p['pressure']) == (scope, pressure)],
                       key=lambda p: -1 if p['kappa'] is None else p['kappa'])
        assert [p['kappa'] for p in group] == [0, .01, .05, .1, .5]
        ax.plot([p['hz_contribution_pp'] for p in group], [p['latency_ms'] for p in group],
                color=color, ls=linestyle, lw=1.4, zorder=2)
        face, edge = color, 'white'
        size, width = 50, .65
        ax.scatter([p['hz_contribution_pp'] for p in group], [p['latency_ms'] for p in group],
                   s=size, color=face, edgecolors=edge, linewidths=width,
                   zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=1.4, marker='o', ms=math.sqrt(size),
                              mfc=face, mec=edge, mew=width, label=label))
    limits = {'x': [3.25, 5.45], 'y': [.45, .65]}
    ax.set(xlim=limits['x'], ylim=limits['y'],
           xlabel=r'$h+z$ contribution to $S_{\mathrm{model}}$ (pp)',
           ylabel='Full-model latency (ms)')
    ax.set_xticks([3.5, 4, 4.5, 5])
    ax.set_yticks([.45, .50, .55, .60, .65])
    ax.tick_params(length=3, width=.65)
    ax.grid(axis='y', color='#E8EAED', lw=.6)
    ax.set_axisbelow(True)
    for p in points:
        assert limits['x'][0] <= p['hz_contribution_pp'] <= limits['x'][1]
        assert limits['y'][0] <= p['latency_ms'] <= limits['y'][1]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=11, handletextpad=.45, columnspacing=1.8,
               labelspacing=.8, bbox_to_anchor=(.55, .02))
    fig.savefig(OUTPUT, metadata={'Title': 'h,z sparsity contribution vs. latency on Pythia-14M',
                                 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert preserved == {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    result = {
        'question': 'How does full-model latency vary with the h+z contribution to model-wide sparsity?',
        'x_definition': '100 * (mlp_w2 zero products + attention_output_projection zero products) / full-model products',
        'x_unit': 'percentage points of S_model; full denominator retained, including dense LM head',
        'y_definition': 'K050 full-model host latency per 2048-token sequence, geometric mean of 1344 timings',
        'y_unit': 'milliseconds', 'cohort': '20 trained 14M T/P settings from manuscript Figure 1 / Analysis024 Figure 08; Base and ReLU controls excluded',
        'points': points, 'limits': limits,
        'caption': 'One point per trained recipe-threshold setting; colors identify the four T/P recipes. Lines connect increasing kappa within each recipe: dashed for Ph, solid for Pall. Base and ReLU controls omitted; no post-hoc clipping.',
        'precision_note': 'Canonical FP16 logical counts; qualified BF16 timings on RTX5090, batch one, full vocabulary output.',
        'session_note': '14M T7/Ph uses Run033; all other plotted timings use Run029. Small cross-session differences are descriptive.',
        'sources_sha256': sources, 'script': Path(__file__).name,
        'script_sha256': sha(Path(__file__)), 'output': OUTPUT.relative_to(HERE).as_posix(),
        'output_sha256': sha(OUTPUT), 'preserved_pdf_sha256': preserved,
    }
    EXPORT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(OUTPUT), 'points': len(points), 'verified_logical_sources': len(rows),
                      'x_range_pp': [min(p['hz_contribution_pp'] for p in points), max(p['hz_contribution_pp'] for p in points)],
                      'y_range_ms': [min(p['latency_ms'] for p in points), max(p['latency_ms'] for p in points)]}))


if __name__ == '__main__':
    main()
