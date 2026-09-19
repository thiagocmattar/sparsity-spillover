"""Complement of Figure 15's h/z contributions, with the same 14M T/P latencies."""
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
HZ_SOURCE = HERE / 'data/14m-hz-sparsity-latency.json'
STYLE = HERE / '14_plot_14m_quality_latency.py'
STYLES = [s for s in import_module(STYLE.stem).STYLES if s[0] in ['4', '7']]
OPERATIONS = {'a': 'qkv_projection', 'm': 'mlp_w1', 'qk': 'qk_scores', 'v': 'probability_value'}
OUTPUT = HERE / 'figures/16-14m-complement-sparsity-latency.pdf'
EXPORT = HERE / 'data/14m-complement-sparsity-latency.json'
TITLE = r'Sparsity outside $h,z$ vs. latency on Pythia-14M'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    hz = json.loads(HZ_SOURCE.read_text(encoding='utf-8'))
    assert hz['sources_sha256'][SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    assert len(hz['points']) == 20
    lookup = {r['checkpoint_key']: r for r in data['checkpoints']}
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in [SOURCE, HZ_SOURCE, STYLE]}
    points = []
    for reference in hz['points']:
        row = lookup[reference['checkpoint_key']]
        assert row['model'] == '14M' and row['scope'] in ['4', '7'] and row['pressure'] in ['h', 'all']
        original = ROOT / reference['logical_source']
        source_key = original.relative_to(ROOT).as_posix()
        sources[source_key] = sha(original)
        assert sources[source_key] == data['sources_sha256'][source_key] == hz['sources_sha256'][source_key]
        logical = json.loads(original.read_text(encoding='utf-8'))
        counts = logical['measured']
        assert counts == row['counts'] and logical['coverage'] == reference['coverage']
        assert logical['coverage']['sequences'] == 338
        operations = counts['per_operation']
        denominator = counts['model_product_count']
        assert denominator == reference['model_product_count']
        assert denominator == counts['block_product_count'] + counts['lm_head_product_count']
        assert set(operations) == set(OPERATIONS.values()) | {'mlp_w2', 'attention_output_projection'}
        zeros = {site: operations[op]['zero_product_count'] for site, op in OPERATIONS.items()}
        assert all(isinstance(v, int) and 0 <= v <= operations[OPERATIONS[s]]['product_count']
                   for s, v in zeros.items())
        numerator = sum(zeros.values())
        assert numerator + sum(reference['zero_product_counts'].values()) == counts['block_zero_product_count']
        contribution = 100 * numerator / denominator
        assert math.isclose(contribution + reference['hz_contribution_pp'], row['sparsity'], abs_tol=1e-12)
        assert row['latency_ms'] == reference['latency_ms']
        point = {k: reference[k] for k in ['checkpoint_key', 'model', 'scope', 'pressure', 'kappa',
                 'latency_ms', 'process_latency_ms', 'sparsity', 'kernel', 'timing_session',
                 'timing_device_uuid', 'timing_workload', 'logical_source', 'coverage', 'hz_contribution_pp']}
        point.update(complement_contribution_pp=contribution,
                     contributions_pp={s: 100 * v / denominator for s, v in zeros.items()},
                     zero_product_counts=zeros, model_product_count=denominator)
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
                       key=lambda p: p['kappa'])
        assert [p['kappa'] for p in group] == [0, .01, .05, .1, .5]
        x = [p['complement_contribution_pp'] for p in group]
        y = [p['latency_ms'] for p in group]
        ax.plot(x, y, color=color, ls=linestyle, lw=1.4, zorder=2)
        ax.scatter(x, y, s=50, color=color, edgecolors='white', linewidths=.65, zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=1.4, marker='o',
                              ms=math.sqrt(50), mfc=color, mec='white', mew=.65, label=label))
    limits = {'x': [2.8, 23.0], 'y': hz['limits']['y']}
    ax.set(xlim=limits['x'], ylim=limits['y'],
           xlabel=r'$a,m,q,k,v$ contribution to $S_{\mathrm{model}}$ (pp)',
           ylabel='Full-model latency (ms)')
    ax.set_xticks([5, 10, 15, 20])
    ax.set_yticks([.45, .50, .55, .60, .65])
    ax.tick_params(length=3, width=.65)
    ax.grid(axis='y', color='#E8EAED', lw=.6)
    ax.set_axisbelow(True)
    for p in points:
        assert limits['x'][0] <= p['complement_contribution_pp'] <= limits['x'][1]
        assert limits['y'][0] <= p['latency_ms'] <= limits['y'][1]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=11, handletextpad=.45, columnspacing=1.8,
               labelspacing=.8, bbox_to_anchor=(.55, .02))
    fig.savefig(OUTPUT, metadata={'Title': 'Sparsity outside h,z vs. latency on Pythia-14M',
                                 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert preserved == {p.name: sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    result = {
        'question': 'How does full-model latency vary with contributions to S_model from all sites except h,z?',
        'x_definition': '100 * sum(zero products in qkv_projection, mlp_w1, qk_scores, probability_value) / full-model products',
        'x_unit': 'percentage points of S_model; unchanged full-model denominator, including dense LM head',
        'complement_identity': 'complement_contribution_pp + hz_contribution_pp = S_model percent; QK counted once',
        'y_definition': hz['y_definition'], 'y_unit': 'milliseconds', 'cohort': hz['cohort'],
        'points': points, 'limits': limits,
        'caption': 'Four T/P recipes at five thresholds; dashed Ph and solid Pall lines connect increasing kappa. Base and ReLU controls omitted.',
        'precision_note': hz['precision_note'], 'session_note': hz['session_note'],
        'sources_sha256': sources, 'script': Path(__file__).name,
        'script_sha256': sha(Path(__file__)), 'output': OUTPUT.relative_to(HERE).as_posix(),
        'output_sha256': sha(OUTPUT), 'preserved_pdf_sha256': preserved,
    }
    EXPORT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(OUTPUT), 'points': len(points), 'verified_complement_identities': len(points),
                      'x_range_pp': [min(p['complement_contribution_pp'] for p in points),
                                     max(p['complement_contribution_pp'] for p in points)]}))


if __name__ == '__main__':
    main()
