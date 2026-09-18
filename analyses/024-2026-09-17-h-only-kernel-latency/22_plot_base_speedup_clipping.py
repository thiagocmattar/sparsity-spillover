"""Base-model-normalized speedup with Figure 08's cohort, style and clipping paths."""
import json
import math
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

reference = import_module('16_plot_matched_quality_latency')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / 'figures/13-14m-70m-sparsity-base-speedup.pdf'
DATA = HERE / 'data/14m-70m-sparsity-base-speedup.json'
TITLE = 'Sparsity vs. full-model speedup on Pythia 14M and 70M'
LIMITS = {'14M': {'x': [-.75, 31], 'y': [.77, 1.48]},
          '70M': {'x': [-1.25, 51], 'y': [.85, 2.16]}}


def normalized_points(trained, clips):
    """Use one unrounded, optimized A0 latency per size for every plotted ratio."""
    bases = {}
    for size in LIMITS:
        base, = [r for r in trained if r['model'] == size and r['scope'] == '0']
        bases[size] = {k: base[k] for k in ['checkpoint_key', 'latency_ms', 'kernel',
                                            'timing_session', 'timing_device_uuid']}
    points = []
    for kind, rows in [('trained', trained), ('posthoc', clips)]:
        for row in rows:
            if kind == 'trained':
                point = {k: row[k] for k in ['checkpoint_key', 'model', 'scope', 'pressure',
                         'kappa', 'sparsity', 'loss', 'latency_ms', 'kernel',
                         'timing_session', 'timing_device_uuid']}
                point['zero_product_count'] = row['counts']['block_zero_product_count']
                point['model_product_count'] = row['counts']['model_product_count']
            else:
                point = dict(row)
            base = bases[row['model']]
            assert math.isfinite(row['latency_ms']) and row['latency_ms'] > 0
            assert row['kernel'].lower() == base['kernel'].lower()
            assert math.isclose(row['sparsity'], 100 * point['zero_product_count'] /
                                point['model_product_count'], abs_tol=1e-10)
            point.update(kind=kind, base_latency_ms=base['latency_ms'],
                         base_checkpoint_key=base['checkpoint_key'],
                         speedup=base['latency_ms'] / row['latency_ms'])
            if kind == 'trained':
                assert math.isclose(point['speedup'], row['dense_speedup'], abs_tol=1e-12)
            points.append(point)
    assert len(points) == 84
    return bases, points


def main():
    preserved = {p.name: reference.sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    data = json.loads(reference.SOURCE.read_text(encoding='utf-8'))
    timings = json.loads(reference.CLIPPING.read_text(encoding='utf-8'))
    trained, clips = reference.joined_points(data, timings)
    bases, points = normalized_points(trained, clips)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 11.5, 'axes.labelsize': 11, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.45))
    fig.subplots_adjust(left=.08, right=.985, top=.82, bottom=.28, wspace=.28)
    fig.suptitle(TITLE, fontsize=14, y=.985)
    series = []
    for ax, size, letter in zip(axes, LIMITS, 'ab'):
        rows = [p for p in points if p['model'] == size and p['kind'] == 'trained']
        clipping = [p for p in points if p['model'] == size and p['kind'] == 'posthoc']
        assert len(rows) == 22 and len(clipping) == 20
        for scope, pressure, label, color, ls in reference.STYLES:
            group = sorted([r for r in rows if (r['scope'], r['pressure']) == (scope, pressure)],
                           key=lambda r: -1 if r['kappa'] is None else r['kappa'])
            assert len(group) == (1 if scope in ['0', '1'] else 5)
            if scope in ['4', '7']:
                assert [r['kappa'] for r in group] == [0, .01, .05, .1, .5]
            ax.plot([p['sparsity'] for p in group], [p['speedup'] for p in group],
                    color=color, ls=ls, lw=1.6, marker='o', ms=9 if scope in ['0', '1'] else 5.8,
                    mfc='white' if scope == '0' else color, mec=color if scope == '0' else 'white',
                    mew=1.5 if scope == '0' else .7, zorder=5 if scope in ['0', '1'] else 4)
            series.append({'model': size, 'kind': 'trained', 'scope': scope, 'pressure': pressure,
                           'color': color, 'linestyle': ls, 'label': label,
                           'keys': [p['checkpoint_key'] for p in group]})
        for scope, color in [('0', reference.STYLES[0][3]), ('1', reference.STYLES[1][3])]:
            group = sorted([p for p in clipping if p['scope'] == scope], key=lambda p: p['target'])
            assert [p['target'] for p in group] == [p / 10 for p in range(10)]
            ax.plot([p['sparsity'] for p in group], [p['speedup'] for p in group],
                    color=color, lw=1.3, ls=':', zorder=2)
            series.append({'model': size, 'kind': 'posthoc', 'scope': scope,
                           'color': color, 'linestyle': ':', 'ids': [p['id'] for p in group]})
        ax.axhline(1, color='#92969B', lw=.8, zorder=1)
        ax.text(4.5 if size == '14M' else 11, .80 if size == '14M' else .89,
                'Post-hoc', color='#646970', fontsize=9.5, ha='left', va='center')
        ax.set(xlim=LIMITS[size]['x'], ylim=LIMITS[size]['y'],
               xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)',
               ylabel=r'Full-model speedup ($\times$)')
        ax.set_title(f'({letter}) {size}', loc='left', pad=11)
        ax.set_xticks([0, 10, 20, 30] if size == '14M' else [0, 10, 20, 30, 40, 50])
        ax.set_yticks([.8, 1., 1.2, 1.4] if size == '14M' else [1., 1.2, 1.4, 1.6, 1.8, 2.])
        ax.tick_params(length=3, width=.65)
        ax.grid(axis='y', color='#E8EAED', lw=.6, zorder=0)
        ax.set_axisbelow(True)
        for p in rows + clipping:
            assert LIMITS[size]['x'][0] <= p['sparsity'] <= LIMITS[size]['x'][1]
            assert LIMITS[size]['y'][0] <= p['speedup'] <= LIMITS[size]['y'][1]
    handles = [Line2D([], [], color=c, ls=ls, marker='o', ms=9 if s in ['0', '1'] else 5.8,
               mfc='white' if s == '0' else c, mec=c if s == '0' else 'white',
               mew=1.5 if s == '0' else .7, lw=1.6, label=l) for s, _, l, c, ls in reference.STYLES]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=11,
               handlelength=2.3, columnspacing=2.3, labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={'Title': TITLE, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert preserved == {p.name: reference.sha(p) for p in (HERE / 'figures').glob('*.pdf') if p != OUTPUT}
    sources = [reference.SOURCE, reference.CLIPPING, Path(reference.__file__), reference.STYLE_SOURCE,
               HERE / 'figures/08-14m-70m-quality-sparsity-latency.pdf',
               HERE / 'figures/A3-dense-reference-speedup.pdf']
    result = {'title': TITLE, 'output': OUTPUT.relative_to(HERE).as_posix(),
              'output_sha256': reference.sha(OUTPUT), 'script': Path(__file__).name,
              'script_sha256': reference.sha(Path(__file__)),
              'sources_sha256': {p.relative_to(ROOT).as_posix(): reference.sha(p) for p in sources},
              'normalization': 'same-size historical optimized A0 geometric-mean latency / optimized setting geometric-mean latency',
              'base_references': bases, 'points': points, 'series': series, 'limits': LIMITS,
              'layout': {'rows': 1, 'columns': 2, 'width_inches': 10.8, 'height_inches': 4.45},
              'preserved_pdf_sha256': preserved,
              'clipping_note': 'All 40 measured doses; recurring clipping is included in latency. No p=0 offset or frontier-specific renormalization.',
              'session_note': 'A0 references are Run029/Run035; 14M T7/Ph is Run033; clipping is Run036. Normalization does not remove session effects.',
              'precision_note': timings['precision_note'], 'kernel_note': timings['kernel_note']}
    DATA.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(OUTPUT), 'trained': 44, 'posthoc': 40,
                      'base_ms': {s: b['latency_ms'] for s, b in bases.items()}}))


if __name__ == '__main__':
    main()
