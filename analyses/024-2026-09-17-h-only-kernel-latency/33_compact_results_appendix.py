"""Compact endpoint tables and a restyled 14M post-hoc view from retained data."""
import hashlib
import json
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
QUALITY = HERE / 'data/all-model-quality-sparsity.json'
CLIPPING = ROOT / 'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json'
STYLES = import_module('20_plot_quality_all_sizes').STYLES
OUTPUT = HERE / 'figures/21-14m-posthoc-quality-sparsity.pdf'
TABLES = HERE / 'tables/compact-results'
SHARED = [('0', 'none'), ('1', 'none'), ('4', 'h'), ('4', 'all'), ('7', 'h'), ('7', 'all')]
ONLY14 = [('1', 'L1'), ('1', 'h'), ('4', 'none'), ('7', 'none')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter(row):
    return row['local_pressure_weight'] if row['scope'] == '1' and row['pressure'] != 'none' else row['kappa']


def group(rows, size, pair):
    return sorted([r for r in rows if r['model'] == size and (r['scope'], r['pressure']) == pair],
                  key=lambda r: -1 if parameter(r) is None else parameter(r))


def recipe(pair):
    scope, pressure = pair
    target = {'none': '0', 'h': '1' if scope == '1' else 'h', 'all': r'\mathrm{all}', 'L1': '1'}[pressure]
    suffix = {('0', 'none'): ' (Base)', ('1', 'none'): ' (ReLU)',
              ('1', 'L1'): ' (L1)', ('1', 'h'): ' (OL1)'}.get(pair, '')
    return rf'$T_{scope}/P_{{{target}}}$' + suffix


def number(value):
    return '---' if value is None else f'{value:g}'


def build():
    quality = json.loads(QUALITY.read_text(encoding='utf-8'))
    clipping = json.loads(CLIPPING.read_text(encoding='utf-8'))
    rows = quality['trained_points']
    parts = {
        '14m-only': [r for pair in ONLY14 for r in group(rows, '14M', pair)],
        '14m-70m': [r for pair in SHARED for size in ['14M', '70M'] for r in group(rows, size, pair)],
        '410m': [r for r in rows if r['model'] == '410M'],
    }
    assert [len(v) for v in parts.values()] == [18, 44, 12]
    included = [r['source_attempt'] for values in parts.values() for r in values]
    assert len(included) == len(set(included)) == 74
    assert set(included) == {r['source_attempt'] for r in rows}
    for r in rows:
        assert r['sparsity'] == 100 * r['zero_product_count'] / r['model_product_count']
    by_hash = {r['final_checkpoint_content_sha256']: r for r in rows if r['model'] == '14M'}
    clips = []
    for p in clipping['points']:
        if p['scale'] != '14M':
            continue
        r = by_hash[p['checkpoint_content_sha256']]
        aliases = {'A0': ('0', 'none'), 'A1-H': ('1', 'none'),
                   'A1-H-L1': ('1', 'L1'), 'A1-H-OL1': ('1', 'h'),
                   'A4': ('4', 'none'), 'A4-OL1': ('4', 'all'),
                   'A7': ('7', 'none'), 'A7-OL1': ('7', 'all')}
        assert aliases[p['family']] == (r['scope'], r['pressure'])
        assert p['training_parameter'] == parameter(r)
        assert all(p['coverage'][k] == v for k, v in quality['coverage'].items())
        counts = p['counts']
        clips.append({'id': p['id'], 'source_attempt': r['source_attempt'],
                      'checkpoint_sha256': p['checkpoint_content_sha256'],
                      'scope': r['scope'], 'pressure': r['pressure'], 'training_parameter': parameter(r),
                      'target': p['dose'], 'loss': p['loss'],
                      'sparsity': 100 * counts['block_zero_product_count'] / counts['model_product_count'],
                      'zero_product_count': counts['block_zero_product_count'],
                      'model_product_count': counts['model_product_count']})
    clip_ids = {p['source_attempt'] for p in clips}
    trained = [r for r in rows if r['source_attempt'] in clip_ids]
    assert len(trained) == 30 and len(clips) == 300
    for r in trained:
        path = [p for p in clips if p['source_attempt'] == r['source_attempt']]
        assert sorted(p['target'] for p in path) == [i / 10 for i in range(10)]
    return {'tables': parts, 'posthoc_trained': trained, 'posthoc_points': clips,
            'posthoc_unmeasured': [r['source_attempt'] for r in rows if r['model'] == '14M' and r['source_attempt'] not in clip_ids],
            'ceilings': quality['ceilings']['14M'], 'coverage': quality['coverage'],
            'sources_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [QUALITY, CLIPPING]},
            'loss_convention': quality['loss_convention']}


def write_table(name, caption, label, spec, header, body):
    text = '\n'.join([
        '% Source: Analysis024 observations/043-compact-results-appendix.md; 33_compact_results_appendix.py.',
        r'\begin{table}[H]' if name == '410m' else r'\begin{table}[!htbp]',
        r'\centering\small', r'\setlength{\tabcolsep}{6pt}',
        r'\renewcommand{\arraystretch}{1.05}', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        rf'\begin{{tabular}}{{{spec}}}', r'\toprule', *header, r'\midrule',
        *body, r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (TABLES / f'{name}.tex').write_text(text, encoding='utf-8', newline='\n')


def table_body(rows, pairs, sizes):
    body = []
    for i, pair in enumerate(pairs):
        if i:
            body.append(r'\midrule')
        groups = [group(rows, size, pair) for size in sizes]
        assert all([parameter(r) for r in g] == [parameter(r) for r in groups[0]] for g in groups)
        for j, point in enumerate(groups[0]):
            label = recipe(pair) if len(groups[0]) == 1 else rf'\multirow{{{len(groups[0])}}}{{*}}{{{recipe(pair)}}}'
            cells = [label if j == 0 else '']
            cells.append(number(point['kappa']))
            for g in groups:
                cells.extend([f"{g[j]['loss']:.4f}", f"{g[j]['sparsity']:.3f}"])
            body.append(' & '.join(cells) + r' \\')
    return body


def tables(data):
    TABLES.mkdir(parents=True, exist_ok=True)
    only = data['tables']['14m-only']
    body = []
    for block, (left, right) in enumerate([(ONLY14[0], ONLY14[1]), (ONLY14[2], ONLY14[3])]):
        if block:
            body += [r'\midrule', r'\multirow{2}{*}{$\kappa$} & \multicolumn{2}{c}{$T_4/P_0$} & \multicolumn{2}{c}{$T_7/P_0$} \\',
                     r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
                     r' & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\', r'\midrule']
        for a, b in zip(group(only, '14M', left), group(only, '14M', right)):
            assert parameter(a) == parameter(b)
            body.append(' & '.join([number(parameter(a)), f"{a['loss']:.4f}", f"{a['sparsity']:.3f}",
                                    f"{b['loss']:.4f}", f"{b['sparsity']:.3f}"]) + r' \\')
    write_table('14m-only', r'\textbf{Additional 14M ablations.} Pressure at $h$ (top) and thresholding without pressure (bottom), paired by $\lambda$ or $\kappa$. Controls appear in Table~\ref{tab:endpoints-14m-70m}.',
                'tab:endpoints-14m-only', '@{}lrrrr@{}',
                [r'\multirow{2}{*}{$\lambda$} & \multicolumn{2}{c}{$T_1/P_1$ (L1)} & \multicolumn{2}{c}{$T_1/P_1$ (OL1)} \\',
                 r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
                 r' & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\'], body)
    write_table('14m-70m', r'\textbf{Matched 14M/70M results.} The same threshold/pressure setting at both sizes. Pressure uses OL1 with $\lambda=b=1$.',
                'tab:endpoints-14m-70m', '@{}llrrrr@{}',
                [r'\multirow{2}{*}{Recipe} & \multirow{2}{*}{$\kappa$} & \multicolumn{2}{c}{14M} & \multicolumn{2}{c}{70M} \\',
                 r'\cmidrule(lr){3-4}\cmidrule(l){5-6}',
                 r' & & Loss & $\Smodel$ (\%) & Loss & $\Smodel$ (\%) \\'],
                table_body(data['tables']['14m-70m'], SHARED, ['14M', '70M']))
    write_table('410m', r'\textbf{410M fixed-token stress test.} All twelve trained conditions; pressure uses OL1 with $\lambda=b=1$. This size has no $P_h$ or unpressured $T_4/T_7$ runs.',
                'tab:endpoints-410m', '@{}llrr@{}',
                [r'Recipe & $\kappa$ & Loss & $\Smodel$ (\%) \\'],
                table_body(data['tables']['410m'], [('0', 'none'), ('1', 'none'), ('4', 'all'), ('7', 'all')], ['410M']))


def plot(data):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                        'axes.titlesize': 12, 'axes.labelsize': 12,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(9.72, 5.35))
    fig.subplots_adjust(left=.08, right=.985, bottom=.29, top=.82, wspace=.24)
    fig.suptitle('Post-hoc clipping on trained Pythia-14M models', fontsize=15, y=.98)
    rows, clips = data['posthoc_trained'], data['posthoc_points']
    styles = [s for s in STYLES if any((r['scope'], r['pressure']) == s[:2] for r in rows)]
    limits = [(5.0, 6.25), (5.0, 9.6)]
    for ax, title, ylim in zip(axes, ['(a) Detail near trained models', '(b) Full loss range'], limits):
        for scope, pressure, label, color, ls in styles:
            points = group(rows, '14M', (scope, pressure))
            for r in points:
                path = sorted([p for p in clips if p['source_attempt'] == r['source_attempt']], key=lambda p: p['target'])
                ax.plot([p['sparsity'] for p in path], [p['loss'] for p in path],
                        color=color, ls=':', lw=.8, alpha=.55, marker='o', ms=2.2,
                        mfc='white', mec=color, mew=.4, zorder=2)
            ax.plot([r['sparsity'] for r in points], [r['loss'] for r in points],
                    color=color, ls=ls, lw=1.5, marker='o', ms=5.8 if len(points) == 1 else 4.7,
                    mfc='white' if scope == '0' else color,
                    mec=color if scope == '0' else 'white', mew=.9 if scope == '0' else .5, zorder=4)
        for scope in ['4', '7']:
            ceiling = data['ceilings'][scope]['R_model_max_percent']
            ax.axvline(ceiling, color='#92969B', lw=.75, ls=(0, (2, 3)), zorder=1)
            ax.text(ceiling + .4 if scope == '4' else ceiling - .4, .98, rf'$T_{scope}$ ceiling',
                    transform=ax.get_xaxis_transform(), ha='left' if scope == '4' else 'right',
                    va='top', fontsize=8.5, color='#6C7177')
        ax.set(xlim=(-.8, 31), ylim=ylim, xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
        ax.set_title(title, loc='left', pad=10)
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(length=3, width=.65)
        ax.grid(axis='y', color='#E8EAED', lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Validation loss')
    handles = [Line2D([], [], color=c, ls=ls, marker='o', ms=5, lw=1.5,
                      mfc='white' if s == '0' else c, label=label)
               for s, p, label, c, ls in styles]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=10.5,
               handlelength=1.8, columnspacing=1.4, labelspacing=1., bbox_to_anchor=(.52, .06))
    fig.text(.52, .035, 'Dotted paths / open markers: post-hoc clipping of each fixed model',
             ha='center', fontsize=10, color='#6C7177')
    fig.savefig(OUTPUT, metadata={'Title': '14M post-hoc quality-sparsity detail', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    data['figure'] = {'path': OUTPUT.relative_to(HERE).as_posix(), 'sha256': sha(OUTPUT),
                      'panels': [{'loss_limits': list(l), 'outside': [p['id'] for p in clips if not l[0] <= p['loss'] <= l[1]]} for l in limits],
                      'styles': styles}


def main():
    data = build()
    tables(data)
    plot(data)
    data['script_sha256'] = sha(Path(__file__))
    data['style_source_sha256'] = sha(HERE / '20_plot_quality_all_sizes.py')
    data['table_sha256'] = {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(TABLES.glob('*.tex'))}
    (HERE / 'data/compact-results-appendix.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'table_endpoints': {k: len(v) for k, v in data['tables'].items()},
                      'posthoc_checkpoints': len(data['posthoc_trained']), 'posthoc_points': len(data['posthoc_points'])}))


if __name__ == '__main__':
    main()
