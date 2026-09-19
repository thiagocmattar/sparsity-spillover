"""Split the retained all-scale quality overview into main-text and appendix views."""
import json
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

original = import_module('20_plot_quality_all_sizes')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'data/all-model-quality-sparsity.json'
VIEWS = [
    {'name': '14-14m-70m-quality-sparsity.pdf', 'sizes': ['14M', '70M'],
     'title': 'Quality-sparsity trade-offs on Pythia 14M and 70M',
     'figsize': [10.8, 4.55], 'loss_limits': [4., 6.2]},
    {'name': 'A5-410m-quality-sparsity.pdf', 'sizes': ['410M'],
     'title': 'Pythia-410M: fixed-token sparsity stress test',
     'figsize': [8.6, 4.55], 'loss_limits': [4.2, 9.35]},
]


def draw(data, view):
    sizes = view['sizes']
    rows = [p for p in data['trained_points'] if p['model'] in sizes]
    clips = [p for p in data['clipping_points'] if p['model'] in sizes]
    fig, axes = plt.subplots(1, len(sizes), figsize=view['figsize'], sharey=True, squeeze=False)
    fig.subplots_adjust(left=.085, right=.985, top=.82, bottom=.34 if len(sizes) == 2 else .25,
                        wspace=.25)
    fig.suptitle(view['title'], fontsize=15, y=.98)
    series, panels = [], []
    styles = [s for s in original.STYLES if any((r['scope'], r['pressure']) == s[:2] for r in rows)]
    for ax, size, letter in zip(axes[0], sizes, 'ab'):
        trained = [r for r in rows if r['model'] == size]
        clipping = [r for r in clips if r['model'] == size]
        assert len(trained) == {'14M': 40, '70M': 22, '410M': 12}[size]
        assert len(clipping) == 20
        for scope, pressure, label, color, ls in styles:
            group = [r for r in trained if (r['scope'], r['pressure']) == (scope, pressure)]
            if not group:
                continue
            dose = 'local_pressure_weight' if scope == '1' and pressure != 'none' else 'kappa'
            group.sort(key=lambda r: -1 if r[dose] is None else r[dose])
            if scope in ['4', '7']:
                assert [r[dose] for r in group] == original.KAPPAS
            control = scope in ['0', '1'] and pressure == 'none'
            ax.plot([r['sparsity'] for r in group], [r['loss'] for r in group],
                    color=color, ls=ls, marker='o', lw=1.55, ms=7.5 if control else 4.8,
                    mfc='white' if scope == '0' else color,
                    mec=color if scope == '0' else 'white', mew=1.25 if scope == '0' else .55,
                    zorder=5 if control else 4)
            series.append({'model': size, 'scope': scope, 'pressure': pressure, 'label': label,
                           'color': color, 'linestyle': ls,
                           'source_attempts': [r['source_attempt'] for r in group]})
        for scope, color in [('0', original.STYLES[0][3]), ('1', original.STYLES[1][3])]:
            group = sorted([p for p in clipping if p['scope'] == scope], key=lambda p: p['target'])
            assert [p['target'] for p in group] == [p/10 for p in range(10)]
            ax.plot([p['sparsity'] for p in group], [p['loss'] for p in group],
                    color=color, ls=':', lw=1.25, zorder=2)
        for scope in ['4', '7']:
            ceiling = data['ceilings'][size][scope]['R_model_max_percent']
            ax.axvline(ceiling, color='#92969B', lw=.75, ls=(0, (2, 3)), alpha=.8, zorder=1)
            right = size == '14M' and scope == '4'
            ax.text(ceiling + .5 if right else ceiling - .4, .98, rf'$T_{scope}$ ceiling',
                    transform=ax.get_xaxis_transform(), ha='left' if right else 'right',
                    va='top', fontsize=9.5, color='#6C7177')
        x, y, angle = {'14M': (3.4, 5.54, 70), '70M': (14.8, 4.80, 58),
                       '410M': (35., 5.85, 28)}[size]
        ax.text(x, y, 'Post-hoc', rotation=angle, color='#6C7177', fontsize=10,
                rotation_mode='anchor', ha='left', va='bottom')
        ax.set(xlim=original.LIMITS[size], ylim=view['loss_limits'],
               xlabel=r'Model-wide sparsity $S_{\mathrm{model}}$ (%)')
        if len(sizes) > 1:
            ax.set_title(f'({letter}) Pythia-{size}', loc='left', pad=10)
        ax.set_xticks({'14M': [0, 10, 20, 30], '70M': [0, 10, 20, 30, 40, 50],
                       '410M': [0, 15, 30, 45, 60, 75, 90]}[size])
        ax.set_yticks([4., 4.5, 5., 5.5, 6.] if size != '410M' else [5., 6., 7., 8., 9.])
        ax.tick_params(length=3, width=.65, labelleft=True)
        ax.grid(axis='y', color='#E8EAED', lw=.6, zorder=0)
        ax.set_axisbelow(True)
        lo, hi = view['loss_limits']
        assert all(lo <= r['loss'] <= hi for r in trained)
        outside = [p['id'] for p in clipping if not lo <= p['loss'] <= hi]
        if size == '410M':
            assert not outside
        panels.append({'model': size, 'trained_count': len(trained), 'clipping_count': len(clipping),
                       'clipping_outside_y': outside})
    axes[0, 0].set_ylabel('Validation loss')
    handles = [Line2D([], [], color=c, ls=ls, marker='o', ms=6,
                    mfc='white' if s == '0' else c, mec=c if s == '0' else 'white',
                    mew=1 if s == '0' else .5, lw=1.55, label=label)
               for s, p, label, c, ls in styles]
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9] if len(styles) == 10 else list(range(len(styles)))
    fig.legend(handles=[handles[i] for i in order], loc='lower center', ncol=5 if len(styles) == 10 else 4,
               frameon=False, fontsize=11, handlelength=2., columnspacing=1.35,
               labelspacing=.9, bbox_to_anchor=(.52, .035))
    output = HERE / 'figures' / view['name']
    fig.savefig(output, metadata={'Title': view['title'], 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return dict(view, output=output.relative_to(HERE).as_posix(), output_sha256=original.sha(output),
                panels=panels, series=series, trained_points=rows, clipping_points=clips,
                ceilings={s: data['ceilings'][s] for s in sizes})


def main():
    names = {v['name'] for v in VIEWS}
    preserved = {p.name: original.sha(p) for p in (HERE/'figures').glob('*.pdf') if p.name not in names}
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5,
                        'axes.titlesize': 12.5, 'axes.labelsize': 12.5,
                        'xtick.labelsize': 11, 'ytick.labelsize': 11,
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'axes.linewidth': .65, 'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    outputs = [draw(data, v) for v in VIEWS]
    assert sum(len(v['trained_points']) for v in outputs) == len(data['trained_points']) == 74
    assert sum(len(v['clipping_points']) for v in outputs) == len(data['clipping_points']) == 60
    assert preserved == {p.name: original.sha(p) for p in (HERE/'figures').glob('*.pdf') if p.name not in names}
    result = {'source': SOURCE.relative_to(ROOT).as_posix(), 'source_sha256': original.sha(SOURCE),
              'script': Path(__file__).name, 'script_sha256': original.sha(Path(__file__)),
              'style_source_sha256': original.sha(Path(original.__file__)), 'outputs': outputs,
              'coverage': data['coverage'], 'preserved_pdf_sha256': preserved,
              'note': 'Exact partition of the retained 74 trained and 60 control-clipping records; no measurement or loss convention changed.'}
    (HERE/'data/quality-main-appendix.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({v['name']: v['panels'] for v in outputs}))


if __name__ == '__main__':
    main()
