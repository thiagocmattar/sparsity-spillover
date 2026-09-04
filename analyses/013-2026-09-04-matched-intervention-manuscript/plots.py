"""Publication figures for the selected matched-intervention evidence."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evidence import HERE, OPS, SITES, one

STYLE = {'A0': ('#555555', 'o'), 'A1-H': ('#999999', 's'),
         'A1-H+L1': ('#009E73', 'o'), 'A1-H+OL1': ('#CC79A7', '^'),
         'A4': ('#0072B2', 'o'), 'A4+OL1@h': ('#009E73', 'v'),
         'A4+OL1@4': ('#0072B2', 's'), 'A7': ('#D55E00', 'o'),
         'A7+OL1@7': ('#D55E00', '^')}
OP_NAMES = ('QKV', 'MLP up', 'MLP down', 'Attn. output', 'QK', 'PV')
OP_COLORS = ('#9ecae1', '#6baed6', '#2171b5', '#a1d99b', '#fd8d3c', '#bd0026')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.titlesize': 10, 'axes.labelsize': 9,
                     'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})


def finish(fig, name):
    path = HERE / 'figures' / name
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches='tight', metadata={'Creator': 'Analysis 013'})
    plt.close(fig)


def axis_style(ax):
    ax.grid(alpha=.23, linewidth=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(r'$R_{\rm model}$ (%)')
    ax.set_ylabel('Validation cross-entropy')


def curve(ax, rows, family, label=None):
    color, marker = STYLE[family]
    rows = sorted(rows, key=lambda r: r['dose'] if r['dose'] is not None else -1)
    ax.plot([r['R_model']*100 for r in rows], [r['loss'] for r in rows],
            color=color, marker=marker, markersize=4, linewidth=1.1,
            linestyle='--' if family in ('A4', 'A7') else '-',
            label=label or family)


def matched(data):
    rows = [r for r in data['trained'] if r['scale'] == '14M']
    panels = [('(a) Local pressure', ('A0', 'A1-H', 'A1-H+L1', 'A1-H+OL1')),
              ('(b) Pressure sites, A4 gates', ('A4', 'A4+OL1@h', 'A4+OL1@4')),
              ('(c) Attention gates and pressure', ('A4', 'A4+OL1@4', 'A7', 'A7+OL1@7'))]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.2), layout='constrained')
    for ax, (title, families) in zip(axs, panels):
        for fam in families:
            curve(ax, [r for r in rows if r['family'] == fam], fam)
        ax.set_title(title)
        axis_style(ax)
        ax.legend(loc='best', frameon=True, framealpha=.9, edgecolor='none', fontsize=7.6)
    axs[0].set_xlim(-.15, 4.3)
    axs[1].set_ylim(5.08, 6.15)
    axs[2].set_ylim(5.32, 6.15)
    finish(fig, '01-matched-14m.pdf')


def frontiers(data, scales, name):
    size = (7.1, 3.75) if len(scales) > 1 else (5.5, 2.8)
    fig, axs = plt.subplots(1, len(scales), figsize=size, squeeze=False, layout='constrained')
    for ax, scale in zip(axs[0], scales):
        for control, color in (('A0', '#555555'), ('A1-H', '#999999')):
            rows = [r for r in data['clipping'] if r['scale'] == scale and r['family'] == control+' clipped']
            ax.plot([100*r['R_model'] for r in rows], [r['loss'] for r in rows],
                    color=color, marker='.', ms=4, lw=1, label=control+' uniform clipping')
        families = ('A1-H+L1', 'A1-H+OL1', 'A4', 'A4+OL1@h',
                    'A4+OL1@4', 'A7', 'A7+OL1@7') if scale == '14M' else ('A4+OL1@4', 'A7+OL1@7')
        for fam in families:
            curve(ax, [r for r in data['trained'] if r['scale'] == scale and r['family'] == fam], fam)
        for fam in ('A0', 'A1-H'):
            curve(ax, [r for r in data['trained'] if r['scale'] == scale and r['family'] == fam],
                  fam, label='_nolegend_')
        ax.set_title('Pythia-'+scale)
        axis_style(ax)
        if len(scales) == 1:
            ax.legend(loc='upper left', fontsize=8, edgecolor='none')
    if len(scales) > 1:
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3,
                   fontsize=8, frameon=False)
    finish(fig, name)


def structure(data):
    fig = plt.figure(figsize=(7.1, 3.3), layout='constrained')
    grid = fig.add_gridspec(1, 3, width_ratios=(1.6, 1, 1))
    heat = fig.add_subplot(grid[0])
    selected = [one(data['trained'], scale=s, family=f, dose=.5)
                for s in ('14M', '70M') for f in ('A4+OL1@4', 'A7+OL1@7')]
    values = np.array([[100*r['sites'][s]['fraction'] for r in selected] for s in SITES])
    heat.imshow(values, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    for i in range(7):
        for j in range(4):
            v = values[i,j]
            heat.text(j, i, '<.01' if v < .01 else f'{v:.1f}', ha='center', va='center',
                      fontsize=8.5, color='white' if v > 65 else 'black')
    heat.set_yticks(range(7), ['a', 'm', 'h', 'q', 'k', 'v', 'z'])
    heat.set_xticks(range(4), ['14M\nA4', '14M\nA7', '70M\nA4', '70M\nA7'])
    heat.set_title('(a) Exact-zero activations (%)')
    for col, scale in enumerate(('14M', '70M'), start=1):
        ax = fig.add_subplot(grid[col])
        rows = [r for r in selected if r['scale'] == scale]
        base = np.zeros(2)
        for op, label, color in zip(OPS, OP_NAMES, OP_COLORS):
            v = np.array([100*r['counts']['per_operation'][op]['zero_product_count']/r['counts']['model_product_count'] for r in rows])
            ax.bar([0,1], v, bottom=base, color=color, width=.62, label=label)
            base += v
        for j, v in enumerate(base):
            ax.text(j, v+.7, f'{v:.2f}', ha='center', fontsize=8.5)
        ax.set_ylim(0,45)
        ax.set_xticks([0,1], ['A4', 'A7'])
        ax.set_title(f'({"b" if col==1 else "c"}) {scale}')
        ax.set_ylabel(r'Contribution to $R_{\rm model}$ (pp)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        if col == 2:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=6,
                       frameon=False, fontsize=8)
    finish(fig, '03-sites-and-operations.pdf')


def dose_changes(data):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.0), layout='constrained')
    for ax, scale in zip(axs, ('14M', '70M', '410M')):
        for fam in ('A4+OL1@4', 'A7+OL1@7'):
            rows = [r for r in data['trained'] if r['scale'] == scale and r['family'] == fam]
            base = one(rows, dose=0.)
            c, m = STYLE[fam]
            ax.plot([100*(r['R_model']-base['R_model']) for r in rows],
                    [r['loss']-base['loss'] for r in rows], color=c, marker=m, lw=1, ms=4, label=fam)
        ax.axhline(0, color='#666666', lw=.6)
        ax.set_title(scale)
        ax.set_xlabel(r'$\Delta R_{\rm model}$ (pp)')
        ax.set_ylabel(r'$\Delta$ validation loss')
        ax.set_ylim(-.6,.7)
        ax.legend(fontsize=7.5, loc='best', frameon=False)
        ax.grid(alpha=.2)
    finish(fig, '05-dose-response-by-size.pdf')


def make_figures(data):
    matched(data)
    frontiers(data, ('14M','70M'), '02-training-and-clipping.pdf')
    structure(data)
    frontiers(data, ('410M',), '04-410m-complete-frontier.pdf')
    dose_changes(data)
