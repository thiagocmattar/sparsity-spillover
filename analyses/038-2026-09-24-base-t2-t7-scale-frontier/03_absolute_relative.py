"""Pair absolute measurements with T2/Ph loss and latency change boxplots."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import NullLocator, ScalarFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TITLE = 'Targeted sparsification exhibits scale-dependent quality\u2013latency trade-offs'
MARKERS = {'14M': 'o', '31M': '^', '70M': 's'}
COLORS = {'hz': '#173FAD', '7': '#C27539'}
LINES = {'hz': (0, (3, 2)), '7': '-'}
LOSS_COLOR = '#8A669C'
TIME_COLOR = '#007F73'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE/'data/scale-figure.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    original_pdf = HERE/data['output']
    assert sha(original_pdf) == data['output_sha256']
    for name, digest in data['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    points = data['executions']
    assert len(points) == 36 and len({p['checkpoint_key'] for p in points}) == 33
    bases = {}
    for scale in MARKERS:
        native, = [p for p in points if p['model'] == scale and p['scope'] == '0' and p['backend'] == 'PyTorch']
        kernel, = [p for p in points if p['model'] == scale and p['scope'] == '0' and p['backend'] == 'kernel']
        assert native['checkpoint_key'] == kernel['checkpoint_key'] and native['loss'] == kernel['loss']
        assert native['latency_ms'] > 0
        bases[scale] = dict(loss=native['loss'],pytorch_latency_ms=native['latency_ms'],
            checkpoint_key=native['checkpoint_key'],timing_session=native['timing_session'])
        for scope in COLORS:
            assert sorted(p['kappa'] for p in points if p['model'] == scale and p['scope'] == scope) == [0., .01, .05, .1, .5]
    changes = []
    for point in points:
        if point['scope'] != 'hz' or point['pressure'] != 'h':
            continue
        base = bases[point['model']]
        delta = point['loss'] - base['loss']
        time_delta = point['latency_ms'] - base['pytorch_latency_ms']
        assert math.isfinite(delta) and math.isfinite(time_delta)
        assert math.isclose(delta + base['loss'], point['loss'], abs_tol=1e-12)
        assert math.isclose(time_delta + base['pytorch_latency_ms'], point['latency_ms'], abs_tol=1e-12)
        changes.append(dict(point, loss_delta=delta, latency_delta_ms=time_delta))
    assert len(changes) == 15
    boxes = {}
    for scale in MARKERS:
        rows = sorted((p for p in changes if p['model'] == scale), key=lambda p:p['kappa'])
        assert [p['kappa'] for p in rows] == [0., .01, .05, .1, .5]
        boxes[scale] = {}
        for metric in ('loss_delta', 'latency_delta_ms'):
            values = [p[metric] for p in rows]
            stats, = boxplot_stats(values, whis=1.5)
            boxes[scale][metric] = dict(n=5, kappas=[p['kappa'] for p in rows], values=values,
                **{key:float(stats[key]) for key in ('q1', 'med', 'q3', 'whislo', 'whishi')},
                fliers=stats['fliers'].tolist())

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.labelsize': 11, 'axes.titlesize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'pdf.fonttype': 42, 'mathtext.fontset': 'dejavusans'})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.3))
    fig.subplots_adjust(left=.08, right=.985, bottom=.24, top=.84, wspace=.32)
    fig.suptitle(TITLE, x=.08, y=.965, ha='left', fontsize=12)
    panels = [dict(rows=points, x='loss', y='latency_ms', title='(a) Absolute',
        xlim=(4.02, 5.86), ylim=(.43, 2.85))]
    for ax, panel in zip(axes, panels):
        rows, x, y = panel['rows'], panel['x'], panel['y']
        for scale, marker in MARKERS.items():
            for scope in COLORS:
                curve = sorted((p for p in rows if p['model'] == scale and p['scope'] == scope), key=lambda p:p['kappa'])
                ax.plot([p[x] for p in curve], [p[y] for p in curve], color=COLORS[scope],
                    ls=LINES[scope], lw=1.6, marker=marker, ms=5.8, mec='white', mew=.7, zorder=4)
        refs = [p for p in rows if p['scope'] == '0']
        for point in refs:
            kernel = point['backend'] == 'kernel'
            ax.plot(point[x], point[y], marker=MARKERS[point['model']], ls='none',
                ms=10 if kernel else 5.5, mfc='white' if kernel else '#52565C',
                mec='#52565C', mew=1.5 if kernel else .7, zorder=6 if kernel else 7)
        for base in bases.values():
            ax.axvline(base['loss'], color='#92969B', lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.set(xscale='log', yscale='log', xlabel='Validation loss', ylabel='Full-model latency (ms)')
        ax.set_xticks([4.2, 4.5, 4.8, 5.1, 5.4, 5.7])
        ax.set_yticks([.5, 1., 1.5, 2., 2.5])
        ax.set(xlim=panel['xlim'], ylim=panel['ylim'])
        assert all(panel['xlim'][0] <= p[x] <= panel['xlim'][1] and panel['ylim'][0] <= p[y] <= panel['ylim'][1] for p in rows)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_locator(NullLocator())
        ax.set_title(panel['title'], loc='left', pad=10)
        ax.grid(axis='y', color='#E8EAED', lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=.65)
    loss_ax = axes[1]
    position = loss_ax.get_position()
    loss_ax.set_position([position.x0, position.y0, position.width-.06, position.height])
    time_ax = loss_ax.twinx()
    for ax, metric, color, offset in [(loss_ax, 'loss_delta', LOSS_COLOR, -.17),
            (time_ax, 'latency_delta_ms', TIME_COLOR, .17)]:
        stats = [boxes[scale][metric] for scale in MARKERS]
        ax.bxp(stats, positions=[i+offset for i in range(3)], widths=.27,
            patch_artist=True, manage_ticks=False, showfliers=True, zorder=4,
            boxprops=dict(facecolor=matplotlib.colors.to_rgba(color, .22), edgecolor=color, linewidth=1.2),
            medianprops=dict(color=color, linewidth=1.5), whiskerprops=dict(color=color, linewidth=1.1),
            capprops=dict(color=color, linewidth=1.1),
            flierprops=dict(marker='o', markersize=4.5, markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.))
        ax.tick_params(axis='y', colors=color, length=3, width=.65)
        ax.yaxis.set_minor_locator(NullLocator())
    # Different units, with zero aligned at one quarter of each axis height.
    loss_ax.set(ylim=(-.30, .90), xlim=(-.6, 2.6), xlabel='Model size',
        ylabel=r'Loss change, $L-L_{\mathrm{Base}}$')
    time_ax.set(ylim=(-.45, 1.35), ylabel=r'Latency change, $t-t_{\mathrm{PyTorch\,Base}}$ (ms)')
    loss_ax.set_xticks([0, 1, 2], list(MARKERS))
    loss_ax.set_yticks([-.2, 0, .2, .4, .6, .8])
    time_ax.set_yticks([-.3, 0, .3, .6, .9, 1.2])
    loss_ax.yaxis.label.set_color(LOSS_COLOR)
    time_ax.yaxis.label.set_color(TIME_COLOR)
    loss_ax.spines['left'].set_color(LOSS_COLOR)
    time_ax.spines['left'].set_visible(False)
    time_ax.spines['right'].set_visible(True)
    time_ax.spines['right'].set_color(TIME_COLOR)
    loss_ax.axhline(0, color='#92969B', lw=.8, ls=(0, (2, 3)), zorder=2)
    loss_ax.grid(axis='y', color='#E8EAED', lw=.6)
    loss_ax.set_axisbelow(True)
    loss_ax.tick_params(axis='x', length=3, width=.65)
    loss_ax.set_title(r'(b) $T_2/P_h$: changes from Base', loc='left', pad=10)
    loss_ax.legend(handles=[Patch(facecolor=matplotlib.colors.to_rgba(c, .22), edgecolor=c, label=label)
        for c, label in [(LOSS_COLOR, 'Loss change'), (TIME_COLOR, 'Latency change')]],
        loc='upper left', frameon=False, fontsize=8.5, handlelength=1.3)
    assert math.isclose(-loss_ax.get_ylim()[0]/(loss_ax.get_ylim()[1]-loss_ax.get_ylim()[0]),
        -time_ax.get_ylim()[0]/(time_ax.get_ylim()[1]-time_ax.get_ylim()[0]))
    for row in changes:
        assert loss_ax.get_ylim()[0] < row['loss_delta'] < loss_ax.get_ylim()[1]
        assert time_ax.get_ylim()[0] < row['latency_delta_ms'] < time_ax.get_ylim()[1]
    families = [Line2D([], [], marker='o', ls='none', ms=10, mfc='white', mec='#52565C', mew=1.5, label='Base (kernel)'),
        Line2D([], [], marker='o', ls='none', ms=5.5, color='#52565C', label='Base (PyTorch)'),
        *[Line2D([], [], color=COLORS[s], ls=LINES[s], lw=1.6, label=label) for s, label in [('hz', r'$T_2/P_h$'), ('7', r'$T_7/P_h$')]]]
    sizes = [Line2D([], [], marker=m, ls='none', color='#52565C', ms=5.8, label=s) for s, m in MARKERS.items()]
    fig.legend(handles=families, loc='lower center', bbox_to_anchor=(.53, .080), ncol=4,
        frameon=False, fontsize=9.5, handlelength=2., columnspacing=1.4)
    fig.legend(handles=sizes, loc='lower center', bbox_to_anchor=(.53, .018), ncol=3,
        frameon=False, fontsize=9.5, columnspacing=2.2)
    output = HERE/'figures/02-absolute-relative-scale-frontier.pdf'
    fig.savefig(output, bbox_inches='tight', pad_inches=.04,
        metadata={'Title': TITLE, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert sha(original_pdf) == data['output_sha256']
    result = dict(source=source.relative_to(HERE).as_posix(), source_sha256=sha(source),
        preserved_absolute_pdf_sha256=data['output_sha256'], script=Path(__file__).name,
        script_sha256=sha(Path(__file__)), output=output.relative_to(HERE).as_posix(), output_sha256=sha(output),
        counts=dict(absolute_execution_points=36, absolute_checkpoints=33, boxplot_checkpoints=15,
            boxes=6, observations_per_box=5), bases=bases, absolute_points=points, points=changes, boxplots=boxes,
        formulas=dict(x='model size', left_y='loss - same-scale Base loss',
            right_y='latency_ms - same-scale PyTorch Base latency_ms'),
        boxplot_method=dict(quartiles='linear interpolation', whiskers='most extreme observations within 1.5 IQR',
            show_fliers=True, sampling_unit='one trained endpoint per kappa; not replicate uncertainty'),
        axis_scales=dict(absolute=dict(x='log', y='log'), boxplot=dict(x='categorical', left_y='linear', right_y='linear')),
        interpretation='Panel (b) includes only T2/Ph. Each box summarizes the five kappa settings. Loss and latency use separate zero-aligned axes; latency is an absolute difference in milliseconds. No fitted scaling law.')
    (HERE/'data/absolute-relative-figure.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(pdf=str(output), boxes=6, observations_per_box=5,
        medians={scale:{metric:stats['med'] for metric,stats in rows.items()} for scale,rows in boxes.items()})))


if __name__ == '__main__':
    main()
