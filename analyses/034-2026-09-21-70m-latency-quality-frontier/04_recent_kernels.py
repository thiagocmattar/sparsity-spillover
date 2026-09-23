"""Plot retained 14M measurements with only the Run051/052 70M executions."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter

import recent_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HIGHLIGHTS = [('hz', 'h'), ('7', 'all')]


def main():
    source = HERE / 'data/combined-figure.json'
    original = recent_runs.read(source)
    assert recent_runs.sha(HERE / original['output']) == original['output_sha256']
    for name, digest in original['sources_sha256'].items():
        assert recent_runs.sha(ROOT / name) == digest, name
    small, large, palette, timing = [recent_runs.read(ROOT / name)
                                    for name in original['sources_sha256']]
    assert original['cohorts'] == {'14M': small['points'], '70M': large['points']}
    points14 = original['cohorts']['14M']
    bases14 = [r for r in original['base_references'] if r['model'] == '14M']
    base_timing, = [r for r in timing['points'] if r['family'] == 'A0']
    for reference in bases14:
        assert reference['latency_ms'] == base_timing[reference['source_field']]
    additions = recent_runs.load(original['cohorts']['70M'])
    assert len(points14) == 40 and len(bases14) == 2
    assert all(p['session'] in ('Run051', 'Run052') for p in additions['points'])
    output = HERE / 'figures/03-14m-70m-recent-kernel-latency-quality.pdf'
    preserved = {p.name: recent_runs.sha(p) for p in (HERE / 'figures').glob('*.pdf')
                 if p != output}

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelsize': 11, 'axes.titlesize': 12,
                         'xtick.labelsize': 10, 'ytick.labelsize': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .65, 'pdf.fonttype': 42,
                         'mathtext.fontset': 'dejavusans'})
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    fig.subplots_adjust(left=.095, right=.985, top=.92, bottom=.355)
    styles = sorted(palette['series'], key=lambda s: (s['scope'], s['pressure']) in HIGHLIGHTS)
    handles, drawn = {}, []
    for style in styles:
        key = (style['scope'], style['pressure'])
        if key == ('0', 'none'):
            continue
        points = sorted((r for r in points14 if (r['scope'], r['pressure']) == key),
                        key=lambda r: r.get('dose') or 0)
        if not points:
            continue
        drawn.extend(points)
        highlighted = key in HIGHLIGHTS
        color = style['color'] if highlighted else '#9DA3AB'
        ls = style['linestyle']
        if isinstance(ls, list):
            ls = (ls[0], tuple(ls[1]))
        width, alpha = (1.6, 1.) if highlighted else (.85, .65)
        marker = dict(marker='o', ms=5.8 if highlighted else 4.2, mfc=color,
                      mec='white', mew=.7 if highlighted else .5)
        x, y = [r['loss'] for r in points], [r['latency_ms'] for r in points]
        ax.plot(x, y, color=color, ls=ls, lw=width, alpha=alpha, zorder=4 if highlighted else 2)
        ax.plot(x, y, color=color, ls='none', **marker, zorder=5 if highlighted else 3)
        if highlighted:
            handles[key] = Line2D([], [], color=color, ls=ls, lw=width, **marker,
                                  label=f"14M {style['label']}")
    assert len(drawn) == len({r['checkpoint_key'] for r in drawn}) == 39
    legend14 = []
    for reference in bases14:
        kernel = reference['backend'] == 'kernel'
        marker = dict(marker='o', ms=10 if kernel else 5.5,
                      mfc='white' if kernel else '#52565C', mec='#52565C', mew=1.5 if kernel else .7)
        ax.plot(reference['loss'], reference['latency_ms'], color='#52565C', ls='none',
                **marker, zorder=6 if kernel else 7)
        legend14.append(Line2D([], [], color='#52565C', ls='none', **marker,
                               label=f"14M Base ({reference['backend']})"))
    legend14.extend(handles[key] for key in HIGHLIGHTS)
    colors = {'T2/Ph': next(s['color'] for s in styles if (s['scope'], s['pressure']) == ('hz', 'h')),
              'T7/Pall': next(s['color'] for s in styles if (s['scope'], s['pressure']) == ('7', 'all'))}
    legend70 = recent_runs.draw(ax, additions, colors)
    for size, loss in [('14M', bases14[0]['loss']),
                       ('70M', next(r['loss'] for r in additions['points'] if r['recipe'] == 'Base'))]:
        ax.axvline(loss, color='#92969B', lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.text(loss + .025, .98, f'{size} Base loss', transform=ax.get_xaxis_transform(),
                color='#747A81', fontsize=8.5, ha='left', va='top')
    all_points = drawn + bases14 + additions['points']
    xlim, ylim = (4.02, 6.10), (.40, 1.85)
    assert len(all_points) == 59
    assert all(xlim[0] <= r['loss'] <= xlim[1] and ylim[0] <= r['latency_ms'] <= ylim[1]
               for r in all_points)
    assert all(line.get_marker() != 's' for line in ax.lines)
    ax.set(xlim=xlim, ylim=ylim, xscale='log', yscale='log',
           xlabel='Validation loss', ylabel='Full-model latency (ms)')
    ax.set_xticks([4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0])
    ax.set_yticks([.5, 1., 1.5])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_title('Pythia-14M and Pythia-70M', loc='left', pad=12)
    ax.tick_params(length=3, width=.65)
    ax.grid(axis='y', color='#E8EAED', lw=.6)
    ax.set_axisbelow(True)
    old_legend = fig.legend(handles=legend14, loc='lower center', ncol=4, frameon=False,
                            fontsize=8.5, handlelength=2., columnspacing=1.4,
                            bbox_to_anchor=(.54, .205), title='14M: retained K050 and PyTorch measurements',
                            title_fontsize=9)
    new_legend = fig.legend(handles=legend70, loc='lower center', ncol=3, frameon=False,
                            fontsize=8.5, handlelength=2., columnspacing=2., labelspacing=.7,
                            bbox_to_anchor=(.54, .055),
                            title='70M: Run051/052 only; C = sparse h in five layers, dense z', title_fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not ax.xaxis.label.get_window_extent(renderer).overlaps(old_legend.get_window_extent(renderer))
    assert not old_legend.get_window_extent(renderer).overlaps(new_legend.get_window_extent(renderer))
    fig.savefig(output, bbox_inches='tight', pad_inches=.04,
                metadata={'Title': 'Pythia-14M and Pythia-70M: recent kernel loss-latency measurements',
                          'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    assert all(recent_runs.sha(HERE / 'figures' / name) == digest for name, digest in preserved.items())
    result = dict(output=output.relative_to(HERE).as_posix(), output_sha256=recent_runs.sha(output),
                  script=Path(__file__).name, script_sha256=recent_runs.sha(Path(__file__)),
                  original_data_sha256=recent_runs.sha(source), sources_sha256=original['sources_sha256'],
                  points14=points14, base_references14=bases14, recent_70m=additions,
                  x_metric='ordinary_FP16_validation_loss', y_unit='milliseconds',
                  axis_scales={'x': 'log', 'y': 'log'}, xlim=xlim, ylim=ylim,
                  legend=[h.get_label() for h in legend14 + legend70],
                  verification=dict(displayed_14m_executions=41, displayed_70m_executions=18,
                                    total_displayed=59, qualified=57, unqualified=2,
                                    historical_70m_executions=0, square_markers=0,
                                    unchanged_previous_pdfs=preserved),
                  interpretation='Retained 14M measurements and only Run051/052 70M data. Separate absolute sessions; failed numerical points marked; no inferred component-to-full-model latency.')
    (HERE / 'data/recent-kernel-figure.json').write_text(json.dumps(result, indent=2) + '\n',
                                                       encoding='utf-8', newline='\n')
    print(json.dumps({'pdf': str(output), **result['verification']}))


if __name__ == '__main__':
    main()
