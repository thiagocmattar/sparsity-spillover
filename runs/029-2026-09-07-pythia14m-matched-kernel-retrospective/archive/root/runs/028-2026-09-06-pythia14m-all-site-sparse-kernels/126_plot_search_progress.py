"""Single-metric progress: matched full-model graph speedup, K031-K050."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import RUN, read_json, record, verify_record, write_json


def build_figure(data):
    points = [p for p in data['points'] if p['execution'] == 'graph']
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(6.1, 3.9))
    blue, orange = '#0072B2', '#D55E00'
    for qualified, marker, color, size in [(True, 'o', blue, 17), (False, 'x', orange, 27)]:
        group = [p for p in points if p['qualified'] == qualified]
        ax.scatter([p['iteration'] for p in group], [p['speedup'] for p in group],
                   marker=marker, color=color, s=size, linewidths=0.65,
                   alpha=0.6 if qualified else 1, zorder=3)

    trace = data['best_fixed_c30']['graph']
    ax.step([p['iteration'] for p in trace], [p['speedup'] for p in trace],
            where='post', color=orange, linewidth=1.45, zorder=2)
    ax.text(39.3, 1.535, 'Best so far (fixed checkpoint)', color=orange, fontsize=8)
    ax.annotate(f"{trace[-1]['speedup']:.3f}" + r'$\times$',
                xy=(50, trace[-1]['speedup']), xytext=(-7, 9),
                textcoords='offset points', ha='right', color=orange, fontsize=9)
    ax.set(xlim=(30.4, 50.6), ylim=(0.70, 1.86),
           xlabel='Kernel iteration', ylabel=r'Full-model speedup ($\times$)')
    ax.set_xticks([31, 35, 40, 45, 50])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(False, which='both')
    fig.text(0.5, 0.965, 'Auto-research progress', ha='center', va='top', fontsize=11)
    fig.text(0.5, 0.899, 'Pythia-14M | RTX 5090 | BF16 | B = 1, T = 2,048',
             ha='center', fontsize=7.7)
    fig.text(0.5, 0.066, 'Native graph / candidate graph. One point per model; expanded y-axis.',
             ha='center', fontsize=7)
    fig.text(0.5, 0.029, r'$\times$: numerical check failed; excluded from best-so-far line.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.225)
    return fig, ax


def main():
    data = read_json(RUN / 'results/search-progress-001.json')
    verify_record(data['script'])
    sources = {r['path']: r for p in data['points'] for r in p['sources']}
    for source in sources.values():
        verify_record(source)
    fig, ax = build_figure(data)
    output = RUN / 'figures/06-autoresearch-kernel-progress.pdf'
    fig.savefig(output, metadata={
        'Title': 'Auto-research progress: full-model graph speedup, K031-K050',
        'Creator': 'Run028/126_plot_search_progress.py',
        'CreationDate': None, 'ModDate': None,
    })
    points = [p for p in data['points'] if p['execution'] == 'graph']
    write_json(RUN / 'results/search-progress-figure-001.json', {
        'figure': record(output), 'source': record(RUN / 'results/search-progress-001.json'),
        'script': record(__file__), 'matplotlib': matplotlib.__version__,
        'metric': 'Matched full-model native graph / candidate graph paired-geomean speedup',
        'iterations': sorted({p['iteration'] for p in points}),
        'points': len(points), 'qualified_points': sum(p['qualified'] for p in points),
        'presentation': 'Single axes; exact integer x; one marker per model; no bars, boxes, grid, or legend',
        'line': 'Cumulative best fully numerically qualified c30 speedup, checkpoint and execution fixed',
        'excluded_metric': 'Eager timings remain in source inventory, not mixed into this figure',
        'x_limits': list(ax.get_xlim()), 'y_limits': list(ax.get_ylim()),
    })
    plt.close(fig)
    print(record(output))


if __name__ == '__main__':
    main()
