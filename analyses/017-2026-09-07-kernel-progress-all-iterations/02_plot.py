"""Single-panel historical view with visible hardware/execution boundaries."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('analysis017_reduce', HERE / '01_reduce.py')
reduce = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reduce)


def build_figure(data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 10,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.7, 'pdf.fonttype': 42,
    })
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    blue, orange = '#0072B2', '#D55E00'
    for qualified, marker, color, size in [(True, 'o', blue, 15), (False, 'x', orange, 20)]:
        group = [p for p in data['points'] if p['qualified'] == qualified]
        ax.scatter([p['iteration'] for p in group], [p['speedup'] for p in group],
                   marker=marker, color=color, s=size, linewidths=0.65,
                   alpha=0.58 if qualified else 0.75, zorder=3)
    for phase in reduce.PHASES:
        trace = data['best_fixed_checkpoint_by_phase'][phase]
        ax.step([p['iteration'] for p in trace], [p['speedup'] for p in trace],
                where='post', color=orange, linewidth=1.5, zorder=2)
    for boundary in [16.5, 30.5]:
        ax.axvline(boundary, color='#777777', linestyle=(0, (4, 4)), linewidth=0.8, zorder=1)
    for x, label in [(8.5, 'RTX PRO 4500\nEager'), (23.5, 'RTX 5090\nEager'),
                     (40.5, 'RTX 5090\nCUDA graphs')]:
        ax.text(x, 1.035, label, transform=ax.get_xaxis_transform(), ha='center',
                va='bottom', fontsize=8.3, linespacing=1.5)
    ax.text(34, 2.13, 'Best so far (fixed checkpoint;\nrestarted at protocol changes)',
            fontsize=8, color=orange, linespacing=1.5)
    ax.annotate(r'1.739$\times$', xy=(50, data['best_fixed_checkpoint_by_phase']['5090_graph'][-1]['speedup']),
                xytext=(-5, 11), textcoords='offset points', ha='right', fontsize=8, color=orange)
    ax.set(xlim=(0.4, 50.8), ylim=(0.1, 2.52), xlabel='Kernel iteration',
           ylabel=r'Matched full-model speedup ($\times$)')
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_axisbelow(True)
    ax.grid(True, which='major', color='#DEDEDE', linewidth=0.55)
    fig.text(0.5, 0.975, 'Auto-research progress, K001-K050', ha='center', va='top', fontsize=11)
    fig.text(0.5, 0.900, 'Pythia-14M | BF16 | B = 1, T = 2,048 | Native / candidate within each phase',
             ha='center', fontsize=7.7)
    fig.text(0.5, 0.075, f"{len(data['points'])} model/setting measurements; " + r'$\times$ marks numerical-check failures.',
             ha='center', fontsize=7)
    fig.text(0.5, 0.038, 'Blank iterations lack 14M full-model timing. GPU and execution change at dashed boundaries.',
             ha='center', fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.775, bottom=0.215)
    return fig, ax


def main():
    data_path = HERE / 'results/progress.json'
    data = reduce.read(data_path)
    reduce.verify(data['script'])
    records = {r['path']: r for p in data['points'] for r in p['sources']}
    for source in records.values():
        reduce.verify(source)
    fig, ax = build_figure(data)
    output = HERE / 'figures/01-kernel-progress-all-iterations.pdf'
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, metadata={
        'Title': 'Pythia-14M autoresearch history, K001-K050, with protocol boundaries',
        'Creator': 'Analysis017/02_plot.py', 'CreationDate': None, 'ModDate': None,
    })
    reduce.write(HERE / 'results/figure-provenance.json', {
        'figure': reduce.record(output), 'source': reduce.record(data_path),
        'script': reduce.record(__file__), 'matplotlib': matplotlib.__version__,
        'points': len(data['points']), 'missing_iterations': data['missing_iterations'],
        'x_limits': list(ax.get_xlim()), 'y_limits': list(ax.get_ylim()),
        'presentation': 'One axes, individual points, light grid, no bars/boxes/legend; separate incumbent per hardware/execution phase',
    })
    plt.close(fig)
    print(reduce.record(output))


if __name__ == '__main__':
    main()
