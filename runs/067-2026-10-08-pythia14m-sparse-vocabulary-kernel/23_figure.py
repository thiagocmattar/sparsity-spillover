"""Final process-aware layout; preserve the sealed prelaunch plot source unchanged."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from io_utils import RUN, read, record, write


def main():
    rows = read(RUN/'results/comparison.json')['rows']
    modes = ['dense', 'custom_no_skip', 'custom_sparse']
    labels = ['Dense cuBLAS', 'Custom no-skip', 'Custom sparse']
    hatches = ['', '//', 'xx']
    boundaries = ['projection_only', 'gate_cast_projection', 'full_forward']
    titles = ['Projection only', 'Gate + cast + projection', 'Complete forward']
    colors = ['#D55E00', '#D55E00', '#444444']
    fig = plt.figure(figsize=(13, 4.9), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[.23, 1])
    header = fig.add_subplot(grid[0, :]); header.axis('off')
    header.text(.5, .82, '14M frozen vocabulary-head checkpoints | RTX 5090 | B1/T2048 | FP16 projection',
                ha='center', va='center', fontsize=13)
    handles = [Patch(facecolor='white', edgecolor='black', hatch=h, label=l)
               for h, l in zip(hatches, labels)]
    handles.append(Line2D([], [], color='black', marker='o', linestyle='', markersize=4,
                         label='Dots: 3 process means'))
    header.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .02),
                  ncol=4, frameon=False, fontsize=10)
    axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
    x = np.arange(len(rows)); width = .24
    for ax, boundary, title, color in zip(axes, boundaries, titles, colors):
        for offset, (mode, hatch) in enumerate(zip(modes, hatches)):
            positions = x+(offset-1)*width
            values = [r['timings'][boundary]['host_ms']['geomean_ms'][mode]*1000 for r in rows]
            ax.bar(positions, values, width, facecolor=color, edgecolor='black', hatch=hatch, linewidth=.6)
            for i, row in enumerate(rows):
                process = [p['timings'][boundary]['host_ms']['geomean_ms'][mode]*1000
                           for p in row['processes']]
                ax.scatter(positions[i]+np.linspace(-.035, .035, 3), process,
                           s=13, facecolor='black', edgecolor='white', linewidth=.45, zorder=3)
        ax.set_title(title); ax.set_ylabel('Synchronized host latency (µs)')
        ax.set_xticks(x, [f'κ={r["kappa"]:g}\n'+('Ph+head' if len(r['pressure_sites']) == 2 else 'Ph')
                         for r in rows])
        ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    head_limit = 1.1*max(r['timings'][b]['host_ms']['geomean_ms'][m]*1000
                        for r in rows for b in boundaries[:2] for m in modes)
    for ax in axes[:2]: ax.set_ylim(0, head_limit)
    output = RUN/'figures/01-sparse-vocabulary-head.pdf'
    fig.savefig(output, bbox_inches='tight'); plt.close(fig)
    observation = RUN/'observations/01-sparse-vocabulary-head.md'
    text = observation.read_text(encoding='utf-8').replace('[21_figure.py](../21_figure.py)',
                                                         '[23_figure.py](../23_figure.py)')
    text = text.replace('Every output vocabulary entry is computed.',
                        'Black dots show each of the three process geometric means, with horizontal offsets '
                        'for visibility. The two head panels share a y-axis scale; complete forward has its '
                        'own scale. Every output vocabulary entry is computed.')
    text = text.replace('The separately measured panels have independent y-axis scales',
                        'The separately measured full-forward panel has its own y-axis scale')
    observation.write_text(text, encoding='utf-8', newline='\n')
    write(RUN/'results/figure-provenance.json', {'input':record(RUN/'results/comparison.json'),
          'script':record(RUN/'23_figure.py'), 'pdf':record(output),
          'aggregation':'paired geometric means; dots are the three process geometric means'})


if __name__ == '__main__':
    main()
