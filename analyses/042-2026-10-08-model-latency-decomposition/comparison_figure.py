"""Final dense/sparse comparison using the same semantic legend at every size."""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import labels


def build(data, root):
    conditions = data['conditions']
    maximum = max(m['active_us'] for c in conditions for m in c['modes'].values()) * 1.04
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), layout='constrained')
    positions = [0, 1, 3, 4, 6, 7]
    for ax, kappa in zip(axes, ['0', '0.1']):
        points = [c for c in conditions if c['kappa'] == kappa]
        names, modes = [], []
        for c in points:
            gate = 'TILED' if c['tiled_threshold'] else 'SCALAR'
            for mode, label in [('native', 'Dense'), (c['selected_sparse'], 'Sparse')]:
                names.append(f'{c["model"]} ({gate}) — {label}')
                modes.append(c['modes'][mode])
        left = [0.] * len(modes)
        for category, style in labels.COMMON_STYLES.items():
            values = [m['semantic_categories'][category] for m in modes]
            ax.barh(positions, values, left=left, color=style['color'],
                    hatch=style['hatch'], edgecolor='white', linewidth=.3)
            left = [a+b for a, b in zip(left, values)]
        for y, m in zip(positions, modes):
            ax.text(m['active_us'] + maximum*.012, y, f'{m["active_us"]:,.0f}',
                    va='center', fontsize=8)
        ax.set_yticks(positions, names)
        ax.set_ylim(7.7, -.7)
        ax.set_xlim(0, maximum)
        ax.set_xlabel('Complete forward — CUDA activity budget (us)')
        ax.set_title(f'kappa = {kappa}')
        ax.grid(axis='x', alpha=.2)
        for y in [2, 5]:
            ax.axhline(y, color='#BBBBBB', linewidth=.6, linestyle=':')
    fig.legend(handles=[Patch(facecolor=style['color'], hatch=style['hatch'], label=category)
                        for category, style in labels.COMMON_STYLES.items()],
               loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('Dense vs sparse execution across model sizes — identical semantic colours\n'
                 'Dense = native arithmetic on the same gated checkpoint. B1 / T2048 / BF16 / RTX 5090; 14M separate session.',
                 fontsize=11)
    name = '08-dense-sparse-across-models'
    fig.savefig(root / 'figures' / (name + '.pdf'), bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    caption = ('Final direct comparison of native dense arithmetic and the selected sparse consumer at 14M, '
               '31M and 70M, for kappa 0 and 0.1. Each size has an adjacent dense/sparse pair. Colours, hatching, '
               'category order, legend and microsecond scale are identical across both panels and model sizes. '
               'The purple H/Z output block always includes gates, sparse preparation, W2, Wo and residual combination. '
               'Labels identify the 14M tiled threshold and 31M/70M scalar thresholds. Dense means native execution '
               'of that same gated checkpoint, not a different ungated model; sparse selects the corner-specialized '
               '14M consumer or the existing scalar consumers at 31M/70M. Bars sum actual CUPTI activity means '
               'from 12 profiled forwards; end labels show GPU-active microseconds. This is a component budget, '
               'not uninstrumented host latency; gaps and host overhead are outside these bars. Original timing '
               'distributions remain in the CSVs. Different gates/consumers and the separate physical 14M GPU '
               'session restrict cross-size interpretation to a descriptive comparison.')
    result = ('The common categories expose the growing 31M sparse H/Z output cost and the smaller 70M sparse '
              'H/Z output cost versus native, while keeping vocabulary projection and attention comparable.')
    return name, 'How do dense and sparse forward budgets compare across model sizes with one fixed semantic legend?', caption, result
