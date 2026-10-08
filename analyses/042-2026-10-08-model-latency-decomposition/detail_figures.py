"""Show only measured children of the common H/Z output category."""
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import labels


def build(data, root, mode_order, plot_labels):
    conditions = data['conditions']
    maximum = max(m['semantic_categories']['H/Z output block']
                  for c in conditions for m in c['modes'].values()) * 1.08
    color = labels.COMMON_STYLES['H/Z output block']['color']
    entries, rows = [], []
    for number, size in enumerate(['14M', '31M', '70M'], 5):
        cs = [c for c in conditions if c['model'] == size]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), layout='constrained')
        for ax, c in zip(axes, cs):
            names = mode_order(c)
            details = {mode: labels.detail_values(c['modes'][mode]['categories']) for mode in names}
            for mode, parts in details.items():
                expected = c['modes'][mode]['semantic_categories']['H/Z output block']
                assert abs(sum(parts.values()) - expected) < .001
                if c['model'] in {'14M', '31M'} and mode != 'native':
                    assert list(parts) == ['Fused H/Z output block (indivisible)']
                rows.extend({'condition': c['key'], 'model': size, 'mode': mode,
                             'detail_category': name, 'cupti_mean_us': value,
                             'parent_category': 'H/Z output block', 'parent_color': color}
                            for name, value in parts.items())
            left = [0.] * len(names)
            for name, hatch in labels.DETAIL_HATCHES.items():
                values = [details[mode].get(name, 0.) for mode in names]
                ax.barh(range(len(names)), values, left=left, color=color, hatch=hatch,
                        edgecolor='white', linewidth=.4)
                left = [a+b for a, b in zip(left, values)]
            for y, value in enumerate(left):
                ax.text(value + maximum*.012, y, f'{value:.1f}', va='center', fontsize=8)
            ax.set_yticks(range(len(names)), [plot_labels[name] for name in names])
            ax.invert_yaxis()
            ax.set_xlim(0, maximum)
            ax.set_xlabel('H/Z output block — summed CUDA activities (us)')
            ax.set_title(f'kappa = {c["kappa"]}')
            ax.grid(axis='x', alpha=.2)
        fig.legend(handles=[Patch(facecolor=color, hatch=hatch, label=name)
                            for name, hatch in labels.DETAIL_HATCHES.items()],
                   loc='outside lower center', ncol=3, fontsize=8)
        gate = 'TILED 8 x 16 threshold' if size == '14M' else 'SCALAR threshold (not tiled)'
        fig.suptitle(f'{size} — measured detail of H/Z output block — {gate}')
        name = f'{number:02d}-{size.lower()}-hz-detail'
        fig.savefig(root / 'figures' / (name + '.pdf'), bbox_inches='tight', pad_inches=.18)
        plt.close(fig)
        caption = (f'{size}, kappa 0/0.1. Every bar is the same H/Z output block used in the common decomposition. '
                   'All components keep the parent purple color; fixed patterns identify H gate/preparation, '
                   'H down (W2), Z gate + down (Wo), residual, or an indivisible fused H/Z consumer. '
                   'H gate/preparation sums separately measured H gate and feature compaction. '
                   'Z gate + down sums the measured Z gate/projection or their fused equivalent. '
                   'The solid fused category contains the complete measured consumer, including prepass and fallback '
                   'where present. Its internal components are not assigned separate latencies. Missing components '
                   'have no independent measurement, not a demonstrated zero cost. The same complete legend and '
                   'microsecond axis are used in all three detail figures. All values are arithmetic sums of actual '
                   'CUPTI activity means; these accounting groups are not independent contiguous timed intervals.')
        result = ('Native components are separated, while the optimized 14M/31M H/Z consumer remains indivisible.'
                  if size in {'14M', '31M'} else
                  'H down (W2) dominates the skip-disabled H/Z output block; it is a component of the common purple category. All Z products stay dense.')
        entries.append((name, f'Which independently measured components form the {size} H/Z output block?', caption, result))
    with (root / 'results/hz-detail.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return entries
