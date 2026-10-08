"""Explain the existing output path and hypothetical feature-block skipping."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import labels

ROOT = Path(__file__).resolve().parent
NAME = '09-output-head-dataflow'
QUESTION = 'Where are vocabulary scores computed, and which sparsity could reduce that work?'
CAPTION = ('The existing forward ends at full vocabulary logits Y, after final LayerNorm and the dense LM head. '
           'Greedy selection and probability-based sampling are separate downstream alternatives. '
           'A hypothetical gate after final normalization could supply sparse head inputs; zero feature blocks '
           'allow a matching weight slice to be skipped, while all vocabulary scores are still computed. '
           'The small matrix drawing is illustrative, not measured.')


def build(root=ROOT):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    dark, gray, pale = '#222222', '#666666', '#F3F4F5'
    head = labels.COMMON_STYLES['Vocabulary projection']['color']
    norm = labels.COMMON_STYLES['Final normalization']['color']
    green = '#D7ECD6'

    def text(x, y, value, size=12, **kwargs):
        return ax.text(x, y, value, fontsize=size, color=dark, va='center', **kwargs)

    def box(x, y, w, h, value, color=pale, ink=dark, dashed=False, size=12):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.005',
                                   facecolor=color, edgecolor=gray, linewidth=1,
                                   linestyle='--' if dashed else '-'))
        ax.text(x+w/2, y+h/2, value, fontsize=size, color=ink,
                va='center', ha='center', linespacing=1.4)

    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                    color=gray, linewidth=1.3,
                                    linestyle='--' if dashed else '-'))

    text(.025, .967, 'Vocabulary head: the projection already computes the scores', 21, weight='bold')
    text(.025, .929, 'Existing B=1, T=2,048 forward; V=50,304; d=128 / 256 / 512. Conceptual diagram, not a new benchmark.', 12)
    text(.025, .887, 'A. Current output path', 15, weight='bold')
    boxes = [(.025, .805, .145, .062, 'Last block\nresidual R_L [T,d]', pale, dark),
             (.205, .805, .105, .062, 'Final\nLayerNorm', norm, 'white'),
             (.345, .805, .125, .062, 'Head input\nX [T,d]', pale, dark),
             (.505, .805, .195, .062, 'Vocabulary projection\nY = X @ W_out.T', head, 'white'),
             (.755, .805, .220, .062, 'Y [T,V] = logits\nRaw vocabulary scores', pale, dark)]
    for x, y, w, h, value, color, ink in boxes:
        box(x, y, w, h, value, color, ink)
    for left, right in zip(boxes, boxes[1:]):
        arrow((left[0]+left[2]+.006, .836), (right[0]-.008, .836))
    box(.505, .699, .195, .059, 'W_out [V,d]\nDense learned weights', size=12)
    arrow((.6025, .764), (.6025, .798))
    text(.025, .761, 'Each score is one dot product:', 13, weight='bold')
    text(.025, .723, r'$Y_{t,v}=\sum_{j=1}^{d} X_{t,j}\,W_{v,j}$', 19)
    text(.025, .682, 'There is no additional Y -> scores dot product.', 12)
    ax.plot([.744, .989], [.786, .786], color=head, linewidth=2)
    text(.865, .767, 'Timed forward ends at Y', 11, ha='center', weight='bold')
    box(.755, .671, .220, .054, 'For the next generated token:\nselect the final position row', dashed=True, size=11)
    arrow((.865, .753), (.865, .732), dashed=True)
    box(.450, .562, .230, .063, 'Greedy: argmax over V\nSoftmax is unnecessary', dashed=True)
    box(.745, .562, .230, .063, 'Sampling: softmax over V\nConvert logits to probabilities', dashed=True, size=11)
    arrow((.804, .664), (.565, .632), dashed=True)
    arrow((.887, .664), (.860, .632), dashed=True)
    box(.450, .466, .230, .060, 'Selected token ID\nTokenizer decode', dashed=True)
    box(.745, .466, .230, .060, 'Sample token ID\nTokenizer decode', dashed=True)
    arrow((.565, .555), (.565, .533), dashed=True)
    arrow((.860, .555), (.860, .533), dashed=True)
    text(.025, .609, 'Other consumer: validation loss', 12, weight='bold')
    text(.025, .557, 'Position logits + shifted token labels\nfeed cross-entropy (log-softmax + loss).\nAlso outside the timed forward.', 11, linespacing=1.5)
    text(.025, .471, 'Solid = current measured path.\nDashed = downstream use or hypothetical intervention.\nHead/norm colors match Figure 08.', 10.5, linespacing=1.5)
    ax.plot([.025, .975], [.417, .417], color='#BBBBBB', linewidth=.8)

    text(.025, .389, 'B. Hypothetical sparse input: reduce features K, keep vocabulary N', 15, weight='bold')
    text(.025, .357, 'Existing h/z gates do not gate X. A head-input gate would act after LayerNorm and require quality evaluation.', 11.5)
    sparse_boxes = [(.025, .279, .125, .058, 'Final LayerNorm', norm, 'white'),
                    (.190, .279, .195, .058, 'Optional new gate\nCreates exact-zero blocks', pale, dark),
                    (.425, .279, .130, .058, 'Sparse X\n[T,d]', green, dark),
                    (.595, .279, .205, .058, 'Skip zero K blocks +\nmatching weight slices', head, 'white'),
                    (.840, .279, .130, .058, 'Dense Y\n[T,V]', pale, dark)]
    for x, y, w, h, value, color, ink in sparse_boxes:
        box(x, y, w, h, value, color, ink, dashed=True, size=11)
    for left, right in zip(sparse_boxes, sparse_boxes[1:]):
        arrow((left[0]+left[2]+.006, .308), (right[0]-.008, .308), dashed=True)

    def matrix(x, y, w, h, rows, cols, inactive, axis):
        for row in range(rows):
            for col in range(cols):
                zero = (col if axis == 'columns' else row) in inactive
                ax.add_patch(Rectangle((x+col*w/cols, y+(rows-1-row)*h/rows),
                                       w/cols, h/rows, facecolor='#E2E2E2' if zero else green,
                                       edgecolor='white', linewidth=1, hatch='///' if zero else None))
                if zero and axis == 'columns':
                    text(x+(col+.5)*w/cols, y+(rows-row-.5)*h/rows, '0', 10, ha='center')
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=gray, linewidth=.8))

    text(.025, .244, 'Toy matrices: gray = skipped feature blocks (not measured).', 11)
    matrix(.055, .093, .290, .108, 4, 8, {2, 3, 6, 7}, 'columns')
    matrix(.440, .073, .155, .149, 8, 6, {2, 3, 6, 7}, 'rows')
    matrix(.700, .093, .240, .108, 4, 6, set(), 'columns')
    text(.200, .218, 'Sparse X [4,8]', 11, ha='center', weight='bold')
    text(.518, .239, 'W_out.T [8,6]', 11, ha='center', weight='bold')
    text(.820, .218, 'Y [4,6]: all scores', 11, ha='center', weight='bold')
    text(.390, .147, r'$\times$', 24, ha='center')
    text(.647, .147, '=', 24, ha='center')
    text(.200, .065, 'Shared zero features across a token group', 10, ha='center')
    text(.820, .065, 'Sparse input need not produce sparse logits', 10, ha='center')
    text(.025, .029, 'No gate/head changes were implemented. Smaller logical work does not establish speedup; full logits still require output storage.', 10.5)
    output = root / 'figures' / (NAME + '.pdf')
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, metadata={'Title': QUESTION, 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return NAME, QUESTION, CAPTION, 'Conceptual explanation only; no new timing or head-input sparsity measurement.'


if __name__ == '__main__':
    build()
    print('Created figures/' + NAME + '.pdf')
