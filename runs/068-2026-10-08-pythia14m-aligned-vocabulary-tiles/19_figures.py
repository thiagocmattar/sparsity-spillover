"""Plot the verified training and endpoint results without changing their reductions."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUN = Path(__file__).resolve().parent
MODES = ('dense', 'custom_no_skip', 'custom_sparse')
COLORS = ('#0072B2', '#E69F00', '#009E73')
LABELS = ('Native dense', 'Custom no-skip', 'Custom sparse')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': path.relative_to(RUN).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest}


def main():
    if not read(RUN / 'prelaunch/retrieval-training-001.json')['archive_and_members_verified']:
        raise ValueError('Verified training retrieval required')
    if read(RUN / 'inference/transfer/receipt-001.json')['status'] != 'passed':
        raise ValueError('Verified endpoint retrieval required')
    training = read(RUN / 'results/training-comparison.json')
    endpoint = read(RUN / 'inference/results/comparison.json')
    rows = sorted(training['rows'], key=lambda r: r['head_kappa'])
    comparisons = {r['id']: r for r in endpoint['rows']}
    if len(rows) != 2 or set(comparisons) != {r['id'] for r in rows}:
        raise ValueError('Complete matched two-model cohort required')
    for row in rows:
        counts = comparisons[row['id']]['head_counts']
        if counts['empty_gate_tiles'] != counts['empty_consumer_tiles']:
            raise ValueError('Gate/consumer alignment failed')
        if counts['skipped_mma'] * counts['gate_tiles'] != counts['empty_gate_tiles'] * counts['potential_mma']:
            raise ValueError('Empty-tile and MMA bypass fractions differ')
    figures = RUN / 'figures'
    observations = RUN / 'observations'
    figures.mkdir(exist_ok=True)
    observations.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    x = np.arange(len(rows))
    ticks = [f"Head κ={r['head_kappa']:g}" for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout='constrained')
    losses = [r['loss'] for r in rows]
    axes[0].bar(x, losses, color=COLORS[0], width=.55)
    axes[0].set(xticks=x, xticklabels=ticks, ylabel='Validation loss (nats / prediction)',
                ylim=(0, max(losses) * 1.18), title='Native dense evaluation after training')
    for i, loss in enumerate(losses):
        axes[0].text(i, loss + .05, f'{loss:.6f}', ha='center')
    native = [100 * r['head_tiles']['16x16']['empty'] / r['head_tiles']['16x16']['total'] for r in rows]
    bypass = [100 * comparisons[r['id']]['head_counts']['skipped_mma'] /
              comparisons[r['id']]['head_counts']['potential_mma'] for r in rows]
    axes[1].bar(x - .17, native, width=.32, color=COLORS[0], label='B32 native empty head tiles')
    axes[1].bar(x + .17, bypass, width=.32, color=COLORS[2], hatch='//', label='B1 kernel MMA bypass')
    axes[1].set(xticks=x, xticklabels=ticks, ylabel='Pooled fraction (%)', ylim=(0, 100),
                title='Aligned 16×16 gate and M16/K16 consumer')
    axes[1].legend(loc='upper left', fontsize=8)
    for i in x:
        for offset, value in ((-.17, native[i]), (.17, bypass[i])):
            axes[1].text(i + offset, value + 1, f'{value:.2f}', ha='center', fontsize=8)
    fig.suptitle('Pythia-14M: h/z κ=0, head16×16, Ph+head, matched fresh training')
    fig.savefig(figures / '01-quality-and-aligned-occupancy.pdf')
    plt.close(fig)

    boundaries = [('projection_only', 'Projection only'),
                  ('gate_cast_projection', 'Gate + cast + projection'),
                  ('full_forward', 'Complete forward')]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), layout='constrained')
    for axis, (boundary, title) in zip(axes, boundaries):
        maximum = 0.
        for mode_index, mode in enumerate(MODES):
            values, low, high = [], [], []
            for row in rows:
                comparison = comparisons[row['id']]
                value = comparison['timings'][boundary]['host_ms']['geomean_ms'][mode]
                processes = [p['timings'][boundary]['host_ms']['geomean_ms'][mode]
                             for p in comparison['processes']]
                values.append(value)
                low.append(value - min(processes))
                high.append(max(processes) - value)
                maximum = max(maximum, max(processes))
            positions = x + (mode_index - 1) * .23
            axis.bar(positions, values, width=.21, color=COLORS[mode_index],
                     hatch=('', '..', '//')[mode_index], label=LABELS[mode_index],
                     yerr=np.array([low, high]), capsize=3)
        axis.set(xticks=x, xticklabels=ticks, ylabel='Synchronized host latency (ms)',
                 ylim=(0, maximum * 1.25), title=title)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('RTX 5090 · B1/T2048 · three fresh processes/model · 64 inputs × 7 paired passes\n'
                 'Same mode colors in every panel; timing boundaries overlap and are not additive', fontsize=11)
    fig.savefig(figures / '02-dense-and-sparse-endpoint-latency.pdf')
    plt.close(fig)

    source_rows = [identity(RUN / name) for name in ('results/training-comparison.json',
                   'inference/results/comparison.json', '19_figures.py')]
    (RUN / 'results/figure-sources.json').write_text(json.dumps(source_rows, indent=2) + '\n', encoding='utf-8')
    scope = ('Both models are fresh random-initialization training, seed1234,712 updates. '
             'Only head κ varies (.1/.5); h/z8×16 κ0 and seven-site Ph+head pressure are matched. '
             'All500 validation documents yield338 complete2048-token blocks,692224 input tokens '
             'and691886 predictions;1444 tail tokens are excluded. Native training evaluation uses '
             'B32; the endpoint uses B1 on one RTX5090. The head gate is sparse activation masking; '
             'training and the native head baseline use dense matmul. All h/z projections remain '
             'native dense in the endpoint comparison. Only custom sparse head execution bypasses MMA.')
    quality = f'''# Quality and aligned head occupancy

Question: does raising the aligned head threshold create useful skippable sparsity, and at what loss?

Method and coverage: {scope}

Sources: [training reduction](../results/training-comparison.json), [endpoint reduction](../inference/results/comparison.json),
and [byte identities](../results/figure-sources.json). Integer activation/tile/instruction counts are pooled before division.
Loss is summed NLL divided by all691886 predictions, not an average of batch losses.

Caption: left, final native validation loss (zero-baseline axis). Right, empty16×16 input tiles under B32 native evaluation
and independently counted B1 custom-consumer MMA bypass. Blue denotes native dense measurements; green/hatching denotes
custom sparse execution. B1 empty gate tiles equal empty consumer tiles and skipped/potential MMA exactly.
Different B32/B1 activation arithmetic can change occupancy; these are reported separately.

Result: head κ.1 loss={losses[0]:.6f}, B32 empty tiles={native[0]:.3f}%, B1 bypass={bypass[0]:.3f}%;
head κ.5 loss={losses[1]:.6f}, B32 empty tiles={native[1]:.3f}%, B1 bypass={bypass[1]:.3f}%.
The paired higher-threshold loss delta is {losses[1]-losses[0]:+.6f}.

Caveats: one seed, model size and training budget. The two models isolate threshold at fixed alignment and pressure;
historical Run066 checkpoints do not isolate tile alignment alone. Empty tiles/logical products do not establish speed.
No acceptable-quality cutoff was declared. This observation is not a promoted manuscript claim.

Source script: [19_figures.py](../19_figures.py). Output: [PDF](../figures/01-quality-and-aligned-occupancy.pdf).
'''
    latency = f'''# Dense, no-skip and sparse vocabulary-head execution

Question: does the aligned input gate let the custom head beat native dense execution?

Method and coverage: {scope}

Each model has three fresh qualified processes. Each boundary uses64 fixed validation inputs×7 randomized paired passes,
with all three modes measured on each pair. The primary timer is synchronized host wall time; CUDA-event measurements
use separate replays. Geometric means first reduce each process, then the three processes. Error bars span the three
process geometric means and are descriptive variability, not confidence intervals over training seeds.

Sources: [endpoint reduction](../inference/results/comparison.json), [paired raw timing rows](../inference/artifacts/attempts/),
and [byte identities](../results/figure-sources.json).

Caption: three separate boundaries: projection only; head gate+cast+projection; complete model forward.
Blue=Native dense, orange/dots=Custom no-skip, green/hatching=Custom sparse, with identical colors across both models
and panels. Panels have individually labeled millisecond scales. The boundaries overlap and cannot be summed or subtracted
to form a measured decomposition. No-skip isolates bypass within the custom package; the dense comparison also includes
layout/preparation differences. Static weight packing, input staging, loss, softmax and token selection are outside timing.
All50304 logits are computed and written.

Result: the full table and process ratios are in [REPORT.md](../inference/results/REPORT.md). The figure shows absolute
latencies so sparsity and bypass can be assessed against the costs that remain.

Caveats: full-sequence B1/T2048 inference, one RTX5090, one model size and seed. These timings do not establish
cached-decoding performance or HBM traffic. All transformer h/z projection kernels stay native dense.
The two trained models have different weights; comparing their absolute latency is not a pure runtime sparsity ablation.
Within each model, all three modes use the same weights, gate and vocabulary contract.
This observation is not a promoted manuscript claim.

Source script: [19_figures.py](../19_figures.py). Output: [PDF](../figures/02-dense-and-sparse-endpoint-latency.pdf).
'''
    (observations / '01-quality-and-aligned-occupancy.md').write_text(quality, encoding='utf-8')
    (observations / '02-dense-and-sparse-endpoint-latency.md').write_text(latency, encoding='utf-8')
    (observations / 'INDEX.md').write_text('# Run068 observations\n\n'
        '- [Quality and aligned head occupancy](01-quality-and-aligned-occupancy.md): native loss, empty tiles and independent B1 MMA bypass.\n'
        '- [Dense/no-skip/sparse endpoint latency](02-dense-and-sparse-endpoint-latency.md): three overlapping boundaries with common mode colors.\n', encoding='utf-8')
    print({'status': 'plotted', 'figures': 2}, flush=True)


if __name__ == '__main__':
    main()
