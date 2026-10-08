"""Consolidate verified training quality, occupancy and paired endpoint latency."""
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def reduction(ratio):
    return 100 * (1 - 1 / ratio)


def main():
    if not read('prelaunch/retrieval-training-001.json')['archive_and_members_verified']:
        raise ValueError('Verified training retrieval required')
    if read('inference/transfer/receipt-001.json')['status'] != 'passed':
        raise ValueError('Verified endpoint retrieval required')
    training = read('results/training-comparison.json')
    rows = sorted(training['rows'], key=lambda r: r['head_kappa'])
    endpoint = {r['id']: r for r in read('inference/results/comparison.json')['rows']}
    if len(rows) != 2 or set(endpoint) != {r['id'] for r in rows}:
        raise ValueError('Complete two-model cohort required')
    lines = ['# Run068 — aligned sparse vocabulary head', '',
        'Two fresh Pythia-14M models, head κ={0.1,0.5}, head16×16 gate, h/z8×16 κ=0.',
        'Ph+head is OL1 pressure at six h tensors plus one head tensor, each weighted1/7; z is gated but has no direct pressure.',
        'Initialization, seed1234, training order,712 updates and optimizer are matched.', '',
        'The head gate masks the FP32 output of the learned affine final LayerNorm, before the vocabulary matrix multiplication.',
        'A top-left value≤κ zeros its entire16-token×16-feature tile; live tiles retain signed interiors.',
        'Weights and all50304 output logits remain dense. Training uses native dense matmul.',
        'At inference, all h/z projections remain native dense. Only the custom sparse head bypasses MMA.', '',
        '## Quality and occupancy', '',
        'Both evaluations cover all500 MiniPile validation documents:338 complete2048-token blocks,692224 input tokens,',
        '691886 predictions and1444 excluded tail tokens. Counts are pooled before division.',
        'B32 native training evaluation and B1 inference counts are separate measurements.', '',
        '| Head κ | Native loss (B32) | Perplexity | Head exact zeros (B32) | Empty16×16 tiles (B32) | Consumer/MMA bypass (B1) |',
        '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        counts = endpoint[row['id']]['head_counts']
        tiles = row['head_tiles']['16x16']
        if counts['empty_gate_tiles'] != counts['empty_consumer_tiles']:
            raise ValueError('Gate and consumer occupancy differ')
        if counts['skipped_mma'] * counts['gate_tiles'] != counts['empty_gate_tiles'] * counts['potential_mma']:
            raise ValueError('Gate and MMA bypass fractions differ')
        lines.append(f"|{row['head_kappa']:g}|{row['loss']:.6f}|{row['perplexity']:.3f}|"
                     f"{100*row['head_exact_zero_fraction']:.3f}%|{100*tiles['empty']/tiles['total']:.3f}%|"
                     f"{100*counts['skipped_mma']/counts['potential_mma']:.3f}%|")
    delta = rows[1]['loss'] - rows[0]['loss']
    direction = 'increases' if delta >= 0 else 'decreases'
    lines += ['', f'Raising only head κ from.1 to.5 {direction} final native validation loss by{abs(delta):.6f}',
              f'and changes perplexity by a factor of{rows[1]["perplexity"]/rows[0]["perplexity"]:.4f}.',
              'No acceptable-quality cutoff was declared.', '',
              'The M16/K16 consumer now matches the16×16 gate. In B1 inference, empty gate tiles, empty consumer tiles',
              'and skipped/potential MMA fractions agree exactly. Real-operand device counters verify the branches over',
              'all338 blocks in replicate1 for both custom modes. This removes the M8-to-M16 union mismatch of Run067.',
              'Source-level work counters do not measure HBM traffic.', '',
              '## Latency on one RTX5090', '',
              'Synchronized host geometric means, milliseconds. Three fresh processes/model;64 fixed inputs×7 randomized',
              'paired passes/process. Static weight packing, input staging, loss, softmax and token selection are outside timing.',
              'CUDA-event measurements use separate replays. The three boundaries overlap and are not additive.', '',
              '| Head κ | Boundary | Native dense | Custom no-skip | Custom sparse | Sparse reduction vs dense | Reduction vs no-skip |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        comparison = endpoint[row['id']]
        for boundary, label in [('projection_only', 'Projection only'),
                                ('gate_cast_projection', 'Gate + cast + projection'),
                                ('full_forward', 'Complete forward')]:
            timing = comparison['timings'][boundary]['host_ms']
            values = timing['geomean_ms']
            lines.append(f"|{row['head_kappa']:g}|{label}|{values['dense']:.6f}|{values['custom_no_skip']:.6f}|"
                         f"{values['custom_sparse']:.6f}|{reduction(timing['dense_over_sparse']):+.2f}%|"
                         f"{reduction(timing['no_skip_over_sparse']):+.2f}%|")
    lines += ['', 'Positive reductions mean faster sparse execution; negative values mean slower.',
              'The no-skip control isolates bypass inside the custom package. The dense comparison also includes layout/preparation costs.', '',
              '## Numerical qualification and conclusion', '',
              '| Head κ | B1 native loss | Maximum absolute logit error, all modes/processes | Maximum pooled loss delta | Full-forward reduction range |',
              '|---:|---:|---:|---:|---:|']
    for row in rows:
        comparison = endpoint[row['id']]
        processes = comparison['processes']
        maximum = max(p['max_logit_error'] for p in processes)
        loss_delta = max(abs(loss-p['loss']['native']) for p in processes for mode,loss in p['loss'].items())
        ranges = [reduction(r) for r in comparison['timings']['full_forward']['host_ms']['process_dense_over_sparse']]
        lines.append(f"|{row['head_kappa']:g}|{processes[0]['loss']['native']:.9f}|{maximum:g}|{loss_delta:g}|"
                     f"{min(ranges):+.2f}% to{max(ranges):+.2f}%|")
    lines += ['', 'All six processes passed full338-block native-reference qualification and standalone head-graph checks',
              'under the unchanged numerical bounds. B1 native losses are the kernel reference; B32 training losses are quality measurements.', '']
    for row in rows:
        timing = endpoint[row['id']]['timings']
        full = reduction(timing['full_forward']['host_ms']['dense_over_sparse'])
        head = reduction(timing['gate_cast_projection']['host_ms']['dense_over_sparse'])
        full_direction = 'reduces' if full >= 0 else 'increases'
        head_direction = 'reduces' if head >= 0 else 'increases'
        lines.append(f"For head κ={row['head_kappa']:g}, custom sparse execution {head_direction} head-path latency by"
                     f"{abs(head):.2f}% and {full_direction} complete-forward latency by{abs(full):.2f}% versus native dense.")
    lines += ['', 'The two-model training comparison isolates head threshold at fixed alignment and pressure. Comparisons with Run066',
              'are historical context and do not isolate alignment alone. Each within-model runtime comparison keeps its checkpoint,',
              'gate and complete vocabulary contract matched. One training seed/model size and one GPU/workload limit generalization;',
              'these measurements do not establish cached-decoding performance or benefits from integrating earlier block kernels.', '',
              'Sources: [native training reduction](training-comparison.json), [endpoint reduction](../inference/results/comparison.json),',
              '[training transfer receipt](../prelaunch/retrieval-training-001.json), [endpoint transfer receipt](../inference/transfer/receipt-001.json).',
              'Figures: [quality and aligned occupancy](../figures/01-quality-and-aligned-occupancy.pdf),',
              '[dense/no-skip/sparse latency](../figures/02-dense-and-sparse-endpoint-latency.pdf).',
              'Methods/captions: [observations](../observations/INDEX.md). Generating script: [21_summary.py](../21_summary.py).']
    (RUN / 'results/REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print({'status':'consolidated','conditions':len(rows)},flush=True)


if __name__ == '__main__':
    main()
