"""Explain the verified head comparison using pooled counts and process ranges."""
import math
from io_utils import RUN, read, record, write


def label(row):
    return 'Ph+head' if len(row['pressure_sites']) == 2 else 'Ph'


def main():
    comparison = read(RUN/'results/comparison.json')
    rows = comparison['rows']
    manifests = [read(RUN/'artifacts/attempts'/f'full-{r["id"]}-{p}-001'/'result.json')
                 for r in rows for p in (1, 2, 3)]
    uuids = {m['runtime']['gpu_uuid'] for m in manifests}
    commits = {m['source_checkout']['git_commit'] for m in manifests}
    assert len(uuids) == len(commits) == 1 and 'unavailable' not in uuids
    assert all(p['max_logit_error'] == 0 for r in rows for p in r['processes'])
    assert all(len(set(p['loss'].values())) == 1 for r in rows for p in r['processes'])
    lines = [
        '# Vocabulary-head kernel: diagnosis and quality', '',
        'The kernel is numerically exact on this workload, but slower than dense cuBLAS. '
        'Skipping helps within the custom implementation; its baseline execution cost outweighs that saving.', '',
        '## What is sparse and what executes densely', '',
        'All four frozen 14M models have the original T2 corner gates at h, z and the head input. '
        'The head gate acts on the FP32 output of the learned affine final LayerNorm, using 8-token by '
        '16-feature tiles. An original corner <= kappa rejects the entire tile. Surviving signed '
        'interiors, checkpoint weights and all 50,304 vocabulary scores are preserved.', '',
        '- **Dense cuBLAS:** gated activations, dense block projections, dense vocabulary projection.',
        '- **Custom no-skip:** the same activations and blocks; custom head preparation and projection with bypass disabled.',
        '- **Custom sparse:** the same activations and blocks; only the vocabulary consumer skips empty reduction groups.', '',
        'Ph and Ph+head name the retained checkpoints\' training pressure histories. Both histories have '
        'the head gate. They do not name different inference kernels. No new training or pressure is applied. '
        'Earlier BF16 block consumers are not part of this FP32-parameter/FP16-autocast comparison.', '',
        '```text',
        'native transformer blocks (h/z gated; projections execute densely)',
        '  -> learned affine final LayerNorm (FP32)',
        '  -> original head 8x16 corner gate (FP32)',
        '  -> FP16 input conversion; custom paths also retain native FP32 corner support',
        '  -> vocabulary projection [2048,128] @ [128,50304]',
        '       dense cuBLAS / same custom no-skip / custom sparse',
        '  -> [2048,50304] FP16 logits (all entries written)',
        '```', '',
        'Loss, softmax, token selection and input staging are outside the latency measurements. '
        'Static weight conversion and custom fragment packing are outside timing and recorded separately.', '',
        '## Quality and pooled head activation statistics', '',
        'Native, dense, custom no-skip and custom sparse losses agree exactly within each process. '
        'All 12 processes have maximum logit error zero over all 338 validation blocks. '
        'These B1 RTX 5090 native losses are their own reference; Run066 B32 and local-laptop losses are provenance.', '',
        '| kappa | Pressure history | Native = all-path loss | Exact zeros | Empty 8x16 gate tiles | Empty 16x16 consumer tiles / skipped MMA | RMS |',
        '|---:|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        c = r['head_counts']
        assert c['skipped_mma']*c['consumer_tiles'] == c['empty_consumer_tiles']*c['potential_mma']
        lines.append(f'| {r["kappa"]:g} | {label(r)} | {r["processes"][0]["loss"]["native"]:.9f} | '
                     f'{100*c["exact_zero"]/c["elements"]:.3f}% | '
                     f'{100*c["empty_gate_tiles"]/c["gate_tiles"]:.3f}% | '
                     f'{100*c["empty_consumer_tiles"]/c["consumer_tiles"]:.3f}% | '
                     f'{math.sqrt(c["sum_squares"]/c["elements"]):.6f} |')
    lines += ['', 'Counts are pooled as integers over all 338 blocks before division. RMS uses the pooled '
              'sum of squares. Exact/near-zero counts and L1/L2 accumulators are retained in comparison.json '
              'and each head-diagnostics.json.', '',
              'The M16/N256/K16 consumer combines two independent M8/K16 gate tiles. A reduction group is '
              'skipped only when **both** are empty. The 17.34-27.42% gate emptiness therefore becomes only '
              '7.07-15.92% actual consumer/MMA bypass. This is a measured occupancy mismatch. '
              'MMA counts are source-level work counts, not measured DRAM traffic.', '',
              '## Projection-only diagnostic', '',
              'Times below are synchronized host microseconds. Projection-only excludes recurring input '
              'preparation and is a diagnostic, separate from the primary head path in REPORT.md.', '',
              '| kappa | Pressure history | Dense cuBLAS | Custom no-skip | Custom sparse |',
              '|---:|---|---:|---:|---:|']
    for r in rows:
        t = r['timings']['projection_only']['host_ms']['geomean_ms']
        lines.append(f'| {r["kappa"]:g} | {label(r)} | {t["dense"]*1000:.2f} | '
                     f'{t["custom_no_skip"]*1000:.2f} | {t["custom_sparse"]*1000:.2f} |')
    lines += ['', '## Where the net latency change comes from', '',
              'The measured head path includes the original gate plus recurring conversion/certification '
              'and projection. The same-custom-kernel control isolates the benefit of bypassing within '
              'that package. The cuBLAS comparison additionally reflects different custom layout and execution costs.', '',
              '| kappa | Pressure history | Custom no-skip penalty vs dense | Head saving from skipping | Head saved (us) | Sparse forward slowdown | Range across 3 processes |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        h = r['timings']['gate_cast_projection']['host_ms']
        f = r['timings']['full_forward']['host_ms']
        t = h['geomean_ms']
        process = [100*(1/x-1) for x in f['process_dense_over_sparse']]
        lines.append(f'| {r["kappa"]:g} | {label(r)} | {100*(t["custom_no_skip"]/t["dense"]-1):.2f}% | '
                     f'{100*(1-1/h["no_skip_over_sparse"]):.2f}% | '
                     f'{1000*(t["custom_no_skip"]-t["custom_sparse"]):.2f} | '
                     f'{100*(1/f["dense_over_sparse"]-1):.2f}% | {min(process):.2f}-{max(process):.2f}% |')
    lines += ['', 'The custom no-skip head is about 63% slower than cuBLAS. Skipping recovers 11.43-16.79% '
              'of its own head-path latency, or about 38-56 us. That saving is smaller than the baseline '
              'custom penalty, leaving the sparse head 35.37-44.55% slower than dense and complete forward '
              '7.19-8.86% slower. All three processes reproduce the forward slowdown.', '',
              'Every execution writes 2048 x 50304 FP16 logits (206,045,184 bytes). Sparsity changes '
              'reduction work, not that output contract. The source-level bypass, occupancy and paired '
              'timings establish the saving and baseline penalty; they do not identify an exact instruction, '
              'cache or DRAM bottleneck. Gate and cast times were not independently measured, and the '
              'three timing boundaries must not be subtracted as an additive decomposition.', '',
              '## Independent CUDA-event check', '',
              'The separate event-timed replays reproduce the slowdown. These are GPU elapsed times '
              'and are not substituted for the primary host measurements.', '',
              '| kappa | Pressure history | Dense head (us) | Sparse head (us) | Dense forward (us) | Sparse forward (us) | Forward slowdown |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        h = r['timings']['gate_cast_projection']['cuda_ms']['geomean_ms']
        f = r['timings']['full_forward']['cuda_ms']
        t = f['geomean_ms']
        lines.append(f'| {r["kappa"]:g} | {label(r)} | {1000*h["dense"]:.2f} | '
                     f'{1000*h["custom_sparse"]:.2f} | {1000*t["dense"]:.2f} | '
                     f'{1000*t["custom_sparse"]:.2f} | {100*(1/f["dense_over_sparse"]-1):.2f}% |')
    lines += ['', '## Coverage, aggregation and provenance', '',
              'One physical RTX 5090; 3 fresh processes per checkpoint, 64 fixed inputs x 7 randomized '
              'paired passes per process and boundary. Runtime seed 2801, timing seed 2504. Full validation: '
              '500 documents, 338 complete 2048-token blocks, 692,224 inputs, 691,886 predictions; '
              '1,444 tail tokens excluded. The 12 full processes cover 12,168 full-graph logit comparisons '
              'and 4,608 standalone head-graph comparisons, all exact. The cloud component suite adds 48 checks.', '',
              'Latency values are geometric means of the paired samples, aggregated equally over the three '
              'processes. Ratios are calculated from the paired records. Process ranges show repeatability '
              'on this GPU, not a population confidence interval. CUDA-event timings use separate replays '
              'and remain in comparison.json alongside the primary synchronized-host times.', '',
              f'- GPU UUID: `{next(iter(uuids))}`.',
              f'- Verified deployment commit: `{next(iter(commits))}`.',
              '- [Primary table](REPORT.md), [raw reduction](comparison.json), [PDF](../figures/01-sparse-vocabulary-head.pdf).',
              '- [Scientific transfer receipt](../transfer/receipt-001.json) verifies 163 files; [delivery receipt](../ops/delivery-receipt-001.json) verifies 8 supplemental files.',
              '- [Resource closeout](../ops/teardown-001.json) confirms the owned Pod was deleted after verification. Existing foreign resources were unchanged.',
              '- [20_analyze.py](../20_analyze.py) reduces raw paired records; [22_diagnostic_report.py](../22_diagnostic_report.py) generates these tables and interpretation.', '',
              'This result refutes practical acceleration by this implementation for B1/T2048 at 14M. '
              'It does not refute sparse-head kernels generally, establish cached-decoding performance, '
              'or justify changing trained gates. No new training, manuscript change or finding promotion is included.']
    output = RUN/'results/DIAGNOSTICS.md'
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    observation = RUN/'observations/01-sparse-vocabulary-head.md'
    original = observation.read_text(encoding='utf-8').split('\n## Verified numerical interpretation\n')[0]
    original += ('\n## Verified numerical interpretation\n\n'
                 'All 12 full processes and all head graphs match their native logits/loss exactly. '
                 'The custom sparse head saves 11.43-16.79% versus custom no-skip, but is 35.37-44.55% '
                 'slower than dense cuBLAS; full forward is 7.19-8.86% slower. All three processes '
                 'reproduce the slowdown. Empty 8x16 gate tiles are 17.34-27.42%; the paired 16x16 '
                 'consumer bypasses only 7.07-15.92%. The baseline custom execution penalty outweighs '
                 'skipping savings. See [diagnostic tables and process ranges](../results/DIAGNOSTICS.md), '
                 'generated by [22_diagnostic_report.py](../22_diagnostic_report.py). '
                 'The two head panels share their y-axis scale; full forward has its own scale. '
                 'The separately measured boundaries do not form an additive decomposition.\n')
    observation.write_text(original, encoding='utf-8', newline='\n')
    write(RUN/'results/diagnostic-provenance.json', {'input':record(RUN/'results/comparison.json'),
          'script':record(RUN/'22_diagnostic_report.py'), 'output':record(output),
          'gpu_uuid':next(iter(uuids)), 'deployment_commit':next(iter(commits))})
    print({'status':'diagnosed', 'models':len(rows), 'gpu_uuids':len(uuids)}, flush=True)


if __name__ == '__main__':
    main()
