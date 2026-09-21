"""Reduce the complete paired cohort; never discard a slow or failed process."""
import argparse
import math
import statistics
from io_utils import RUN, read, write
from controls import MODES, MASKS, contrasts

BACKENDS = ('native_graph', 'frozen_graph') + tuple(m+'_graph' for m in MODES)


def geomean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite timings required')
    return math.exp(statistics.mean(math.log(v) for v in values))


def check_samples(samples, inputs=64, passes=7):
    expected = {(r, i, m) for r in range(passes) for i in range(inputs) for m in BACKENDS}
    actual = [(s['repeat'], s['input_index'], s['mode']) for s in samples]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicated paired timing cells')
    for row in samples:
        if row['output_shape'] != [1, 2048, 50304]:
            raise ValueError('Full logits required')
        for field in ['host_ms', 'cuda_ms']:
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError('Invalid timing')


def reduce(tag):
    cfg = read(RUN/'config.json')
    rows = read(RUN/f'artifacts/final/summary-{tag}.json')
    assert len(rows) == 3 and {r['replicate'] for r in rows} == {1, 2, 3}
    pooled = {m: [] for m in BACKENDS}
    cuda = {m: [] for m in BACKENDS}
    process = []
    indices = None
    for row in rows:
        dest = RUN/'artifacts/attempts'/row['attempt']
        result = read(dest/'result.json')
        assert row['returncode'] == 0 and result['status'] == 'complete' and result['qualified']
        assert result['validation_blocks'] == 338
        quality = read(dest/'quality.json')
        assert quality['prediction_tokens'] == 691886 and all(quality['pass'].values())
        assert all(quality['exact_eager_agreement'].values())
        assert all(len(gates) == 338 and all(g['max_abs'] == 0 for g in gates)
                   for gates in quality['gates'].values())
        timing = read(dest/'timing.json')
        if indices is None:
            indices = timing['indices']
        assert indices == timing['indices'] and len(set(indices)) == 64
        check_samples(timing['samples'])
        values = {m: [] for m in BACKENDS}
        for sample in timing['samples']:
            m = sample['mode']
            values[m].append(sample['host_ms'])
            pooled[m].append(sample['host_ms'])
            cuda[m].append(sample['cuda_ms'])
        means = {m: geomean(v) for m, v in values.items()}
        process.append({'replicate': row['replicate'], 'host_ms': means,
                        'effects': contrasts({m: means[m+'_graph'] for m in MODES}),
                        'loss': quality['loss']})
    first = RUN/'artifacts/attempts'/next(r['attempt'] for r in rows if r['replicate']==1)
    zero_check = read(first/'zero-pattern-check.json')
    assert zero_check['blocks'] == 338 and zero_check['identical_values_and_masks']
    diagnostics = {m: read(first/f'diagnostics-{m}.json') for m in MODES}
    for m, d in diagnostics.items():
        assert d['coverage']['blocks'] == 338 and len(d['hz_operand_hashes']) == 12
        assert d['hz_operand_hashes'] == diagnostics['A']['hz_operand_hashes']
        assert d['per_site_layer'] == diagnostics['A']['per_site_layer']
        for counts in d['hybrid_counts_by_layer'].values():
            for active, skip_index, scalar_index in zip(MASKS[m], [1, 3], [4, 5]):
                if not active:
                    assert counts[skip_index] == counts[scalar_index] == 0
    latency = {m: {'host_geomean_ms': geomean(pooled[m]),
                   'host_median_ms': statistics.median(pooled[m]),
                   'cuda_geomean_ms': geomean(cuda[m]), 'samples': len(pooled[m]),
                   'process_geomean_range_ms': [min(p['host_ms'][m] for p in process),
                                                max(p['host_ms'][m] for p in process)]}
               for m in BACKENDS}
    effects = contrasts({m: latency[m+'_graph']['host_geomean_ms'] for m in MODES})
    output = {'status': 'verified', 'question': 'Fixed-checkpoint contribution of h/z zero exploitation',
              'checkpoint': read(RUN/'provenance/inputs.json')['checkpoints'][0],
              'configuration': cfg, 'latency': latency, 'effects': effects, 'processes': process,
              'zero_patterns_and_values_identical': True, 'all_logits_equal_to_eager': True,
              'diagnostic_counters': {m: d['hybrid_counts_by_layer'] for m,d in diagnostics.items()},
              'untouched_k050_over_A_ratio': latency['frozen_graph']['host_geomean_ms']/latency['A_graph']['host_geomean_ms'],
              'same_checkpoint_native_over_A_ratio': latency['native_graph']['host_geomean_ms']/latency['A_graph']['host_geomean_ms'],
              'interpretation': 'Conditional h/z sparse-path savings; no additive allocation or cross-session Base attribution.'}
    write(RUN/'results/hz-latency.json', output)
    lines = ['# Fixed-checkpoint h/z latency control', '', '| Mode | h skip | z skip | Host latency (ms) | Process range (ms) |',
             '| --- | --- | --- | ---: | ---: |']
    for m in MODES:
        value = latency[m+'_graph']; low, high = value['process_geomean_range_ms']
        lines.append(f"| {m} | {MASKS[m][0]} | {MASKS[m][1]} | {value['host_geomean_ms']:.6f} | {low:.6f}–{high:.6f} |")
    lines += ['', 'Thresholds, gate values and zero masks are identical in A–D on all338 validation blocks.',
              'Every final-process logit equals the common eager reference numerically (maximum absolute difference0).', '']
    for name in ['h_given_z', 'z_given_h', 'joint_hz']:
        effect = effects[name]
        lines.append(f"- {name}: {1000*effect['saved_ms']:.3f} us saved; {effect['reduction_percent']:.3f}% reduction relative to its disabled mode; {effect['speedup']:.5f}x.")
    lines += [f"- Interaction D−B−C+A: {1000*effects['interaction_ms']:.3f} us.", '',
              'Ranges describe three fresh processes, not population confidence intervals. Both tile skipping and short-row scalar execution are part of the enabled sparse path.']
    (RUN/'results/hz-latency.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print({m: latency[m+'_graph']['host_geomean_ms'] for m in MODES}, effects, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    assert args.tag.isalnum()
    reduce(args.tag)
