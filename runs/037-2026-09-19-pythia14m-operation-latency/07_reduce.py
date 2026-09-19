"""Reduce the ten matched modes; retain conditional, nonadditive effects."""
import math
from controls import MODES, OPS, contrasts, mask
from io_utils import RUN, read, write, record


def geomean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite times required')
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def main():
    summary = read(RUN / 'artifacts/scientific/summary-001.json')
    assert {(r['mode'], r['replicate']) for r in summary} == {(m, r) for m in MODES for r in [1, 2, 3]}
    assert len(summary) == 30
    samples = {mode: [] for mode in MODES}
    processes = {mode: [] for mode in MODES}
    sources = []
    indices = None
    counters = {}
    for item in summary:
        folder = RUN / 'artifacts/attempts' / item['attempt']
        result, timing, quality = [read(folder / name) for name in ['result.json', 'timing.json', 'quality.json']]
        assert result['status'] == 'complete' and result['qualified'] and result['validation_blocks'] == 338
        assert result['condition'] == 'c30' and result['candidate'] == item['mode']
        assert result['implementation_coverage']['operation_mask'] == mask(item['mode'])
        assert all(quality['pass'].values()) and quality['excluded_tail_tokens'] == 1444
        if indices is None: indices = timing['indices']
        assert timing['indices'] == indices and len(set(indices)) == 64
        per_mode = {}
        for label in ['candidate_graph', 'native_graph']:
            selected = [s for s in timing['samples'] if s['mode'] == label]
            assert len(selected) == 448
            assert {(s['input_index'], s['repeat']) for s in selected} == {(i, r) for i in range(64) for r in range(7)}
            assert all(s['output_shape'] == [1, 2048, 50304] for s in selected)
            per_mode[label] = geomean([s['host_ms'] for s in selected])
            if label == 'candidate_graph': samples[item['mode']].extend(s['host_ms'] for s in selected)
        processes[item['mode']].append({'replicate': item['replicate'], **per_mode})
        for name in ['result.json', 'timing.json', 'quality.json']:
            sources.append(record(folder / name))
        if item['replicate'] == 1:
            diag = read(folder / 'diagnostics.json')
            assert diag['coverage']['blocks'] == 338
            counters[item['mode']] = diag
            sources.append(record(folder / 'diagnostics.json'))
    latencies = {mode: geomean(values) for mode, values in samples.items()}
    ranges = {mode: [min(r['candidate_graph'] for r in rows), max(r['candidate_graph'] for r in rows)]
              for mode, rows in processes.items()}
    overlap = max(ranges['full'][0], ranges['frozen'][0]) <= min(ranges['full'][1], ranges['frozen'][1])
    report = {'status': 'qualified' if overlap else 'fidelity-review-required',
              'interpretation': 'Conditional full-model effects, not additive site allocations',
              'latencies_ms': latencies, 'processes': processes, 'process_ranges_ms': ranges,
              'full_over_frozen_latency_ratio': latencies['full'] / latencies['frozen'],
              'full_frozen_process_ranges_overlap': overlap,
              'fidelity_rule': 'Non-overlapping observed process ranges require review before K050 attribution; '
                               'overlap is descriptive and does not prove equivalence.',
              'conditional_effects': contrasts(latencies), 'diagnostics': counters, 'sources': sources}
    write(RUN / 'results/operation-latency.json', report)
    print(f"Reduced 30 processes; status={report['status']}; full/frozen={report['full_over_frozen_latency_ratio']:.6f}")


if __name__ == '__main__': main()
