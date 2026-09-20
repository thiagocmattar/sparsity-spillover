"""Recompute every contrast from qualified raw samples and fixed references."""
import math
from collections import defaultdict
from io_utils import RUN, read, write, record


def gm(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def reduce_attempts():
    cfg = read(RUN / 'config.json')
    groups = defaultdict(list)
    sources = []
    devices = set()
    for path in sorted((RUN / 'artifacts/attempts').glob('*/result.json')):
        result = read(path)
        if result['arguments'].get('smoke') or result['arguments'].get('development'):
            continue
        if result['status'] != 'complete' or not result.get('qualified'):
            continue
        assert result['validation_blocks'] == 338
        quality = read(path.with_name('quality.json'))
        assert quality['blocks'] == 338 and quality['documents'] == 500 and quality['excluded_tail_tokens'] == 1444
        assert quality['prediction_tokens'] == 691886 and all(quality['pass'].values())
        timing = read(path.with_name('timing.json'))
        expected_indices = read(RUN / 'provenance/timing-indices.json')
        assert timing['indices'] == expected_indices
        values = {}
        by_pair = defaultdict(dict)
        for sample in timing['samples']:
            assert sample['output_shape'] == [1, 2048, 50304]
            pair = by_pair[(sample['input_index'], sample['repeat'])]
            assert sample['mode'] not in pair, 'Duplicate paired timing sample'
            pair[sample['mode']] = sample
        assert set(by_pair) == {(i, p) for i in range(64) for p in range(7)}
        modes = set(next(iter(by_pair.values())))
        assert all(set(pair) == modes for pair in by_pair.values())
        assert {'native_graph', 'candidate_graph'} <= modes
        for mode in modes:
            values[mode] = {timer: [pair[mode][timer] for pair in by_pair.values()]
                            for timer in ('host_ms', 'cuda_ms')}
        devices.add(result['runtime']['device_uuid'])
        groups[(result['condition'], result['candidate'])].append((result, values))
        sources += [record(path), record(path.with_name('quality.json')), record(path.with_name('timing.json'))]
    assert len(devices) == 1, 'All comparisons require one physical GPU session'
    output = []
    for (condition, mode), rows in sorted(groups.items()):
        assert len(rows) == 3 and {r['arguments']['replicate'] for r, _ in rows} == {1, 2, 3}
        keys = set(rows[0][1])
        assert all(set(v) == keys for _, v in rows)
        means = {key: {timer: gm([x for _, values in rows for x in values[key][timer]])
                       for timer in ('host_ms', 'cuda_ms')} for key in keys}
        reference = 'candidate_graph' if mode == 'full' else 'frozen_graph'
        assert reference in means
        output.append({'condition': condition, 'mode': mode, 'geometric_means_ms': means,
                       'full_minus_mode_ms': means[reference]['host_ms'] - means['candidate_graph']['host_ms'],
                       'same_checkpoint_native_speedup': means['native_graph']['host_ms'] / means['candidate_graph']['host_ms'],
                       'process_candidate_host_ms': [gm(v['candidate_graph']['host_ms']) for _, v in rows],
                       'samples_per_mode': 1344})
    base, = [r for r in output if (r['condition'], r['mode']) == ('c00', 'full')]
    reference = base['geometric_means_ms']['native_graph']['host_ms']
    for row in output:
        row['native_base_speedup'] = reference / row['geometric_means_ms']['candidate_graph']['host_ms']
    expected = {(c, m) for c in cfg['conditions'] for m in cfg['diagnostic_modes']}
    complete = expected <= {(r['condition'], r['mode']) for r in output}
    result = {'status': 'complete' if complete else 'partial', 'device_uuid': next(iter(devices)),
              'native_base_reference_ms': reference, 'rows': output, 'sources': sources,
              'interpretation': 'One native T0/P0 reference for every point; replacement effects are conditional and nonadditive.'}
    write(RUN / 'results/component-overhead.json', result)
    return result


if __name__ == '__main__':
    result = reduce_attempts()
    print(result['status'], len(result['rows']), 'conditions reduced')
