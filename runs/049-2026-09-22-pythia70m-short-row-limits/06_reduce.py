"""Verify complete paired measurements and reduce each backend without pooling sessions."""
import argparse
import math
from io_utils import RUN, read, write, verify, record


def geomean(values):
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError('Positive finite timing observations required')
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def reduce():
    cfg = read(RUN / 'config.json')
    selection = read(RUN / 'provenance/selection.json')
    for source in selection['evidence'] + [selection['source_freeze']]:
        verify(source)
    conditions, devices = [], set()
    for cid in cfg['conditions']:
        processes = []
        for rep in range(1, 4):
            files = list((RUN / 'artifacts/attempts').glob(f'final-{cid}-r{rep}-*/result.json'))
            assert len(files) == 1, (cid, rep)
            p = files[0]
            r, timing, quality = read(p), read(p.with_name('timing.json')), read(p.with_name('quality.json'))
            assert r['status'] == 'complete' and r['validation_blocks'] == 338
            assert quality['blocks'] == 338 and quality['prediction_tokens'] == 338 * 2047
            assert quality['documents'] == 500 and quality['excluded_tail_tokens'] == 1444
            assert r['selection'] == record(RUN / 'provenance/selection.json')
            devices.add(r['runtime']['device_uuid'])
            samples = timing['samples']
            assert len(timing['indices']) == len(set(timing['indices'])) == 64
            assert len(samples) == 6 * 64 * 7
            identities = {(s['mode'], s['input_index'], s['repeat']) for s in samples}
            assert len(identities) == len(samples)
            for sample in samples:
                assert sample['output_shape'] == [1, 2048, 50304]
            for mode in cfg['implementations']:
                key = mode + '_graph'
                assert len([s for s in samples if s['mode'] == key]) == 448
                assert len(quality['gates'][key]) == 338
                assert [g['input_index'] for g in quality['gates'][key]] == list(range(338))
                qualified = all(g['pass'] for g in quality['gates'][key]) and abs(quality['loss_delta'][key]) <= cfg['numerical_bounds']['validation_loss_atol']
                assert qualified == r['qualification'][key] == quality['pass'][key]
            if rep == 1:
                for mode in cfg['implementations'][1:]:
                    d = read(p.with_name('diagnostics-' + mode + '.json'))
                    assert d['status'] == 'complete' and d['coverage']['blocks'] == 338
                    assert len(d['joint_occupancy_by_layer_limit']) == 24
                    assert all(v['rows'] == 338 * 2048 and v['groups'] == 338 * 256 for v in d['joint_occupancy_by_layer_limit'].values())
                assert len(read(p.with_name('profile-summary.json'))['profiles']) == 12
            processes.append((r, timing, quality, p))
        modes = {}
        for mode in cfg['implementations']:
            key = mode + '_graph'
            observations = [s['host_ms'] for _, t, _, _ in processes for s in t['samples'] if s['mode'] == key]
            means = [geomean([s['host_ms'] for s in t['samples'] if s['mode'] == key]) for _, t, _, _ in processes]
            modes[mode] = {'latency_ms': geomean(observations), 'qualified': all(r['qualification'][key] for r, _, _, _ in processes),
                           'observations': len(observations), 'process_range_ms': [min(means), max(means)],
                           'loss_by_process': [q['loss'][key] for _, _, q, _ in processes]}
        conditions.append({'id': cid, 'family': processes[0][0]['checkpoint']['family'],
            'kappa': processes[0][0]['checkpoint']['dose'], 'modes': modes,
            'sources': [record(path) for _, _, _, p in processes for path in (p, p.with_name('timing.json'), p.with_name('quality.json'))]})
    assert len(devices) == 1
    base = next(r for r in conditions if r['id'] == 'c00')['modes']['native']['latency_ms']
    for row in conditions:
        for mode, cell in row['modes'].items():
            cell['speedup_native_base'] = base / cell['latency_ms'] if cell['qualified'] else None
            cell['speedup_same_checkpoint_native'] = row['modes']['native']['latency_ms'] / cell['latency_ms'] if cell['qualified'] else None
            cell['speedup_efficient_hz'] = row['modes']['native_hz']['latency_ms'] / cell['latency_ms'] if cell['qualified'] and row['modes']['native_hz']['qualified'] else None
    return {'status': 'verified', 'fresh_processes': 9, 'implementations': 6, 'device_uuid': devices.pop(),
            'selected_before_validation': selection['selected'], 'native_base_ms': base, 'conditions': conditions,
            'interpretation': 'Same-session full-forward timings; native_hz gains include dense optimization. Numerical failures receive no speedup.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    result = reduce()
    if a.verify_only:
        assert result == read(RUN / 'results/summary.json')
    else:
        write(RUN / 'results/summary.json', result)
    print({k: v for k, v in result.items() if k != 'conditions'})
