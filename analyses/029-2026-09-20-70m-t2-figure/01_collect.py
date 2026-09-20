"""Extend the existing original-port figure with the five verified 70M T2 endpoints."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'analyses/024-2026-09-17-h-only-kernel-latency'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    old = read(OLD / 'data/70m-quality-sparsity-native-latency.json')
    for relative, digest in old['sources_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    assert len(old['trained_points']) == 22
    data = copy.deepcopy(old)
    style = read(ROOT / 'analyses/027-2026-09-20-run044-manuscript/data/figure-data.json')
    t2_style, = [s for s in style['series'] if s['scope'] == 'hz']
    added = []
    ceilings = []
    for name in ['043-2026-09-20-pythia70m-hz-h-only-ol1',
                 '046-2026-09-20-pythia70m-hz-h-only-ol1-kappa05']:
        run = ROOT / 'runs' / name
        summary = read(run / 'artifacts/summary.json')
        training = read(run / 'artifacts/verification.json')
        latency = read(run / 'latency/artifacts/verification.json')
        assert training['status'] == 'verified' and latency['all_required_artifacts_verified']
        assert summary['post_hoc_clipping'] == 'none'
        for source in summary['sources']:
            path = run / source['path']
            assert sha(path) == source['sha256'], path
            sources[path.relative_to(ROOT).as_posix()] = source['sha256']
        for i, row in enumerate(sorted(training['conditions'], key=lambda r:r['condition']['gate_threshold'])):
            condition = row['condition']
            kappa = condition['gate_threshold']
            reported, = [r for r in summary['conditions'] if r['kappa'] == kappa]
            assert condition['active_sites'] == ['h', 'z']
            assert condition['pressure_sites'] == ['h'] and condition['pressure_weight'] == 1
            assert condition['pressure_method'] == 'orthogonal_l1'
            assert row['completed_steps'] == 712
            logical = read(run / 'artifacts/attempts' / row['attempt_id'] / 'diagnostics/logical_products.json')
            measured = logical['measured']
            fraction = measured['block_zero_product_count'] / measured['model_product_count']
            assert fraction == row['R_model'] == reported['R_model']
            assert logical['coverage']['sequences'] == 338
            assert logical['coverage']['excluded_tail_tokens'] == 1444
            assert row['final_validation_loss'] == reported['training_validation_loss']
            ceilings.append(logical['architecture_maximum'])
            candidate_times, native_times, devices = [], [], set()
            for replicate in range(1, 4):
                folder = run / 'latency/artifacts/attempts' / f'scientific-c{i:02d}-r{replicate}-001'
                result = read(folder / 'result.json')
                timing = read(folder / 'timing.json')
                quality = read(folder / 'quality.json')
                assert result['status'] == 'complete' and result['qualified']
                assert result['candidate'] == 'k050-70m-v2'
                assert result['checkpoint']['final_checkpoint_content_sha256'] == row['checkpoint_content_sha256']
                assert quality['blocks'] == 338 and quality['documents'] == 500
                assert quality['excluded_tail_tokens'] == 1444
                devices.add(result['runtime']['device_uuid'])
                for mode, destination in [('candidate_graph', candidate_times), ('native_graph', native_times)]:
                    samples = [s for s in timing['samples'] if s['mode'] == mode]
                    assert len(samples) == 448
                    assert len({(s['repeat'], s['input_index']) for s in samples}) == 448
                    assert all(s['output_shape'] == [1, 2048, 50304] for s in samples)
                    destination.extend(s['host_ms'] for s in samples)
            assert len(devices) == 1 and len(candidate_times) == 1344
            geomean = lambda values: math.exp(statistics.mean(map(math.log, values)))
            assert math.isclose(geomean(candidate_times), reported['k050_geomean_host_ms'], rel_tol=1e-13)
            assert math.isclose(geomean(native_times), reported['native_geomean_host_ms'], rel_tol=1e-13)
            added.append(dict(checkpoint_key=row['checkpoint_content_sha256'], model='70M', scope='hz',
                pressure='h', kappa=kappa, sparsity=100*fraction, loss=row['final_validation_loss'],
                latency_ms=reported['k050_geomean_host_ms'], kernel='k050-70m-v2',
                timing_session='Run'+name[:3], timing_device_uuid=devices.pop(),
                native_same_checkpoint_ms=reported['native_geomean_host_ms'],
                paired_native_speedup=reported['native_relative_paired_geomean_speedup'],
                qualified=True, paired_samples=1344, logical_counts=measured,
                training_run=name, training_attempt=row['attempt_id']))
    added.sort(key=lambda r:r['kappa'])
    assert [r['kappa'] for r in added] == [0, .01, .05, .1, .5]
    assert all(c == ceilings[0] for c in ceilings)
    data['trained_points'] += added
    data['plotted_points'] += [dict(r, displayed_latency_ms=r['latency_ms'], execution='original 70M port') for r in added]
    data['series'].insert(2, dict(t2_style, model='70M', keys=[r['checkpoint_key'] for r in added]))
    assert len({r['checkpoint_key'] for r in data['trained_points']}) == 27
    for panel in data['panels']:
        panel['trained_keys'] += [r['checkpoint_key'] for r in added]
    original = read(OLD / 'data/14m-70m-quality-sparsity-latency.json')
    data['ceilings'] = dict(original['ceilings']['70M'], hz=ceilings[0])
    data['sources_sha256'] = sources
    data['historical_summary'] = data.pop('summary')
    data['added_t2_points'] = added
    data['note'] = ('27 checkpoints. The original 22 points, control clipping and native Base reference are unchanged. '
        'T2/Ph uses the same frozen k050-70m-v2 in Run043 (kappa 0 through .1) and Run046 (.5), '
        'on later RTX5090 sessions. Absolute cross-session differences are descriptive, not controlled speedups. '
        'Same-checkpoint native ratios are retained separately. No Run045 optimized measurements are used.')
    data['observation'] = 'observations/001-70m-t2-figure.md'
    data['layout'].update(legend_columns=4, legend_rows=2)
    data['collector'] = Path(__file__).name
    data['collector_sha256'] = sha(Path(__file__))
    for key in ['script', 'script_sha256', 'output_sha256']:
        data.pop(key, None)
    (HERE/'data').mkdir(exist_ok=True)
    (HERE/'data/70m-quality-sparsity-native-latency.json').write_text(
        json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'trained':27, 'new_t2':5, 'qualified_new_processes':15, 'sources':len(sources)}))


if __name__ == '__main__':
    main()
