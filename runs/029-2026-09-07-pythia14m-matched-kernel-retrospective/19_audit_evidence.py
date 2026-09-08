"""Independent read-only audit of the complete matrix; write only a new audit receipt."""
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from io_utils import RUN, REPO, read, record, verify, write, fs


def geometric(values):
    values = list(values)
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def regression(rows):
    x = np.array([r['R_model'] for r in rows], dtype=np.float64)
    y = np.array([r['speedup'] for r in rows], dtype=np.float64)
    slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    fitted = slope * x + intercept
    return {'n': len(rows), 'intercept': float(intercept), 'slope': float(slope),
            'r_squared': float(1 - np.sum((y-fitted)**2) / np.sum((y-y.mean())**2))}


def qualify(quality, cfg):
    """Recompute qualification from recorded per-block statistics, not summary flags."""
    assert quality['blocks'] == cfg['validation_blocks'] == 338
    assert quality['documents'] == 500 and quality['excluded_tail_tokens'] == 1444
    assert quality['prediction_tokens'] == 338 * 2047
    bounds = cfg['numerical_bounds']
    assert set(quality['gates']) == {'native_graph', 'candidate_graph'}
    losses = quality['loss']
    assert all(math.isfinite(v) for v in losses.values())
    passing = {'native': math.isfinite(losses['native'])}
    for mode, gates in quality['gates'].items():
        assert [g['input_index'] for g in gates] == list(range(338))
        checks = []
        for gate in gates:
            good = (gate['finite'] and gate['elementwise_gate']
                    and gate['relative_l2'] <= bounds['logit_relative_l2'])
            assert good == gate['pass']
            checks.append(good)
        delta = losses[mode] - losses['native']
        assert math.isclose(delta, quality['loss_delta'][mode], abs_tol=1e-12)
        passing[mode] = all(checks) and abs(delta) <= bounds['validation_loss_atol']
    assert passing == quality['pass']
    assert passing['native_graph'] and passing['native']
    return all(passing.values())


def timing_ratios(timing, cfg):
    expected = {(r, i) for r in range(cfg['timing_passes']) for i in range(cfg['timing_inputs'])}
    samples = {}
    for row in timing['samples']:
        key = (row['mode'], row['repeat'], row['input_index'])
        assert key not in samples
        assert row['mode'] in {'native_graph', 'candidate_graph'}
        assert row['output_shape'] == [1, 2048, 50304]
        assert all(math.isfinite(row[f]) and row[f] > 0 for f in ['host_ms', 'cuda_ms'])
        samples[key] = row
    for mode in ['native_graph', 'candidate_graph']:
        assert {(r, i) for m, r, i in samples if m == mode} == expected
    return {field: [samples[('native_graph', r, i)][field] /
                    samples[('candidate_graph', r, i)][field] for r, i in sorted(expected)]
            for field in ['host_ms', 'cuda_ms']}


def main():
    dest = RUN / 'results/implementation-audit-001.json'
    if dest.exists():
        raise ValueError('Audit receipt already exists; do not overwrite')
    checked = set()
    def check(row):
        key = (row['path'], row['sha256'])
        if key not in checked:
            verify(row)
            checked.add(key)
        return RUN / row['path']

    cfg = read(RUN / 'config.json')
    data = read(check(read(RUN / 'results/figures-summary-003.json')['source']))
    plan = read(check(data['plan']))
    for source in plan['sources'] + data['sources']:
        check(source)
    archive = read(RUN / 'provenance/archive.json')
    for source in archive['files']:
        check(source['snapshot'])
        assert source['origin']['sha256'] == source['snapshot']['sha256']
    inputs = read(RUN / 'provenance/inputs.json')
    for source in [inputs['validation']] + [s for c in inputs['checkpoints'] for s in c['files'] + c['provenance']]:
        check(source)
    catalog = read(RUN / 'provenance/candidates.json')['configurations']
    inventory = []
    parsed = set()
    for candidate in catalog:
        for source in candidate.get('source_files', []):
            path = check(source)
            if path.suffix == '.py' and path not in parsed:
                ast.parse(fs(path).read_text(encoding='utf-8'), filename=source['path'])
                parsed.add(path)
        inventory.append({k: candidate[k] for k in ['id', 'candidate', 'status', 'settings', 'note']})

    completed = read(RUN / 'artifacts/scientific/completed.json')
    assert len(completed) == len(plan['jobs']) == 1173
    assert {r['job']['key'] for r in completed} == {r['key'] for r in plan['jobs']}
    indices = np.random.default_rng(cfg['timing_seed']).choice(338, 64, replace=False).tolist()
    groups = defaultdict(list)
    outcomes = Counter()
    native_losses = defaultdict(list)
    uuids = set()
    pair_count = 0
    for item in completed:
        job = item['job']
        path = check(item['result'])
        result = read(path)
        args = result['arguments']
        assert args['smoke'] is False
        assert all(args[k] == job[k] for k in ['candidate', 'condition', 'replicate', 'final'])
        assert args['attempt'] == path.parent.name
        assert result['candidate'] == job['candidate'] and result['condition'] == job['condition']
        for key in ['config', 'archive', 'catalog']:
            check(result[key])
        row = {'replicate': job['replicate'], 'qualified': False, 'status': result['status']}
        if result['status'] == 'complete':
            runtime = result['runtime']
            uuids.add(runtime['device_uuid'])
            assert runtime['gpu'] == cfg['gpu']
            assert runtime['torch'].split('+')[0] == cfg['runtime']['torch']
            assert runtime['transformers'] == cfg['runtime']['transformers']
            assert runtime['cuda'] == cfg['runtime']['cuda']
            quality = read(path.parent / 'quality.json')
            passing = qualify(quality, cfg)
            assert passing == result['qualified']
            native_losses[job['condition']].append(quality['loss']['native'])
            timing = read(path.parent / 'timing.json')
            assert timing['indices'] == indices
            ratios = timing_ratios(timing, cfg)
            pair_count += len(ratios['host_ms'])
            row.update(qualified=passing, speedup=geometric(ratios['host_ms']),
                       cuda_speedup=geometric(ratios['cuda_ms']))
            outcomes['qualified' if passing else 'numerical_failure'] += 1
        else:
            assert result['status'] == 'unsupported'
            assert job['candidate'] in ['k017', 'k018'] and job['condition'] == 'c01'
            outcomes['unsupported'] += 1
        groups[('final' if job['final'] else 'history', job['candidate'], job['condition'])].append(row)
    assert len(uuids) == 1 and uuids == {data['physical_gpu_uuid']}
    assert len(groups) == len(data['points']) == 391
    reconstructed = []
    for point in data['points']:
        key = (point['phase'], point['candidate'], point['condition'])
        rows = groups[key]
        assert sorted(r['replicate'] for r in rows) == [1, 2, 3]
        assert point['qualified'] == all(r['qualified'] for r in rows)
        if all('speedup' in r for r in rows):
            for field in ['speedup', 'cuda_speedup']:
                assert math.isclose(geometric(r[field] for r in rows), point[field], abs_tol=1e-12)
        counts = point['canonical_counts']
        ops = counts['per_operation']
        assert all(type(c[k]) is int and 0 <= c['zero_product_count'] <= c['product_count']
                   for c in ops.values() for k in ['zero_product_count', 'product_count'])
        assert sum(c['product_count'] for c in ops.values()) == counts['block_product_count']
        assert sum(c['zero_product_count'] for c in ops.values()) == counts['block_zero_product_count']
        assert counts['model_product_count'] == counts['block_product_count'] + counts['lm_head_product_count']
        # Independent architecture denominator, including the dense head and causal triangle.
        block = 2048 * (4 * 128**2 + 2 * 128 * 512) + 128 * 2048 * 2049
        assert counts['model_product_count'] == (6 * block + 2048 * 128 * 50304) * 338
        assert point['R_model'] == counts['block_zero_product_count'] / counts['model_product_count']
        reconstructed.append(point)

    mapping = {c['id']: c.get('paper_iteration', 0) for c in catalog if c['status'] == 'eligible'}
    best, winner = 1., 'dense'
    for expected in data['progress']:
        for p in reconstructed:
            if (p['phase'] == 'history' and p['condition'] == cfg['progress_anchor']
                    and p['candidate'] != 'p0' and p['qualified']
                    and mapping[p['candidate']] == expected['iteration'] and p['speedup'] > best):
                best, winner = p['speedup'], p['candidate']
        assert expected['speedup'] == best and expected['incumbent'] == winner

    final = [p for p in reconstructed if p['phase'] == 'final']
    selected = [p for p in final if p['candidate'] == 'k050']
    assert len(selected) == 35 and all(p['qualified'] for p in selected)
    fit = regression(selected)
    for key in ['intercept', 'slope', 'r_squared']:
        assert math.isclose(fit[key], data['k050_regression'][key], abs_tol=1e-12)
    controls = {}
    k050 = {p['condition']: p for p in selected}
    for name in cfg['final_candidates']:
        cohort = [p for p in final if p['candidate'] == name and p['qualified']]
        ratios = [k050[p['condition']]['speedup'] / p['speedup'] for p in cohort]
        controls[name] = {'qualified_checkpoints': len(cohort),
                          'geomean_speedup': geometric(p['speedup'] for p in cohort),
                          'k050_over_control_geomean': geometric(ratios),
                          'k050_faster_count': sum(v > 1 for v in ratios),
                          'regression': regression(cohort)}
    inversions = sum(a['R_model'] < b['R_model'] and a['speedup'] > b['speedup']
                     for a in selected for b in selected)
    leave_one_out = [regression([p for p in selected if p['condition'] != q['condition']])
                     for q in selected]
    figure_manifest = read(RUN / 'results/figures-summary-003.json')
    for key in ['source', 'catalog', 'script', 'figure']:
        check(figure_manifest[key])
    figure = record(RUN / 'figures/04-kernel-autoresearch-and-rmodel.pdf')
    assert figure['sha256'] == figure_manifest['figure']['sha256']
    value = {'utc': datetime.now(timezone.utc).isoformat(), 'script': record(__file__),
             'checked_file_identities': len(checked), 'archived_files': len(archive['files']),
             'candidate_python_files_parsed': len(parsed), 'candidate_inventory': inventory,
             'process_outcomes': dict(outcomes), 'timing_pairs': pair_count, 'comparison_count': len(groups),
             'native_loss_range_by_checkpoint': {c: max(v)-min(v) for c, v in native_losses.items()},
             'k050_regression': fit, 'controls': controls, 'nonmonotonic_checkpoint_pairs': inversions,
             'leave_one_checkpoint_out': {k: [min(r[k] for r in leave_one_out), max(r[k] for r in leave_one_out)]
                                          for k in ['slope', 'r_squared']},
             'source': record(RUN / 'results/matched-retrospective-001.json'), 'figure': figure,
             'agent_identity_source': record(REPO / 'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/prelaunch/controller.json', REPO),
             'limits': 'Local source/artifact audit and CPU tests; not a new GPU rerun, sanitizer run, '
                       'or proof of correctness for arbitrary inputs. Elementwise gates are reconstructed '
                       'from retained per-block statistics; full historical logits were not retained.'}
    write(dest, value)
    print({k: value[k] for k in ['checked_file_identities', 'process_outcomes', 'timing_pairs',
                                  'k050_regression', 'controls', 'nonmonotonic_checkpoint_pairs', 'leave_one_checkpoint_out']})


if __name__ == '__main__':
    main()
