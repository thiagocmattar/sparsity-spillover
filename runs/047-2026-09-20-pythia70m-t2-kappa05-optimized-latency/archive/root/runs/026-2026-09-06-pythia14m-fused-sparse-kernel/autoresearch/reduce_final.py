"""Audit frozen replications, summarize crossed process/input uncertainty, export every mode."""
import csv
import math
import numpy as np
from common import HERE, read_json, record, verify_record, write_json


def paired_cube(sessions, *, inputs=64, passes=7):
    cube = np.full((len(sessions), inputs, passes), np.nan)
    latencies = {'native': [], 'candidate': []}
    for process, session in enumerate(sessions):
        pairs = {}
        for row in session['samples']:
            key = (row['input_index'], row['repeat'])
            mode = row['mode']
            if mode not in latencies or not 0 <= key[0] < inputs or not 0 <= key[1] < passes:
                raise ValueError('Unexpected mode or input/repeat index')
            pair = pairs.setdefault(key, {})
            if mode in pair or not math.isfinite(row['host_ms']) or row['host_ms'] <= 0:
                raise ValueError('Duplicate or invalid timing')
            pair[mode] = row['host_ms']
            latencies[mode].append(row['host_ms'])
        for (index, repeat), pair in pairs.items():
            if set(pair) != set(latencies): raise ValueError('Unpaired timing')
            cube[process, index, repeat] = math.log(pair['native']/pair['candidate'])
    if not np.isfinite(cube).all(): raise ValueError('Missing timing cells')
    return cube, latencies


def crossed_interval(cube, *, draws=10000, seed=2606):
    # Preserve all seven repeats within each process/input cell. Processes and
    # shared input identities are crossed, not 1344 independent observations.
    cells = cube.mean(axis=2)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        process = rng.integers(cells.shape[0], size=cells.shape[0])
        inputs = rng.integers(cells.shape[1], size=cells.shape[1])
        estimates[i] = np.exp(cells[process[:, None], inputs[None, :]].mean())
    return np.quantile(estimates, [.025, .975]).tolist()


def summarize(attempts, frozen=None):
    sessions, losses, identities = [], [], []
    for attempt in attempts:
        dest = HERE/'artifacts'/attempt
        result, manifest, quality = (read_json(dest/name) for name in
            ['result.json', 'manifest.json', 'full-validation.json'])
        if not result['qualified'] or result['status'] != 'complete' or not quality['pass']['candidate']:
            raise ValueError('Final candidate failed qualification')
        if (quality['documents'], quality['blocks'], quality['input_tokens'], quality['excluded_tail']) != (500,338,692224,1444):
            raise ValueError('Incomplete validation')
        if manifest['cublas_workspace_config'] is not None or result['selected_dense'] != 'native':
            raise ValueError('Changed final comparator/workspace')
        for row in [manifest['config'], manifest['development'], manifest['validation']] + manifest['checkpoint']['files'] + manifest['checkpoint']['provenance']:
            verify_record(row)
        if frozen:
            sources = {row['path']: row for row in manifest['sources']}
            for path, expected in frozen['sources'].items():
                row = sources[record(HERE/path)['path']]
                if row['sha256'] != expected: raise ValueError('Unfrozen candidate source')
                verify_record(row)
            arguments = result['arguments']
            if arguments['implementation'] != 'k019' or not arguments['joint_with_rope'] or arguments['mode'] != 'native' or arguments['hoist_attention_import']:
                raise ValueError('Unfrozen configuration')
        sessions.append(read_json(dest/'final-timing.json'))
        losses.append({'attempt': attempt, 'loss': quality['loss'],
            'max_abs_logit_difference': max(row['max_abs'] for row in quality['gates']['candidate']),
            'max_relative_l2': max(row['relative_l2'] for row in quality['gates']['candidate'])})
        identities.extend(record(dest/name) for name in ['result.json','manifest.json','full-validation.json','final-timing.json'])
    if any(s['indices'] != sessions[0]['indices'] for s in sessions): raise ValueError('Changed timing cohort')
    expected = np.random.default_rng(2504).choice(338,64,replace=False).tolist()
    if sessions[0]['indices'] != expected: raise ValueError('Changed timing seed')
    cube, latencies = paired_cube(sessions)
    median = {mode: float(np.median(values)) for mode, values in latencies.items()}
    return {'attempts': attempts, 'processes': len(sessions), 'inputs': 64, 'passes': 7,
        'pairs': int(cube.size), 'paired_geomean_speedup': float(np.exp(cube.mean())),
        'process_speedups': np.exp(cube.mean(axis=(1,2))).tolist(),
        'speedup_ci95': crossed_interval(cube),
        'ci_method': '10000 crossed process/input percentile-bootstrap draws, seed2606; seven repeats retained within each cell; descriptive uncertainty, only three processes for winner and one for ablations',
        'median_host_ms': median, 'input_tokens_per_second_at_median': {k: 2048000/v for k,v in median.items()},
        'quality': losses, 'sources': identities}


def variant_rows():
    for path in sorted((HERE/'artifacts').glob('*/result.json')):
        result = read_json(path)
        setup = read_json(path.parent/'setup.json') if (path.parent/'setup.json').exists() else {}
        quality = read_json(path.parent/'full-validation.json') if (path.parent/'full-validation.json').exists() else {}
        final = read_json(path.parent/'final-timing.json') if (path.parent/'final-timing.json').exists() else None
        counts = result['canonical_logical_products']['measured']
        for scope, timings in [('development', result['timing'])] + ([('final_validation_timing',final['summary'])] if final else []):
            for mode in sorted(set(setup) | set(timings)):
                timing = timings.get(mode,{})
                valid = result.get('development_correctness',{}).get(mode,False) and quality.get('pass',{}).get(mode,False)
                yield {'iteration': result['attempt'], 'idea': result['idea'], 'implementation': result['arguments']['implementation'],
                    'mode': mode, 'candidate_mode': result['arguments']['mode'], 'timing_scope': scope,
                    'supported': setup.get(mode,{}).get('supported',False), 'qualified_full_validation': valid,
                    'latency_ms': timing.get('median_host_ms'), 'speedup_eager': timing.get('paired_geomean_speedup'),
                    'R_model': result['canonical_R_model'], 'R_model_scope': 'canonical source full validation, not recounted BF16',
                    'zero_products': counts['block_zero_product_count'], 'model_products': counts['model_product_count'],
                    'error': setup.get(mode,{}).get('error','') or ('' if valid else 'Numerical gate failed or incomplete qualification')}


def main():
    frozen = read_json(HERE/'frozen.json')
    winner = summarize(['015-final-r1','016-final-r2','017-final-r3'], frozen)
    result = {'winner': winner,
        'rope_only_dense_projections': summarize(['018-final-rope-ablation']),
        'joint_sparse_projections_only': summarize(['019-final-joint-ablation']),
        'canonical_logical_products': read_json(HERE/'artifacts/015-final-r1/result.json')['canonical_logical_products'],
        'frozen': record(HERE/'frozen.json'), 'reducer': record(__file__),
        'comparison_scope': 'Canonical stock eager SDPA; fastest qualified screened stock mode. Not a >1.6x claim over the custom dense QKV-fusion ablation or every possible optimized dense implementation.',
        'objective_met': winner['paired_geomean_speedup'] > 1.6 and winner['speedup_ci95'][0] > 1.6}
    write_json(HERE/'final-summary.json', result)
    rows = list(variant_rows())
    with (HERE/'all-variants.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print({name: {key: value[key] for key in ['paired_geomean_speedup','speedup_ci95','median_host_ms']}
           for name,value in result.items() if isinstance(value,dict) and 'median_host_ms' in value})
    print(f'{len(rows)} variant/scope rows; objective_met={result["objective_met"]}')


if __name__ == '__main__': main()
