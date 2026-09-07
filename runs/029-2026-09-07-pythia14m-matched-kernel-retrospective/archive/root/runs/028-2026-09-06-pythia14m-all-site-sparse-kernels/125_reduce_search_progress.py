"""Audit Run028 candidate-level progress without mixing execution baselines."""
import math
import re

import numpy as np

from common import RUN, read_json, record, verify_record, write_json

PRIMARY = re.compile(r'^k(\d{3})-(c\d{2})-(dev|validation|attribution|graph)-(\d{3})$')


def paired_speedup(samples, reference, target, inputs, passes):
    cells = {}
    for sample in samples:
        if sample['mode'] not in {reference, target}:
            continue
        key = sample['input_index'], sample['repeat']
        if (not 0 <= key[0] < inputs or not 0 <= key[1] < passes
                or sample['output_shape'] != [1, 2048, 50304]
                or not math.isfinite(sample['host_ms']) or sample['host_ms'] <= 0):
            raise ValueError('Timing identity/workload mismatch')
        cell = cells.setdefault(key, {})
        if sample['mode'] in cell:
            raise ValueError('Duplicate timing cell')
        cell[sample['mode']] = sample['host_ms']
    if len(cells) != inputs * passes or any(set(c) != {reference, target} for c in cells.values()):
        raise ValueError('Incomplete paired timing grid')
    return math.exp(np.mean([math.log(c[reference] / c[target]) for c in cells.values()]))


def development_point(folder, result):
    match = PRIMARY.fullmatch(folder.name)
    iteration, condition, _, retry = match.groups()
    iteration = int(iteration)
    args = result['arguments']
    if args['candidate'] != f'k{iteration:03d}' or args['condition'] != condition:
        raise ValueError('Candidate/checkpoint identity mismatch')
    manifest = read_json(folder / 'manifest.json')
    if (manifest['gpu'] != 'NVIDIA GeForce RTX 5090'
            or manifest['torch'] != '2.11.0+cu128' or manifest['cuda'] != '12.8'
            or manifest['checkpoint']['id'] != condition):
        raise ValueError('Runtime or checkpoint changed')
    if 'native_graph' in result['timing']:
        execution, reference, target = 'graph', 'native_graph', 'selected_graph'
        reported = result['timing'][reference][target]['paired_geomean_speedup']
    else:
        execution = 'graph' if args.get('execution') == 'graph' else 'eager'
        reference = 'native'
        target = 'candidate' if 'candidate' in result['timing'] else 'sparse'
        if iteration == 36:
            target = 'no_prefix'  # Uniform policy subsequently frozen for K036.
        reported = result['timing'][target]['paired_geomean_speedup']
    inputs, passes = args.get('inputs', 32), args.get('passes', 7)
    timing = read_json(folder / 'timing.json')
    speedup = paired_speedup(timing['samples'], reference, target, inputs, passes)
    if not math.isclose(speedup, reported, rel_tol=1e-10):
        raise ValueError('Stored speedup differs from raw paired samples')
    quality = read_json(folder / 'quality.json')
    validation_blocks = result.get('validation_blocks', quality.get('blocks', 0))
    checked_blocks = validation_blocks or result['development_quality_blocks']
    if quality['prediction_tokens'] != checked_blocks * 2047:
        raise ValueError('Numerical coverage mismatch')
    quality_modes = [reference, target]
    if execution == 'graph' and 'eager_stock' in result['qualified']:
        quality_modes.append('eager_stock')
    for mode in quality_modes:
        gates = quality['gates'].get(mode, [])
        if mode == target or gates:
            if [g['input_index'] for g in gates] != list(range(checked_blocks)):
                raise ValueError('Missing numerical block')
            passed = all(g['pass'] for g in gates) and abs(quality['loss_delta'][mode]) <= 0.001
            if passed != quality['pass'][mode]:
                raise ValueError('Numerical qualification mismatch')
        if result['qualified'][mode] != quality['pass'][mode]:
            raise ValueError('Result/quality qualification mismatch')
    return {
        'iteration': iteration, 'candidate': args['candidate'], 'condition': condition,
        'execution': execution, 'reference': reference, 'target': target,
        'speedup': speedup, 'qualified': all(quality['pass'][m] for m in quality_modes),
        'qualification_modes': quality_modes, 'validation_blocks': validation_blocks,
        'development_quality_blocks': 0 if validation_blocks else checked_blocks,
        'stage': 'development', 'inputs': inputs, 'passes': passes, 'processes': 1,
        'retry': int(retry), 'attempt': folder.name,
        'sources': [record(folder / name) for name in ['result.json', 'manifest.json', 'quality.json', 'timing.json']],
        'note': ('K044 executed a confounded K033-base composition, not its intended K036 attention-only change.'
                 if iteration == 44 else ''),
    }


def choose_points(points):
    """Richest validation/timing, then retry ID; never rank by achieved speedup."""
    selected, superseded = {}, []
    def priority(p):
        return (p['stage'] == 'final', p['validation_blocks'],
                p['inputs'] * p['passes'] * p['processes'], p['retry'])
    for point in sorted(points, key=lambda p: p['attempt']):
        key = point['iteration'], point['execution'], point['condition']
        previous = selected.get(key)
        if previous is None or priority(point) > priority(previous):
            if previous is not None:
                superseded.append(previous)
            selected[key] = point
        else:
            superseded.append(point)
    return sorted(selected.values(), key=lambda p: (p['execution'], p['iteration'], p['condition'])), superseded


def fixed_checkpoint_best(points, execution, condition='c30'):
    series, best = [], None
    for iteration in sorted({p['iteration'] for p in points if p['execution'] == execution}):
        candidates = [p for p in points if p['execution'] == execution and p['iteration'] == iteration
                      and p['condition'] == condition and p['qualified'] and p['validation_blocks'] == 338]
        for point in candidates:
            best = max(best or 0, point['speedup'])
        if best is not None:
            series.append({'iteration': iteration, 'speedup': best})
    return series


def main():
    points, audit = [], []
    for folder in sorted((RUN / 'artifacts').iterdir()):
        if not re.match(r'^k\d{3}-c\d{2}', folder.name) or not (folder / 'result.json').exists():
            continue
        result = read_json(folder / 'result.json')
        entry = {'attempt': folder.name, 'result': record(folder / 'result.json'), 'status': result['status']}
        if not PRIMARY.fullmatch(folder.name):
            entry['disposition'] = 'Separate control, alternate setting, or precision diagnostic; not primary candidate policy'
        elif result['status'] != 'complete' or not result.get('timing'):
            entry['disposition'] = 'No complete full-model timing; never imputed as zero'
        else:
            point = development_point(folder, result)
            points.append(point)
            entry['disposition'] = 'Audited primary development point; subject to non-performance selection precedence'
        audit.append(entry)
    for iteration, filename, provenance in [(36, 'summary.json', 'figure-provenance.json'),
                                             (50, 'summary-002.json', 'figure-provenance-002.json')]:
        expected = read_json(RUN / 'results' / provenance)['source']
        verify_record(expected)
        data = read_json(RUN / 'results' / filename)
        verify_record(data['policy'])
        if len(data['rows']) != 35 or not all(r['complete'] for r in data['rows']):
            raise ValueError('Incomplete frozen final cohort')
        for row in data['rows']:
            for execution in ['eager', 'graph']:
                comp = row['comparisons'][execution]
                if len(comp['process_ratios']) != 3 or not math.isclose(
                        comp['ratio'], math.exp(np.log(comp['process_ratios']).mean()), rel_tol=1e-12):
                    raise ValueError('Final process aggregation mismatch')
                points.append({
                    'iteration': iteration, 'candidate': f'k{iteration:03d}', 'condition': row['condition'],
                    'execution': execution, 'reference': 'native' if execution == 'eager' else 'native_graph',
                    'target': 'sparse' if execution == 'eager' else 'sparse_graph',
                    'speedup': comp['ratio'], 'qualified': comp['qualified'], 'validation_blocks': 338,
                    'development_quality_blocks': 0, 'stage': 'final', 'inputs': 64, 'passes': 7,
                    'processes': 3, 'retry': 0, 'attempt': f'final-k{iteration:03d}-{row["condition"]}',
                    'sources': [record(RUN / 'results' / filename), data['policy']], 'note': '',
                })
    selected, superseded = choose_points(points)
    if {p['iteration'] for p in selected} != set(range(20, 51)):
        raise ValueError('Missing candidate iteration in Run028 history')
    output = {'scope': 'Run028, K020-K050; Pythia-14M only; RTX 5090',
              'points': selected, 'superseded_points': superseded, 'attempt_audit': audit,
              'best_fixed_c30': {mode: fixed_checkpoint_best(selected, mode) for mode in ['eager', 'graph']},
              'selection': 'Final cohort supersedes development at K036/K050; otherwise richest validation/timing, then retry ID. Never choose by speed or qualification.',
              'distribution': 'One point per candidate/execution/checkpoint, with numerical status retained. Figure06 shows graph points only, without bars or boxes. Between-model spread is not timing uncertainty.',
              'script': record(__file__)}
    write_json(RUN / 'results/search-progress-001.json', output)
    for mode in ['eager', 'graph']:
        print(mode, [{'k': k, 'n': len(group), 'qualified': sum(p['qualified'] for p in group),
                      'minimum': min(p['speedup'] for p in group), 'maximum': max(p['speedup'] for p in group)}
                     for k in sorted({p['iteration'] for p in selected if p['execution'] == mode})
                     if (group := [p for p in selected if p['execution'] == mode and p['iteration'] == k])])


if __name__ == '__main__':
    main()
