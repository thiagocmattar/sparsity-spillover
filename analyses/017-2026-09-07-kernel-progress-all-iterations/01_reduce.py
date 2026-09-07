"""Pythia-14M history from K001 to K050, without hiding protocol boundaries."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R25 = ROOT / 'runs/025-2026-09-05-pythia-agentic-sparse-kernel-search'
R26 = ROOT / 'runs/026-2026-09-06-pythia14m-fused-sparse-kernel'
R27 = ROOT / 'runs/027-2026-09-06-pythia14m-kernel-sparsity-characterization'
R28 = ROOT / 'runs/028-2026-09-06-pythia14m-all-site-sparse-kernels'
EARLY = R25 / 'retrieved/rtxpro4500-002/artifacts'
PHASES = ['pro4500_eager', '5090_eager', '5090_graph']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def record(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(rec):
    if record(ROOT / rec['path']) != rec:
        raise ValueError(f"Evidence hash mismatch: {rec['path']}")


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')


def paired(samples, reference, target, inputs, passes):
    cells = {}
    for sample in samples:
        mode = sample['mode']
        if mode not in {reference, target}:
            continue
        key = sample['input_index'], sample['repeat']
        latency = sample['host_ms']
        if (not 0 <= key[0] < inputs or not 0 <= key[1] < passes
                or sample['output_shape'] != [1, 2048, 50304]
                or not math.isfinite(latency) or latency <= 0):
            raise ValueError('Invalid paired full-model sample')
        cell = cells.setdefault(key, {})
        if mode in cell:
            raise ValueError('Duplicate timing cell')
        cell[mode] = latency
    if len(cells) != inputs * passes or any(set(c) != {reference, target} for c in cells.values()):
        raise ValueError('Incomplete paired grid')
    return math.exp(math.fsum(math.log(c[reference] / c[target]) for c in cells.values()) / len(cells))


def numerical(quality, target):
    blocks = quality['blocks']
    if quality['prediction_tokens'] != blocks * 2047:
        raise ValueError('Numerical coverage mismatch')
    gates = quality['gates'][target]
    if ([g['input_index'] for g in gates] != list(range(blocks))
            or (all(g['pass'] for g in gates) and abs(quality['loss_delta'][target]) <= .001)
            != quality['pass'][target]):
        raise ValueError('Numerical status mismatch')
    return quality['pass'][target] and quality['pass']['native'], blocks


def early_point(folder, iteration, anchor_files):
    manifest = read(folder / 'manifest.json')
    args = manifest['arguments']
    if (not args['condition'].startswith('14m/')
            or manifest['gpu'] != 'NVIDIA RTX PRO 4500 Blackwell'):
        raise ValueError('Early scope mismatch')
    if args['condition'] == '14m/a7-0p5' and manifest['checkpoint']['files'] != anchor_files:
        raise ValueError('Fixed checkpoint identity changed')
    timing = read(folder / 'timing.json')
    target = 'k006' if iteration == 6 else 'candidate'
    summary = timing['matched']['eager']['summary'] if 'matched' in timing else timing['summary']
    raw_path = folder / timing['raw_samples']
    samples = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
    speedup = paired(samples, 'native', target, args['inputs'], args['passes'])
    if not math.isclose(speedup, summary[target]['paired_geomean_speedup'], rel_tol=1e-12):
        raise ValueError('Raw/stored speedup disagreement')
    quality_path = folder / ('full-validation.json' if (folder / 'full-validation.json').exists()
                             else 'development-quality.json')
    qualified, blocks = numerical(read(quality_path), target)
    return {'iteration': iteration, 'condition': args['condition'], 'phase': PHASES[0],
            'setting': args.get('mask', 'primary'), 'speedup': speedup, 'qualified': qualified,
            'validation_blocks': blocks if blocks == 338 else 0, 'checked_blocks': blocks,
            'inputs': args['inputs'], 'passes': args['passes'], 'processes': 1,
            'attempt': folder.name, 'anchor': args['condition'] == '14m/a7-0p5',
            'sources': [record(p) for p in [folder / 'manifest.json', folder / 'timing.json', raw_path, quality_path]]}


def choose_early(points):
    """Retain each distinct mask; supersede retries using coverage, never speed."""
    selected, superseded = {}, []
    def priority(p):
        retry = p['attempt'].startswith('k011retry')
        return (p['checked_blocks'], p['inputs'] * p['passes'], retry, p['attempt'])
    for p in points:
        key = p['iteration'], p['condition'], p['setting']
        old = selected.get(key)
        if old is None or priority(p) > priority(old):
            if old is not None:
                superseded.append(old)
            selected[key] = p
        else:
            superseded.append(p)
    return list(selected.values()), superseded


def incumbent(points, phase):
    best, trace = None, []
    for k in sorted({p['iteration'] for p in points if p['phase'] == phase}):
        for p in points:
            if (p['phase'] == phase and p['iteration'] == k and p['anchor']
                    and p['qualified'] and p['validation_blocks'] == 338):
                best = max(best or 0, p['speedup'])
        if best is not None:
            trace.append({'iteration': k, 'speedup': best})
    return trace


def main():
    pre27 = read(R27 / 'prelaunch/inputs.json')
    anchor_files = next(c['files'] for c in pre27['checkpoints'] if c['id'] == 'c30')
    for source in anchor_files:
        verify(source)
    early, audit = [], []
    for folder in sorted(EARLY.iterdir()):
        name = folder.name
        iteration = (1 if name.startswith('confirm-14m-') and '-k001-' in name else
                     6 if name.startswith('portfolio-14m-') else
                     11 if name.startswith(('k011search-14m-', 'k011confirm-14m-', 'k011full-14m-', 'k011retry-14m-')) else
                     12 if name.startswith('final-14m-') and '-k012-' in name else
                     13 if name.startswith('k013final-14m-') else None)
        if iteration is None:
            continue
        entry = {'attempt': name, 'iteration': iteration, 'full_model_timing': (folder / 'timing.json').exists()}
        if entry['full_model_timing']:
            early.append(early_point(folder, iteration, anchor_files))
        elif (folder / 'status.json').exists():
            entry['status_source'] = record(folder / 'status.json')
        audit.append(entry)
    points, superseded = choose_early(early)
    for k, attempt in [(17, '003-k017-eager'), (18, '006-k018-eager')]:
        folder = R26 / 'autoresearch/artifacts' / attempt
        result, manifest = read(folder / 'result.json'), read(folder / 'manifest.json')
        args = result['arguments']
        if (args['implementation'] != f'k{k:03d}' or args['mode'] != 'native'
                or manifest['gpu'] != 'NVIDIA GeForce RTX 5090'
                or manifest['checkpoint']['files'] != anchor_files):
            raise ValueError('Run026 scope mismatch')
        raw = folder / 'timing-samples.jsonl'
        speedup = paired([json.loads(line) for line in raw.read_text().splitlines()],
                         'native', 'candidate', args['inputs'], args['passes'])
        if not math.isclose(speedup, result['timing']['candidate']['paired_geomean_speedup'], rel_tol=1e-12):
            raise ValueError('Run026 raw/stored speedup mismatch')
        qualified, blocks = numerical(read(folder / 'full-validation.json'), 'candidate')
        points.append({'iteration': k, 'condition': 'c30', 'phase': PHASES[1], 'setting': 'primary',
                       'speedup': speedup, 'qualified': qualified, 'validation_blocks': blocks,
                       'checked_blocks': blocks, 'inputs': args['inputs'], 'passes': args['passes'],
                       'processes': 1, 'anchor': True, 'attempt': attempt,
                       'sources': [record(folder / f) for f in ['result.json', 'manifest.json', 'timing-samples.jsonl', 'full-validation.json']]})
    expected27 = read(R27 / 'results/figure-provenance.json')['source']
    verify(expected27)
    data27 = read(R27 / 'results/summary.json')
    if len(data27['rows']) != 35 or any(r['replicates'] != 3 or r['pairs'] != 1344 for r in data27['rows']):
        raise ValueError('Incomplete K019 cohort')
    for row in data27['rows']:
        points.append({'iteration': 19, 'condition': row['condition'], 'phase': PHASES[1],
                       'setting': 'frozen', 'speedup': row['overall_speedup'], 'qualified': row['overall_qualified'],
                       'validation_blocks': 338, 'checked_blocks': 338, 'inputs': 64, 'passes': 7,
                       'processes': 3, 'anchor': row['condition'] == 'c30',
                       'attempt': f"run027-{row['condition']}", 'sources': [expected27]})
    provenance28 = read(R28 / 'results/search-progress-figure-001.json')
    verify(provenance28['source'])
    data28 = read(R28 / 'results/search-progress-001.json')
    for p in data28['points']:
        if (p['iteration'] < 31 and p['execution'] == 'eager') or (p['iteration'] >= 31 and p['execution'] == 'graph'):
            for source in p['sources']:
                verify(source)
            points.append({**p, 'phase': PHASES[1] if p['execution'] == 'eager' else PHASES[2],
                           'setting': 'primary', 'anchor': p['condition'] == 'c30',
                           'checked_blocks': p['validation_blocks'] or p['development_quality_blocks']})
    points.sort(key=lambda p: (p['iteration'], p['condition'], p['setting']))
    missing = sorted(set(range(1, 51)) - {p['iteration'] for p in points})
    write(HERE / 'results/progress.json', {
        'scope': 'Pythia-14M; complete candidate-ID axis K001-K050; measured history only',
        'metric': 'Native / candidate paired-geomean full-model speedup; execution matched within each labeled phase',
        'points': points, 'missing_iterations': missing, 'early_attempt_audit': audit,
        'early_superseded_points': superseded,
        'best_fixed_checkpoint_by_phase': {phase: incumbent(points, phase) for phase in PHASES},
        'sources': [record(R27 / 'prelaunch/inputs.json'), expected27, provenance28['source']],
        'anchor_checkpoint_files': anchor_files, 'script': record(__file__),
        'selection': 'Run025 confirmed K001, K006 portfolio, every K011 mask (coverage-selected retries), final K012/K013; Run026 primary eager K017/K018; Run027 complete K019 cohort; Run028 primary eager until K030 and primary graphs from K031, using its frozen-cohort precedence.',
        'limits': 'Hardware, reference internals, execution, model cohort and timing coverage change. Incumbent resets at hardware/execution boundaries; no cross-phase causal gain or imputed missing speedup.',
    })
    print({'points': len(points), 'qualified': sum(p['qualified'] for p in points), 'missing_iterations': missing,
           'phase_counts': {phase: sum(p['phase'] == phase for p in points) for phase in PHASES}})


if __name__ == '__main__':
    (HERE / 'results').mkdir(exist_ok=True)
    main()
