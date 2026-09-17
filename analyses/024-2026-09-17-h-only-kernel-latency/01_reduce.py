"""Combine retained Run029 K050 pairs with only the five new Run033 models."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'
NEW = ROOT/'runs/033-2026-09-17-a7-h-only-k050-benchmark'
SOURCES = {}


def read(path):
    raw = path.read_bytes()
    SOURCES[path.relative_to(ROOT).as_posix()] = {
        'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)


def gm(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def paired_values(samples):
    modes = {'native_graph':{}, 'candidate_graph':{}}
    for sample in samples:
        key = sample['repeat'], sample['input_index']
        assert sample['mode'] in modes and key not in modes[sample['mode']]
        assert sample['output_shape'] == [1, 2048, 50304]
        modes[sample['mode']][key] = sample['host_ms']
    expected = {(r, i) for r in range(7) for i in range(64)}
    assert all(set(rows) == expected for rows in modes.values())
    native = [modes['native_graph'][key] for key in sorted(expected)]
    candidate = [modes['candidate_graph'][key] for key in sorted(expected)]
    return native, candidate, [n/c for n, c in zip(native, candidate)]


def point(run, checkpoint, attempts, session):
    native, candidate, ratios, replicates = [], [], [], []
    uuids, timing_indices = set(), set()
    for attempt in attempts:
        folder = run/'artifacts/attempts'/attempt
        result, quality, timing = [read(folder/name) for name in
                                   ['result.json', 'quality.json', 'timing.json']]
        args = result['arguments']
        assert result['status'] == 'complete' and not args['smoke']
        assert result['candidate'] == 'k050' and result['condition'] == checkpoint['id']
        assert quality['blocks'] == 338 and quality['documents'] == 500
        assert quality['excluded_tail_tokens'] == 1444
        assert quality['prediction_tokens'] == 691886
        assert result['qualification'] == quality['pass']
        assert result['qualified'] == all(quality['pass'].values())
        assert all(math.isfinite(v) for v in quality['loss'].values())
        n, c, pairs = paired_values(timing['samples'])
        assert math.isclose(gm(pairs), timing['summary']['candidate_graph']['paired_geomean_speedup'], rel_tol=1e-12)
        native.extend(n); candidate.extend(c); ratios.extend(pairs)
        runtime = result['runtime']
        assert runtime['gpu'] == 'NVIDIA GeForce RTX 5090'
        for key, value in [('cuda','12.8'), ('torch','2.11.0+cu128'), ('transformers','5.12.1'), ('numpy','2.5.0')]:
            assert runtime[key] == value
        uuids.add(runtime['device_uuid'])
        timing_indices.add(tuple(timing['indices']))
        replicates.append({'attempt':attempt, 'replicate':args['replicate'],
                           'qualified':result['qualified'], 'speedup':gm(pairs),
                           'native_gm_ms':gm(n), 'k050_gm_ms':gm(c),
                           'loss_bf16':quality['loss'], 'loss_delta':quality['loss_delta'],
                           'peak_allocated_bytes':result['peak_allocated_bytes'],
                           'elapsed_seconds':result['elapsed_seconds'],
                           'runtime':runtime})
    assert sorted(r['replicate'] for r in replicates) == [1, 2, 3]
    assert len(uuids) == len(timing_indices) == 1 and len(ratios) == 1344
    logical = checkpoint['canonical_logical_products']
    assert logical['coverage']['sequences'] == 338
    counts = logical['measured']
    zeros, total = counts['block_zero_product_count'], counts['model_product_count']
    assert zeros == sum(op['zero_product_count'] for op in counts['per_operation'].values())
    assert total == counts['block_product_count'] + counts['lm_head_product_count']
    assert math.isclose(zeros/total, counts['R_model'], abs_tol=1e-15)
    return {'condition':checkpoint['id'], 'family':checkpoint['family'], 'kappa_or_lambda':checkpoint['dose'],
            'session':session, 'gpu_uuid':next(iter(uuids)), 'timing_block_indices':list(next(iter(timing_indices))),
            'S_model':zeros/total, 'sparsity_percent':100*zeros/total, 'canonical_counts':counts,
            'checkpoint_files':checkpoint['original_files'], 'qualified':all(r['qualified'] for r in replicates),
            'speedup':gm(ratios), 'native_gm_ms':gm(native), 'k050_gm_ms':gm(candidate),
            'process_speedup_min':min(r['speedup'] for r in replicates),
            'process_speedup_max':max(r['speedup'] for r in replicates), 'replicates':replicates}


def collect_historical():
    summary = read(OLD/'results/matched-retrospective-001.json')
    inputs = read(OLD/'provenance/inputs.json')['checkpoints']
    selected = {p['condition']:p for p in summary['points'] if p['candidate']=='k050' and p['phase']=='final'}
    assert len(selected) == len(inputs) == 35
    rows = []
    for checkpoint in inputs:
        original = selected[checkpoint['id']]
        row = point(OLD, checkpoint, [r['attempt'] for r in original['replicates']], 'Run029')
        assert row['qualified'] and math.isclose(row['speedup'], original['speedup'], rel_tol=1e-12)
        rows.append(row)
    # Validate all re-read old measurement files against the preserved reduction inventory.
    for source in summary['sources']:
        name = (OLD/source['path']).relative_to(ROOT).as_posix()
        if name in SOURCES:
            assert SOURCES[name] == {k:source[k] for k in ['bytes','sha256']}
    return rows


def write_table(points):
    lines = ['# Final K050 latency and speedup', '',
             'BF16, B=1, T=2048, full vocabulary; three processes and 1,344 pairs per checkpoint.',
             'S_model uses the retained pooled FP16 logical-product counts. Latencies are geometric means of raw host timings.',
             'Speedup is the geometric mean of paired native/K050 ratios; range is across the three process means, not a confidence interval.',
             'Run029 measurements are reused; only Run033 is new. Absolute latencies span two physical RTX 5090 GPUs.', '',
             '| Model | κ or λ | S_model (%) | Native (ms) | K050 (ms) | Speedup | Process range | Session | Qualified |',
             '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for p in points:
        dose = '—' if p['kappa_or_lambda'] is None else f"{p['kappa_or_lambda']:g}"
        lines.append(f"| {p['family']} | {dose} | {p['sparsity_percent']:.6f} | {p['native_gm_ms']:.6f} | {p['k050_gm_ms']:.6f} | {p['speedup']:.4f}× | {p['process_speedup_min']:.4f}–{p['process_speedup_max']:.4f} | {p['session']} | {p['qualified']} |")
    (HERE/'TABLE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')


def main():
    points = collect_historical()
    for checkpoint in read(NEW/'provenance/inputs.json')['checkpoints']:
        points.append(point(NEW, checkpoint,
            [f"scientific-{checkpoint['id']}-r{r}-001" for r in [1,2,3]], 'Run033'))
    assert len(points) == 40 and len({p['condition'] for p in points}) == 40
    assert len({tuple(p['timing_block_indices']) for p in points}) == 1
    data = {'question':'How do the new A7 h-only checkpoints compare under the frozen final K050?',
            'protocol':{'gpu_model':'RTX 5090','precision':'BF16','sparsity_precision':'FP16',
                        'batch':1,'sequence_length':2048,'vocabulary':50304,'validation_blocks':338,
                        'validation_documents':500,'excluded_tail_tokens':1444,'timing_inputs':64,
                        'passes_per_process':7,'processes':3,'paired_samples_per_checkpoint':1344},
            'speedup_definition':'Geometric mean of raw paired same-checkpoint native_graph / candidate_graph host latencies.',
            'latency_definition':'Geometric mean of all 1,344 raw host-latency samples per implementation; not the older geometric mean of process medians.',
            'cross_session_limit':'Same GPU model, frozen kernel and timing protocol; different GPU/host sessions may shift absolute latencies and ratios. No new A0 or skip ablations.',
            'points':points, 'sources':SOURCES}
    (HERE/'data').mkdir(exist_ok=True)
    (HERE/'data/results.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    write_table(points)
    print(f"40 checkpoints, {sum(p['qualified'] for p in points)} qualified; 5 newly benchmarked")


if __name__ == '__main__':main()
