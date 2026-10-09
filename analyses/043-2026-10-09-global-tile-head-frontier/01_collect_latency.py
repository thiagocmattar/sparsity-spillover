"""Collect existing coordinates and measured boundaries without pooling sessions."""
import math
from statistics import geometric_mean
from evidence import HERE, ROOT, SOURCES, identity, read, run, verify, write, frontier

A37 = 'analyses/037-2026-09-24-14m-31m-70m-latency-quality/data/combined-figure.json'
A38 = 'analyses/038-2026-09-24-base-t2-t7-scale-frontier/data/scale-figure.json'
A41 = 'analyses/041-2026-10-04-corner-tile-versus-elementwise/data/'
A42 = 'analyses/042-2026-10-08-model-latency-decomposition/'


def canonical(path):
    metrics = read(path / 'metrics.json')
    manifest = read(path / 'manifest.json')
    assert manifest['status'] == 'completed' and manifest['completed_steps'] == 712
    values = metrics['validation']
    final = values.get('step_000712', values.get('final'))
    assert final['complete_block_coverage'] and final['sequences'] == 338
    assert final['input_tokens'] == 692224 and final['excluded_tail_tokens'] == 1444
    return final['loss'], manifest


def check_declared(mapping):
    for path, digest in mapping.items():
        verify({'path': path.replace('\\', '/'), 'sha256': digest})


def group(scope):
    return 'base' if scope == '0' else 'scalar_t2' if scope == 'hz' else 'scalar_t7' if scope == '7' else 'scalar_other'


def historical():
    previous = read(A37)
    check_declared(previous['sources_sha256'])
    points = []
    for model, rows in previous['cohorts'].items():
        for row in rows:
            if row['scope'] == '0':
                continue
            points.append(dict(row, id=f"historical-{model}-{row['checkpoint_key']}",
                latency_ms=row[previous['latency_fields'][model]], group=group(row['scope']),
                gate_rule='scalar', head_gate='off', head_kappa=None, block_kappa=row.get('kappa', row.get('dose')),
                execution=row.get('kernel', 'historical kernel'), session=row.get('timing_session', 'historical'),
                execution_cohort='BF16 block-kernel cohort', latency_precision='BF16', quality_precision='canonical FP16',
                source_coordinate_file=A37, control=False))
    for row in previous['base_references']:
        source = next(p for p in previous['cohorts'][row['model']] if p['checkpoint_key'] == row['checkpoint_key'])
        points.append(dict(source, **row, id=f"base-{row['model']}-{row['backend']}",
            group='base', head_gate='off', head_kappa=None, block_kappa=None, gate_rule='none',
            execution=row['backend'], session=row['timing_session'], execution_cohort='BF16 block-kernel cohort',
            latency_precision='BF16', quality_precision='canonical FP16', source_coordinate_file=A37, control=False))
    assert len(points) == 75
    extended = read(A38)
    check_declared(extended['source_sha256'])
    for row in extended['executions']:
        if row['model'] == '31M' and row['scope'] == '7':
            points.append(dict(row, id=f"historical-31M-{row['checkpoint_key']}", group='scalar_t7',
                head_gate='off', head_kappa=None, block_kappa=row['kappa'], gate_rule='scalar',
                execution='31M kernel', session='Run055', execution_cohort='BF16 block-kernel cohort',
                latency_precision='BF16', quality_precision='canonical FP16', source_coordinate_file=A38, control=False))
    assert len(points) == 80
    for file, keys in [(A41+'quality-latency-frontier.json', ['new_tile_points', 'same_session_scalar_references']),
                       (A41+'t7-quality-latency-frontier.json', ['t7_tile_points', 'matched_t7_scalar_references'])]:
        data = read(file)
        for source in data['source_files']:
            verify(source)
        for key in keys:
            for row in data[key]:
                tile = 'tile' in key and 'scalar' not in key
                session = 'Run062' if 't7-' in file else 'Run061'
                points.append(dict(row, id=f"{session}-{row['condition']}-sparse", group='tile_t7' if tile and row['scope']=='7' else 'tile_t2' if tile else group(row['scope']),
                    gate_rule='corner 8x16' if tile else 'scalar', head_gate='off', head_kappa=None,
                    block_kappa=row['kappa'], execution='sparse', session=session,
                    execution_cohort='BF16 block-kernel cohort', latency_precision='BF16',
                    quality_precision='canonical FP16', source_coordinate_file=file, control=False))
    assert len(points) == 88
    return points


def specialized(points):
    r = run('063')
    data = read(r / 'results/results.json')
    assert data['status'] == 'verified'
    for pair in data['pairs']:
        assert pair['qualified'] and pair['paired_samples_per_comparison'] == 1344
        for row in pair['source_files']:
            verify(row, r)
        quality = next(p for p in points if p['session']=='Run061' and p['group']=='tile_t2' and p['kappa']==pair['kappa'])
        process = {}
        for item in pair['processes']:
            dest = r / 'artifacts/attempts' / item['attempt']
            result = read(dest / 'result.json')
            assert result['qualified'] and result['status']=='complete'
            samples = read(dest / 'timing.json')['samples']
            for mode in pair['host_geomean_ms']:
                values = [x['host_ms'] for x in samples if x['mode']==mode]
                assert len(values)==448
                process.setdefault(mode, []).append(geometric_mean(values))
        for mode, latency in pair['host_geomean_ms'].items():
            assert math.isclose(latency, geometric_mean(process[mode]), rel_tol=1e-12)
            points.append(dict(quality, id=f"Run063-{quality['condition']}-{mode}", execution=mode,
                latency_ms=latency, process_latency_ms=process[mode], session='Run063',
                source_coordinate_file=(r/'results/results.json').relative_to(ROOT).as_posix(),
                control='no_skip' in mode))


def profiles(points):
    data = read(A42+'results/decomposition.json')
    assert data['status']=='verified'
    for row in read(A42+'provenance/sources.json')['sources']:
        verify(row)
    for condition in data['conditions']:
        model, kappa = condition['model'], float(condition['kappa'])
        if model == '14M':
            source = next(p for p in points if p['session']=='Run061' and p['group']=='tile_t2' and p['kappa']==kappa)
            loss, manifest = canonical(PathLike(source['source_attempt']))
        else:
            meta = next(p for p in data['checkpoint_sources']['31M/70M']
                        if p['size']==model.lower() and float(p['pair'].replace('p','.'))==kappa)
            attempt = ROOT/'runs'/meta['source_run']/'artifacts/attempts'/meta['source_attempt']
            loss, manifest = canonical(attempt)
            source = dict(model=model, kappa=kappa, scope='hz', pressure='h',
                          source_attempt=attempt.relative_to(ROOT).as_posix(), gate_rule='scalar', group='scalar_t2')
        assert manifest['condition']['gate_threshold'] == kappa
        for mode, values in condition['modes'].items():
            latency = values['host']['geomean']/1000
            processes = [v/1000 for v in values['process_host_geomean_us']]
            assert len(processes)==3 and math.isclose(latency, geometric_mean(processes), rel_tol=1e-12)
            points.append(dict(source, id=f"Run{condition['source_run'][:3]}-{condition['key']}-{mode}",
                loss=loss, latency_ms=latency, process_latency_ms=processes, execution=mode,
                session='Run'+condition['source_run'][:3], block_kappa=kappa, head_gate='off', head_kappa=None,
                execution_cohort='BF16 block-kernel cohort', latency_precision='BF16', quality_precision='canonical FP16',
                source_coordinate_file=A42+'results/decomposition.json', control='no_skip' in mode,
                profiled_condition=condition['key']))
    return data


def PathLike(name):
    return ROOT/name


def heads(points):
    boundaries=[]
    for training, benchmark in [('066','067'), ('068','068')]:
        r = run(training)
        b = run(benchmark)/('inference' if benchmark=='068' else '')
        train = read(r/('results/training-comparison.json' if training=='068' else 'results/comparison.json'))
        timing = read(b/'results/comparison.json')
        assert read(b/'transfer/receipt-001.json')['status']=='passed'
        verified = read(b/'artifacts/verification-001.json')
        assert verified['status']=='passed' and verified['count']==len(timing['rows'])*3
        for row in timing['sources']:
            verify(row)
        for t in timing['rows']:
            q = next(x for x in train['rows'] if x['id']==t['id'])
            assert q['validation']['sequences']==338 and q['validation']['complete_block_coverage']
            loss, manifest = canonical(r/q['source'])
            assert loss == q['loss']
            block = manifest['condition'].get('block_gate', {}).get('kappa', t['kappa'])
            head_tile = 16 if training=='068' else 8
            descriptor = dict(model='14M', scope='hz', pressure='h+head' if len(t['pressure_sites'])==2 else 'h',
                head_kappa=t['kappa'], kappa=t['kappa'], block_kappa=block, group=f'head{head_tile}',
                gate_rule='corner 8x16 at h/z', head_gate=f'corner {head_tile}x16', head_tile_tokens=head_tile,
                loss=loss, source_attempt=(r/q['source']).relative_to(ROOT).as_posix(), session='Run'+benchmark,
                execution_cohort='FP16 native-body/head cohort', latency_precision='FP16', quality_precision='canonical FP16',
                source_coordinate_file=(b/'results/comparison.json').relative_to(ROOT).as_posix())
            for mode, latency in t['timings']['full_forward']['host_ms']['geomean_ms'].items():
                process = [p['timings']['full_forward']['host_ms']['geomean_ms'][mode] for p in t['processes']]
                assert len(process)==3 and math.isclose(latency, geometric_mean(process), rel_tol=1e-12)
                points.append(dict(descriptor, id=f"Run{benchmark}-{t['id']}-{mode}", execution=mode,
                    latency_ms=latency, process_latency_ms=process, control=mode=='custom_no_skip',
                    head_arithmetic='sparse' if mode=='custom_sparse' else 'dense', body_arithmetic='native dense'))
            boundaries.append(dict(descriptor, condition=t['id'], timings=t['timings'], processes=t['processes'],
                head_counts=t['head_counts'], native_loss_b1=t['processes'][0]['loss']['native'],
                max_logit_error=max(p['max_logit_error'] for p in t['processes']),
                head_exact_zero_fraction=q['head_exact_zero_fraction']))
    return boundaries


def main():
    points=historical()
    specialized(points)
    decomposition=profiles(points)
    head=heads(points)
    assert len(points)==len({p['id'] for p in points})==138
    assert all(math.isfinite(p['loss']) and math.isfinite(p['latency_ms']) and p['latency_ms']>0 for p in points)
    edge=frontier(points)
    cohort_edges={cohort:[p['id'] for p in frontier([q for q in points if q['execution_cohort']==cohort])]
                  for cohort in sorted({p['execution_cohort'] for p in points})}
    dominance=[dict(id=p['id'], nondominated=p in edge,
        dominators=[q['id'] for q in points if q['loss']<=p['loss'] and q['latency_ms']<=p['latency_ms']
                    and (q['loss']<p['loss'] or q['latency_ms']<p['latency_ms'])]) for p in points if p['group'].startswith('head')]
    write('data/latency.json', dict(points=points, global_frontier_ids=[p['id'] for p in edge],
        compatible_frontier_ids=cohort_edges, head_dominance=dominance, head_boundaries=head,
        decomposition=decomposition, counts=dict(points=len(points), global_frontier=len(edge),
            head_checkpoints=len(head), head_executions=len(dominance)),
        coverage=dict(documents=500, blocks=338, input_tokens=692224, predictions=691886, excluded_tail_tokens=1444),
        caveat='Observed point-estimate envelope across different GPU sessions, kernel packages and precisions; not a controlled causal frontier.'))
    identity(__file__)
    write('provenance/latency-sources.json', list(SOURCES.values()))
    print({'status':'collected', 'points':len(points), 'frontier':len(edge), 'head_checkpoints':len(head)}, flush=True)


if __name__=='__main__':
    main()
