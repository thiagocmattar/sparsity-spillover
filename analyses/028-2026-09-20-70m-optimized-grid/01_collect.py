"""Join canonical checkpoint evidence to the verified matched Run045 timings."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/045-2026-09-20-pythia70m-kernel-grid'
TRAIN = ROOT / 'runs/043-2026-09-20-pythia70m-hz-h-only-ol1'
OLD = ROOT / 'analyses/027-2026-09-20-run044-manuscript'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')


def collect(prepare_only=False):
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    previous = read(OLD / 'data/full-trained-results.json')
    inputs = read(RUN / 'provenance/inputs.json')
    verification = read(TRAIN / 'artifacts/verification.json')
    assert verification['status'] == 'verified' and verification['condition_count'] == 4
    full = copy.deepcopy(previous)
    old70 = {r['final_checkpoint_content_sha256']: r for r in full['trained_points'] if r['model'] == '70M'}
    assert len(old70) == 22
    by_id = {}
    for checkpoint in inputs['checkpoints']:
        digest = checkpoint['final_checkpoint_content_sha256']
        logical = checkpoint['canonical_logical_products']
        counts = logical['measured']
        assert counts['model_product_count'] == counts['block_product_count'] + counts['lm_head_product_count']
        assert sum(p['zero_product_count'] for p in counts['per_operation'].values()) == counts['block_zero_product_count']
        assert logical['coverage']['sequences'] == 338 and logical['coverage']['excluded_tail_tokens'] == 1444
        if digest in old70:
            row = old70[digest]
            assert row['family'] == checkpoint['family'] and row['kappa'] == checkpoint['dose']
            assert row['zero_product_count'] == counts['block_zero_product_count']
            assert row['model_product_count'] == counts['model_product_count']
        else:
            assert checkpoint['family'] == 'HZ+OL1@h'
            training, = [r for r in verification['conditions'] if r['checkpoint_content_sha256'] == digest]
            condition = training['condition']
            assert condition['active_sites'] == condition['one_sided_sites'] == ['h', 'z']
            assert condition['pressure_sites'] == ['h'] and condition['pressure_method'] == 'orthogonal_l1'
            assert condition['pressure_weight'] == condition['step_budget'] == 1
            assert condition['gate_threshold'] == checkpoint['dose']
            attempt = TRAIN / 'artifacts/attempts' / training['attempt_id']
            metrics = read(attempt / 'metrics.json')
            manifest = read(attempt / 'manifest.json')
            original_logical = read(attempt / 'diagnostics/logical_products.json')
            assert original_logical == logical
            assert manifest['checkpoints']['final']['content_sha256'] == digest
            assert manifest['seeds'] == {'model': 1234, 'data_order': 1234}
            assert metrics['training']['completed_steps'] == metrics['training']['optimizer_step_count'] == 712
            assert metrics['training']['input_tokens'] == 1493172224
            final = metrics['validation']['final']
            assert all(final[k] == v for k, v in previous['coverage'].items())
            assert final['loss'] == training['final_validation_loss']
            row = dict(model='70M', family=checkpoint['family'], scope='hz', pressure='h',
                       kappa=checkpoint['dose'], local_pressure_weight=None,
                       source_attempt=attempt.relative_to(ROOT).as_posix(), checkpoint_key=digest,
                       final_checkpoint_content_sha256=digest, loss=final['loss'],
                       sparsity=100*counts['block_zero_product_count']/counts['model_product_count'],
                       zero_product_count=counts['block_zero_product_count'], model_product_count=counts['model_product_count'],
                       coverage=previous['coverage'], initial_parameter_sha256=manifest['initial_parameter_sha256'],
                       training_schedule_hash=manifest['training_schedule_hash'], training_steps=712,
                       training_tokens=1493172224, logical_pass_loss=logical['coverage']['loss'])
            full['trained_points'].append(row)
            full['ceilings']['70M']['hz'] = logical['architecture_maximum']
        by_id[checkpoint['id']] = row
    assert len(full['trained_points']) == len({r['source_attempt'] for r in full['trained_points']}) == 83
    assert full['trained_points'][:79] == previous['trained_points']
    assert len(by_id) == 26
    # User-approved appendix-only endpoint: never mix its original-port timing
    # from another session into the 26-checkpoint optimized comparison.
    additional = read(ROOT / 'analyses/029-2026-09-20-70m-t2-figure/data/70m-quality-sparsity-native-latency.json')
    point, = [r for r in additional['added_t2_points'] if r['kappa'] == .5]
    for relative, digest in additional['sources_sha256'].items():
        if relative.startswith('runs/046-'):
            assert sha(ROOT / relative) == digest, relative
            sources[relative] = digest
    attempt = ROOT / 'runs' / point['training_run'] / 'artifacts/attempts' / point['training_attempt']
    metrics = read(attempt / 'metrics.json')
    manifest = read(attempt / 'manifest.json')
    logical = read(attempt / 'diagnostics/logical_products.json')
    assert manifest['checkpoints']['final']['content_sha256'] == point['checkpoint_key']
    assert metrics['validation']['final']['loss'] == point['loss']
    assert logical['measured'] == point['logical_counts']
    assert logical['coverage']['sequences'] == 338 and logical['coverage']['excluded_tail_tokens'] == 1444
    full['trained_points'].append(dict(point, family='HZ+OL1@h',
        source_attempt=attempt.relative_to(ROOT).as_posix(),
        final_checkpoint_content_sha256=point['checkpoint_key'],
        zero_product_count=logical['measured']['block_zero_product_count'],
        model_product_count=logical['measured']['model_product_count'],
        coverage=previous['coverage'], appendix_only=True,
        optimized_latency_ms=None, original_port_latency_ms=point['latency_ms'],
        timing_note='Run046 original port, separate session; optimized latency unmeasured.'))
    assert len(full['trained_points']) == 84
    full.update(sources_sha256=sources, title='Canonical 84-endpoint manuscript cohort',
                loss_convention='Ordinary final-checkpoint FP16 validation; logical-pass and BF16-check losses remain separate.',
                scope_note='45/27/12 endpoints at 14M/70M/410M. Main figures use 41 14M and 26 matched 70M endpoints; the additional 70M T2/Ph kappa=.5 is appendix-only.',
                observation='observations/001-matched-70m-grid.md')
    write(HERE / 'data/canonical-results.json', full)
    if prepare_only:
        print(json.dumps({'canonical_points':84, 'new_70m':[
            {k:r[k] for k in ('kappa','loss','logical_pass_loss','sparsity')}
            for r in by_id.values() if r['scope']=='hz']}))
        return

    local = read(RUN / 'results/local-verification.json')
    assert local['status'] == 'verified'
    matched = read(RUN / 'results/matched-grid.json')
    assert matched['status'] == 'complete' and matched['checkpoints'] == 26 and matched['fresh_processes'] == 78
    assert matched['timings_per_implementation_checkpoint'] == 1344
    # Preserve a failed qualification as missing publishable latency; never select
    # another backend to rescue an endpoint or silently discard its quality point.
    for result in matched['conditions']:
        row = by_id[result['id']]
        assert result['dose'] == row['kappa']
        assert result['canonical_logical_products']['measured']['block_zero_product_count'] == row['zero_product_count']
        if 'latency_ms' in row:
            row['historical_latency'] = {k:row[k] for k in ('latency_ms','timing_session','kernel')}
        row.update(run045_id=result['id'], latency_ms=result['latency_ms']['candidate_graph'] if result['qualified']['candidate_graph'] else None,
                   qualified=result['qualified'], implementation_latency_ms=result['latency_ms'],
                   native_base_speedup=result['native_base_speedup'], process_ranges_ms=result['process_geomean_ranges_ms'],
                   timing_session='Run045', timing_device_uuid=matched['device_uuid'], kernel='opt073',
                   displayed_latency_ms=matched['native_base_ms'] if result['id']=='c00' else
                       result['latency_ms']['candidate_graph'] if result['qualified']['candidate_graph'] else None)
    pairs = []
    for size in ('14M','70M'):
        for scope in ('4','7'):
            for dose in (0,.01,.05,.1,.5):
                pair = {r['pressure']:r for r in full['trained_points']
                        if (r['model'],r['scope'],r['kappa'])==(size,scope,dose) and r['pressure'] in ('all','h')}
                a,h = pair['all'],pair['h']
                pairs.append(dict(model=size,scope=scope,kappa=dose,
                    loss=a['loss']-h['loss'],sparsity=a['sparsity']-h['sparsity'],
                    latency_us=1000*(a['latency_ms']-h['latency_ms']) if a['latency_ms'] and h['latency_ms'] else None,
                    treatment_key=a['checkpoint_key'],reference_key=h['checkpoint_key']))
    summary = dict(native_base_ms=matched['native_base_ms'],
                   qualified=matched['all_qualified'],
                   faster_local_pairs={size:sum(p['latency_us'] is not None and p['latency_us']>0
                     for p in pairs if p['model']==size) for size in ('14M','70M')},
                   before_after=[{k:r[k] for k in ('run045_id','scope','pressure','kappa','loss','sparsity',
                       'implementation_latency_ms','native_base_speedup','qualified')} for r in by_id.values()])
    full.update(sources_sha256=sources, paired_pressure=pairs, matched_summary=summary,
                timing_note='14M historical K050 timings retained. All 26 70M endpoints use Run045 paired native/original-port/opt073 measurements; the Figure 3 Base marker uses native execution.')
    # Legacy summaries encode a different backend/reference; do not carry stale headlines forward.
    full.pop('high_threshold_speedups',None)
    write(HERE / 'data/full-trained-results.json',full)
    write(HERE / 'data/summary.json',summary)
    write(HERE / 'data/70m-quality-sparsity-native-latency.json',dict(
        trained_points=[r for r in full['trained_points'] if 'run045_id' in r],
        clipping_points=[r for r in full['clipping_points'] if r['model']=='70M'],
        ceilings=full['ceilings']['70M'],native_base_ms=matched['native_base_ms'],
        source='data/full-trained-results.json',source_sha256=sha(HERE/'data/full-trained-results.json'),
        base_marker_backend='native_graph',other_markers_backend='candidate_graph',
        observation='observations/001-matched-70m-grid.md'))
    write(HERE / 'data/paired-pressure-figure-data.json',dict(
        pairs=pairs,source='data/full-trained-results.json',
        source_sha256=sha(HERE/'data/full-trained-results.json'),
        observation='observations/001-matched-70m-grid.md'))
    print(json.dumps({'all_qualified':matched['all_qualified'], 'native_base_ms':matched['native_base_ms'],
                      'faster_local_pairs':summary['faster_local_pairs']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    collect(parser.parse_args().prepare_only)
