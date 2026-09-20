"""Audit complete paired coverage and retain a single publication latency estimand."""
import argparse
import math
from statistics import median
from io_utils import RUN, read, write, record, verify

MODES = ('native_graph', 'legacy_graph', 'candidate_graph')


def geomean(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Finite positive timings required')
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def audit_samples(timing, inputs, passes):
    rows = timing['samples']
    if len(timing['indices']) != inputs or len(set(timing['indices'])) != inputs:
        raise ValueError('Timing input identities are not unique')
    expected = {(mode, index, repeat) for mode in MODES for index in range(inputs) for repeat in range(passes)}
    actual = [(row['mode'], row['input_index'], row['repeat']) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate paired timing cells')
    for row in rows:
        if row['output_shape'] != [1, 2048, 50304]:
            raise ValueError('Every observation must produce the full vocabulary logits')
    return {mode: [row['host_ms'] for row in rows if row['mode'] == mode] for mode in MODES}


def audit_quality(quality, bounds):
    coverage = {key: quality[key] for key in ('blocks', 'documents', 'input_tokens', 'prediction_tokens', 'excluded_tail_tokens')}
    if coverage != dict(blocks=338, documents=500, input_tokens=692224, prediction_tokens=691886, excluded_tail_tokens=1444):
        raise ValueError('Incomplete validation coverage or incorrect pooled loss denominator')
    passed = {}
    for mode in MODES:
        gates = quality['gates'][mode]
        if [row['input_index'] for row in gates] != list(range(338)):
            raise ValueError('Missing or duplicate validation block')
        passed[mode] = all(row['finite'] and row['elementwise_gate'] and
                           row['relative_l2'] <= bounds['logit_relative_l2'] for row in gates)
        passed[mode] &= math.isfinite(quality['loss'][mode]) and math.isfinite(quality['loss']['native'])
        passed[mode] &= abs(quality['loss'][mode]-quality['loss']['native']) <= bounds['validation_loss_atol']
        if passed[mode] != quality['pass'][mode]:
            raise ValueError('Qualification summary disagrees with numerical evidence')
    return passed


def reduce(tag):
    config = read(RUN/'config.json')
    inputs = read(RUN/'provenance/inputs.json')
    summary = read(RUN/'artifacts/final'/f'summary-{tag}.json')
    expected = {(c, r) for c in config['conditions'] for r in range(1, config['process_replicates']+1)}
    actual = [(row['condition'], row['replicate']) for row in summary]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('A complete three-checkpoint/nine-process cohort is required')
    devices, rows = set(), []
    for checkpoint in inputs['checkpoints']:
        samples = {mode: [] for mode in MODES}
        process_means = {mode: [] for mode in MODES}
        qualified = {mode: True for mode in MODES}
        losses = {mode: [] for mode in MODES}
        sources = []
        paired = {mode: [] for mode in ('candidate_over_legacy', 'native_over_candidate')}
        for entry in summary:
            if entry['condition'] != checkpoint['id']:
                continue
            folder = RUN/'artifacts/attempts'/entry['attempt']
            result, timing, quality = [read(folder/name) for name in ('result.json', 'timing.json', 'quality.json')]
            if result['status'] != 'complete' or result['validation_blocks'] != 338:
                raise ValueError('Incomplete full-validation result')
            if result['config'] != record(RUN/'config.json') or result['source_freeze'] != record(RUN/'provenance/source-freeze.json'):
                raise ValueError('Mixed configuration or implementation identity')
            if result['checkpoint'] != checkpoint:
                raise ValueError('Checkpoint identity mismatch')
            devices.add(result['runtime']['device_uuid'])
            values = audit_samples(timing, config['timing_inputs'], config['timing_passes'])
            passed = audit_quality(quality, config['numerical_bounds'])
            by_cell = {(r['mode'], r['input_index'], r['repeat']): r['host_ms'] for r in timing['samples']}
            for mode in MODES:
                samples[mode].extend(values[mode])
                process_means[mode].append(geomean(values[mode]))
                qualified[mode] &= passed[mode]
                losses[mode].append(quality['loss'][mode])
            for i in range(config['timing_inputs']):
                for repeat in range(config['timing_passes']):
                    c = by_cell[('candidate_graph', i, repeat)]
                    paired['candidate_over_legacy'].append(by_cell[('legacy_graph', i, repeat)]/c)
                    paired['native_over_candidate'].append(by_cell[('native_graph', i, repeat)]/c)
            if entry['replicate'] == 1:
                for mode in ('candidate', 'legacy'):
                    diagnostic = read(folder/f'diagnostics-{mode}.json')
                    if diagnostic['status'] != 'complete' or diagnostic['coverage']['blocks'] != 338:
                        raise ValueError('Incomplete implementation-specific diagnostics')
            sources.extend(record(folder/name) for name in ('result.json', 'timing.json', 'quality.json'))
        rows.append({'id': checkpoint['id'], 'family': checkpoint['family'], 'dose': checkpoint['dose'],
                     'qualified': qualified, 'latency_ms': {mode: geomean(samples[mode]) for mode in MODES},
                     'process_geomean_ranges_ms': {mode: [min(process_means[mode]), max(process_means[mode])] for mode in MODES},
                     'median_host_ms': {mode: median(samples[mode]) for mode in MODES},
                     'paired_speedup': {mode: geomean(values) for mode, values in paired.items()},
                     'loss_by_process': losses, 'canonical_logical_products': checkpoint['canonical_logical_products'],
                     'source_records': sources})
    if len(devices) != 1 or 'unavailable' in devices:
        raise ValueError('One identified physical GPU required')
    base = next(row for row in rows if row['id'] == 'c00')['latency_ms']['native_graph']
    for row in rows:
        row['native_base_speedup'] = {mode: base/row['latency_ms'][mode] if row['qualified'][mode] else None for mode in MODES}
    result = {'status': 'complete', 'all_qualified': all(all(r['qualified'].values()) for r in rows),
              'checkpoints': len(rows), 'fresh_processes': len(summary), 'timings_per_implementation_checkpoint': 1344,
              'latency_aggregation': config['latency_aggregation'], 'native_base_ms': base,
              'device_uuid': next(iter(devices)), 'conditions': rows,
              'interpretation': 'One workload and seed. Process ranges are descriptive, not confidence intervals. '
                                'Different recipes are not quality matched. Frozen opt073 includes dense optimizations.'}
    write(RUN/'results/matched-grid.json', result)
    print({'checkpoints': len(rows), 'all_qualified': result['all_qualified'], 'native_base_ms': base}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='001')
    reduce(parser.parse_args().tag)
