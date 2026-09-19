"""Post-retrieval audit and conditional-effect table; never runs GPU code."""
import hashlib
import json
import math
from pathlib import Path

from controls import MODES, OPS, mask

HERE = Path(__file__).resolve().parent
OPERATION = dict(zip(OPS, ['qkv_projection', 'mlp_w1', 'mlp_w2',
                         'attention_output_projection', 'qk_scores', 'probability_value']))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def geomean(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(math.fsum(map(math.log, values)) / len(values))


def counters(diag):
    actual = diag['bf16_scalar_opportunity_lower_bound']['per_operation']
    rows = {}
    for op, key in OPERATION.items():
        if op in ['qk', 'pv']:
            i = 0 if op == 'qk' else 2
            layers = list(diag['attention_counts_by_layer'].values())
            assert len(layers) == 6
            issued, skipped = (sum(r[j] for r in layers) for j in [i, i+1])
            scalar = 0
        else:
            issued, skipped = actual[key]['issued_mmas'], actual[key]['skipped_mmas']
            scalar = actual[key].get('simt_products', 0)
        assert all(isinstance(v, int) and v >= 0 for v in [issued, skipped, scalar])
        rows[op] = {'issued': issued, 'bypassed': skipped, 'scalar_products': scalar,
                    'bypass_percent': 100 * skipped / (issued + skipped)}
    return rows


def main():
    source = HERE/'results/operation-latency.json'
    report = read(source)
    assert report['status'] == 'qualified'
    assert read(HERE/'artifacts/verification.json')['complete_scientific_matrix']
    for row in report['sources']:
        data = (HERE/row['path']).read_bytes()
        assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    full = counters(report['diagnostics']['full'])
    losses, max_errors, gpu_ids, indices = set(), [], set(), None
    samples = 0
    for mode in MODES:
        pooled = []
        for rep in [1, 2, 3]:
            folder = HERE/'artifacts/attempts'/f'scientific-{mode}-r{rep}-001'
            result, timing, quality = [read(folder/name) for name in
                                      ['result.json', 'timing.json', 'quality.json']]
            gpu_ids.add(result['runtime']['device_uuid'])
            assert result['qualified'] and result['validation_blocks'] == 338
            assert result['implementation_coverage']['operation_mask'] == mask(mode)
            assert all(quality['pass'].values()) and quality['excluded_tail_tokens'] == 1444
            assert quality['prediction_tokens'] == 691886
            losses.update(quality['loss'].values())
            for checks in quality['gates'].values():
                assert len(checks) == 338 and all(r['pass'] for r in checks)
                max_errors.extend(r['max_abs'] for r in checks)
            if indices is None:
                indices = timing['indices']
            assert timing['indices'] == indices and len(set(indices)) == 64
            for backend in ['candidate_graph', 'native_graph']:
                rows = [r for r in timing['samples'] if r['mode'] == backend]
                assert len(rows) == 448
                assert {(r['input_index'], r['repeat']) for r in rows} == {
                    (i, p) for i in range(64) for p in range(7)}
                assert all(r['output_shape'] == [1, 2048, 50304] for r in rows)
                mean = geomean([r['host_ms'] for r in rows])
                expected = next(r for r in report['processes'][mode] if r['replicate'] == rep)[backend]
                assert math.isclose(mean, expected, rel_tol=1e-12)
                samples += len(rows)
                if backend == 'candidate_graph':
                    pooled.extend(r['host_ms'] for r in rows)
        assert math.isclose(geomean(pooled), report['latencies_ms'][mode], rel_tol=1e-12)
        diag = report['diagnostics'][mode]
        assert diag['coverage'] == {'blocks': 338, 'documents': 500,
                                    'input_tokens': 692224, 'excluded_tail_tokens': 1444}
        observed = counters(diag)
        for op in OPS:
            assert observed[op]['issued'] + observed[op]['bypassed'] == full[op]['issued'] + full[op]['bypassed']
            if mask(mode)[op]:
                assert observed[op] == full[op]
            else:
                assert observed[op]['bypassed'] == observed[op]['scalar_products'] == 0
        for key in ['zero_products', 'model_products']:
            assert diag['bf16_scalar_opportunity_lower_bound'][key] == report['diagnostics']['full']['bf16_scalar_opportunity_lower_bound'][key]
        for operation in OPERATION.values():
            for field in ['product_count', 'zero_product_count']:
                assert diag['bf16_scalar_opportunity_lower_bound']['per_operation'][operation][field] == report['diagnostics']['full']['bf16_scalar_opportunity_lower_bound']['per_operation'][operation][field]
    assert len(gpu_ids) == 1 and samples == 26880
    rows = []
    reference_range = report['process_ranges_ms']['full']
    for op in OPS:
        interval = report['process_ranges_ms']['without-'+op]
        span = [1000*(interval[0]-reference_range[1]),
                1000*(interval[1]-reference_range[0])]
        rows.append({'operation': op, **full[op], **report['conditional_effects'][op],
                     'without_path_ms': report['latencies_ms']['without-'+op],
                     'saved_microseconds': 1000*report['conditional_effects'][op]['saved_ms'],
                     'cross_process_difference_span_us': span,
                     'span_excludes_zero': span[0] > 0 or span[1] < 0})
    export = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'source': source.relative_to(HERE).as_posix(),
              'generating_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'raw_timing_samples_verified': samples, 'qualified_processes': 30,
              'loss_values': sorted(losses), 'maximum_absolute_logit_error': max(max_errors),
              'counter_control_audit': 'All six disabled paths have zero bypass/scalar counts; enabled paths match full; logical counts unchanged',
              'interpretation': 'Conditional effects, not additive allocations. Spans compare extrema of three process means per mode; not confidence intervals.',
              'rows': rows}
    (HERE/'results/conditional-effects-audit.json').write_text(
        json.dumps(export, indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# Conditional full-model latency effects', '',
             '14M T7/Pall, kappa=0.5. Positive saved time means enabling the path helps.', '',
             '| Path | Instruction bypass (%) | Path off (ms) | Time saved (us) | Difference span (us) | Conditional speedup |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lo, hi = row['cross_process_difference_span_us']
        lines.append(f"| {row['operation']} | {row['bypass_percent']:.2f} | {row['without_path_ms']:.6f} | {row['saved_microseconds']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {row['speedup']:.4f}x |")
    lines += ['', 'Spans use process extrema, not confidence intervals. These effects do not add.', '',
              '| Control | Full-model latency (ms) |', '|---|---:|']
    for mode in ['frozen', 'full', 'off', 'projection']:
        lines.append(f"| {mode} | {report['latencies_ms'][mode]:.6f} |")
    lines += ['', f'Verified {samples:,} raw timing samples, all 30 full-validation processes, and all ten diagnostic passes.',
              'Source and exact values: `conditional-effects-audit.json` and `operation-latency.json`.', '']
    (HERE/'results/conditional-effects.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
