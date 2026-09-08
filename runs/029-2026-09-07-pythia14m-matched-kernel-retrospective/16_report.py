"""Checkpoint-paired descriptive report from the complete matched reduction."""
import math
from collections import Counter
from io_utils import RUN, read, write, record, verify


def describe(values):
    if not values:
        return {'n': 0}
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Positive finite speedups required')
    return {'n': len(values), 'geomean': math.exp(sum(map(math.log, values))/len(values)),
            'minimum': min(values), 'maximum': max(values),
            'above_one': sum(v > 1 for v in values)}


def comparisons(points):
    final = {(p['candidate'], p['condition']): p for p in points if p['phase'] == 'final'}
    controls = ['p0', 'k050-no-skip', 'k050-attention-dense', 'k049']
    rows = []
    for (candidate, condition), point in sorted(final.items()):
        if candidate != 'k050':
            continue
        row = {'condition': condition, 'R_model': point['R_model'],
               'k050_qualified': point['qualified'], 'ratios': {}}
        for control in controls:
            other = final[(control, condition)]
            if point['qualified'] and other['qualified']:
                row['ratios'][control] = point['speedup']/other['speedup']
        rows.append(row)
    return {'per_checkpoint': rows,
            'summary': {c: describe([r['ratios'][c] for r in rows if c in r['ratios']]) for c in controls},
            'interpretation': 'Ratios of native-normalized speedups from separate randomized fresh processes, not direct within-process toggle timings. Only jointly qualified checkpoints enter each comparison.'}


def numerical_summary(quality, bounds):
    gates = quality['gates']['candidate_graph']
    return {'failed_blocks': sum(not g['pass'] for g in gates),
            'nonfinite_blocks': sum(not g['finite'] for g in gates),
            'elementwise_failure_blocks': sum(not g['elementwise_gate'] for g in gates),
            'relative_l2_failure_blocks': sum(g['relative_l2'] > bounds['logit_relative_l2'] for g in gates),
            'maximum_absolute_logit_difference': max(g['max_abs'] for g in gates),
            'maximum_relative_l2': max(g['relative_l2'] for g in gates),
            'loss_delta': quality['loss_delta']['candidate_graph'],
            'loss_gate': abs(quality['loss_delta']['candidate_graph']) <= bounds['validation_loss_atol']}


def main():
    source = RUN/'results/matched-retrospective-001.json'
    data = read(source)
    for row in data['sources']:
        verify(row)
    points = data['points']
    outcomes = Counter()
    for p in points:
        for r in p['replicates']:
            key = 'qualified' if r['qualified'] else 'numerical_failure' if r['status'] == 'complete' else r['status']
            outcomes[key] += 1
    curve = data['progress']
    changes = [r for before, r in zip(curve, curve[1:]) if r['speedup'] > before['speedup']]
    final = [p for p in points if p['phase'] == 'final']
    numerical_failures = [{'phase': p['phase'], 'candidate': p['candidate'],
                           'condition': p['condition'], 'statuses': p['statuses'],
                           'replicate_qualified': [r['qualified'] for r in p['replicates']]}
                          for p in points if not p['qualified']]
    bounds = read(RUN/'config.json')['numerical_bounds']
    for failure in numerical_failures:
        point = next(p for p in points if all(p[k] == failure[k] for k in ['phase', 'candidate', 'condition']))
        failure['numerical_checks'] = [{'replicate': r['replicate'],
            **numerical_summary(read(RUN/'artifacts/attempts'/r['attempt']/'quality.json'), bounds)}
            for r in point['replicates'] if r['status'] == 'complete']
    diagnosed = []
    for d in data['diagnostics']:
        # Pool integer counters over layers before taking ratios.
        attention = [sum(row[i] for row in d['attention_counts_by_layer'].values()) for i in range(4)]
        hybrid = [sum(row[i] for row in d['hybrid_counts_by_layer'].values()) for i in range(6)]
        def fraction(numerator, denominator):
            return numerator/denominator if denominator else None
        diagnosed.append({'condition': d['condition'], 'canonical_R_model': d['canonical_R_model'],
                           'bf16_scalar_lower_bound': d['bf16_scalar_opportunity_lower_bound']['fraction'],
                           'attention_counts_qk_issued_skipped_pv_issued_skipped': attention,
                           'hybrid_counts_h_issued_bypassed_z_issued_bypassed_h_simt_z_simt': hybrid,
                           'qk_skipped_fraction': fraction(attention[1], sum(attention[:2])),
                           'pv_skipped_fraction': fraction(attention[3], sum(attention[2:])),
                           'h_mma_bypassed_fraction': fraction(hybrid[1], sum(hybrid[:2])),
                           'z_mma_bypassed_fraction': fraction(hybrid[3], sum(hybrid[2:4]))})
    value = {'process_outcomes': dict(outcomes), 'group_qualification': data['qualification_summary'],
             'progress_improvements': changes, 'final_progress': curve[-1],
             'final_candidates': {c: describe([p['speedup'] for p in final if p['candidate'] == c and p['qualified']])
                                  for c in sorted({p['candidate'] for p in final})},
             'comparisons': comparisons(points), 'unqualified_groups': numerical_failures,
             'diagnostic_summary': diagnosed, 'k050_regression': data['k050_regression'],
             'limits': 'Descriptive retrospective statistics. Geometric means weight checkpoints equally. MMA bypass includes SIMT substitution and padded work, not pure zero-FLOP removal. Canonical FP16 opportunities remain separate from BF16 execution counters.',
             'source': record(source), 'script': record(__file__)}
    write(RUN/'results/report-001.json', value)
    print({k: value[k] for k in ['process_outcomes', 'group_qualification', 'progress_improvements', 'final_candidates']})
    print(value['comparisons']['summary'])


if __name__ == '__main__':
    main()
