"""Per-setting instruction bypass from retained full-validation counters only."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = {n: next((ROOT/'runs').glob(n+'-*')) for n in ['029', '033', '035', '036']}
OPS = ['qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection',
       'qk_scores', 'probability_value']
LABELS = ['QKV', 'FFN-up', 'FFN-down', 'Attention-out', 'QK', 'PV']
SOURCES = {}


def read(path):
    SOURCES[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))


def pooled_bypass(issued, bypassed):
    if len(issued) != len(bypassed) or not issued:
        raise ValueError('Matched nonempty count lists required')
    if any(type(n) is not int or n < 0 for n in issued+bypassed):
        raise ValueError('Nonnegative integer instruction counts required')
    a, b = sum(issued), sum(bypassed)
    if a+b == 0:
        raise ValueError('Positive instruction denominator required')
    return {'issued_mmas': a, 'bypassed_mmas': b, 'potential_mmas': a+b,
            'bypass_fraction': b/(a+b)}


def operations(diagnostic, canonical):
    assert diagnostic['status'] == 'complete'
    assert diagnostic['coverage'] == {'blocks':338, 'documents':500,
                                     'excluded_tail_tokens':1444, 'input_tokens':692224}
    actual = diagnostic['bf16_scalar_opportunity_lower_bound']['per_operation']
    attention = list(diagnostic['attention_counts_by_layer'].values())
    hybrid = list(diagnostic['hybrid_counts_by_layer'].values())
    assert len(attention) == len(hybrid) == 6
    output = {}
    for i, op in enumerate(OPS):
        if i < 4:
            a = actual[op]
            row = pooled_bypass([a['issued_mmas']], [a['skipped_mmas']])
            row['simt_products'] = a.get('simt_products', 0)
            if op in ['mlp_w2', 'attention_output_projection']:
                j, s = (0, 4) if op == 'mlp_w2' else (2, 5)
                assert row == pooled_bypass([r[j] for r in hybrid], [r[j+1] for r in hybrid]) | {
                    'simt_products': sum(r[s] for r in hybrid)}
                row['counter_method'] = 'Instrumented M8/padded-M16 hybrid; includes SIMT substitution'
            else:
                row['counter_method'] = 'Operand-derived all-zero 16x16 A-fragment predicate'
        else:
            j = 0 if op == 'qk_scores' else 2
            row = pooled_bypass([r[j] for r in attention], [r[j+1] for r in attention])
            row.update(simt_products=0, counter_method='Instrumented attention MMA; includes causal padding')
        fp16, bf16 = canonical['per_operation'][op], actual[op]
        assert fp16['product_count'] == bf16['product_count'] > 0
        row.update(logical_products=fp16['product_count'],
                   fp16_zero_products=fp16['zero_product_count'],
                   fp16_zero_fraction=fp16['zero_product_count']/fp16['product_count'],
                   bf16_activation_zero_products_lower_bound=bf16['zero_product_count'],
                   bf16_activation_zero_fraction_lower_bound=bf16['zero_product_count']/bf16['product_count'])
        output[op] = row
    return output


def verify_checkpoint(result, expected_weight_sha):
    assert result['status'] == 'complete' and result['qualified']
    weight = next(f for f in result['checkpoint']['files'] if Path(f['path']).name == 'model.safetensors')
    assert weight['sha256'] == expected_weight_sha


def graph_means(condition, candidate, expected_weight_sha):
    times = {'candidate_graph': [], 'native_graph': []}
    indices = None
    for rep in [1, 2, 3]:
        folder = RUNS['029']/f'artifacts/attempts/scientific-final-{candidate}-{condition}-r{rep}-a01'
        result = read(folder/'result.json')
        verify_checkpoint(result, expected_weight_sha)
        assert result['candidate'] == candidate and result['validation_blocks'] == 338
        timing = read(folder/'timing.json')
        if indices is None:
            indices = timing['indices']
        assert timing['indices'] == indices and len(indices) == 64
        for mode in times:
            records = [s for s in timing['samples'] if s['mode'] == mode]
            # input_index indexes the retained 64-block list, not the validation corpus.
            assert {(s['input_index'], s['repeat']) for s in records} == {(i, r) for i in range(64) for r in range(7)}
            assert all(s['output_shape'] == [1,2048,50304] for s in records)
            assert len(records) == 448
            times[mode].extend(s['host_ms'] for s in records)
    assert all(len(v) == 1344 and all(math.isfinite(x) and x > 0 for x in v) for v in times.values())
    return {mode: math.exp(math.fsum(math.log(x) for x in v)/len(v)) for mode, v in times.items()}


def main():
    paper = read(HERE/'data/paper-checkpoints.json')
    clipping = read(RUNS['036']/'results/clipping-final-kernel.json')
    by_key = {r['checkpoint_key']: r for r in paper['checkpoints']}
    settings = []
    for p in paper['checkpoints']:
        condition, size = p['timing_condition'], p['model']
        run = '035' if size == '70M' else '033' if p['timing_session'] == 'Run033' else '029'
        attempt = (f'scientific-final-k050-{condition}-r1-a01' if run == '029'
                   else f'scientific-{condition}-r1-001')
        folder = RUNS[run]/'artifacts/attempts'/attempt
        result = read(folder/'result.json')
        weight_sha = p['checkpoint_files']['model.safetensors']['sha256']
        verify_checkpoint(result, weight_sha)
        diagnostic = read(folder/'diagnostics.json')
        row = {'setting_id': f'trained-{size}-{condition}', 'kind':'trained', 'model':size,
               'family':p['family'], 'scope':p['scope'], 'pressure':p['pressure'],
               'kappa':p['kappa'], 'local_pressure_weight':p['local_pressure_weight'], 'p':None,
               'comparison_cohort':p['comparison_cohort'], 'checkpoint_key':p['checkpoint_key'],
               'weight_sha256':weight_sha, 'kernel':p['kernel'], 'session':p['timing_session'],
               'latency_ms':p['latency_ms'], 'fp16_loss':p['loss'],
               'fp16_model_sparsity_fraction':p['sparsity']/100,
               'diagnostic_source':(folder/'diagnostics.json').relative_to(ROOT).as_posix(),
               'operations':operations(diagnostic, p['counts']), 'skip_ablation':None}
        if run == '029':
            timing = {name:graph_means(condition, candidate, weight_sha) for name, candidate in [
                ('all_off', 'k050-no-skip'), ('projection_only', 'k050-attention-dense'), ('full', 'k050')]}
            t = {k:v['candidate_graph'] for k,v in timing.items()}
            assert math.isclose(t['full'], p['latency_ms'], rel_tol=1e-12)
            row['skip_ablation'] = {'source':'Run029, three processes and 1344 host timings per mode',
                'graph_means_ms':timing, 'projection_gain':t['all_off']/t['projection_only'],
                'attention_gain':t['projection_only']/t['full'],
                'attention_increment_ms':t['full']-t['projection_only'],
                'interpretation':'Full-model toggle gains; attention combines QK and PV, not separate site timings'}
        settings.append(row)
    assert len(settings) == 62
    for p in clipping['points']:
        original = by_key[p['checkpoint_key']]
        folder = RUNS['036']/f"artifacts/attempts/scientific-{p['condition']}-r1-001"
        weight_sha = original['checkpoint_files']['model.safetensors']['sha256']
        verify_checkpoint(read(folder/'result.json'), weight_sha)
        settings.append({'setting_id':f"clipping-{p['scale']}-{p['condition']}",
            'kind':'clipping', 'model':p['scale'], 'family':p['family'], 'scope':original['scope'],
            'pressure':'none', 'kappa':None, 'local_pressure_weight':None, 'p':p['p'],
            'comparison_cohort':True, 'checkpoint_key':p['checkpoint_key'], 'weight_sha256':weight_sha,
            'kernel':p['kernel'], 'session':'Run036', 'latency_ms':p['candidate_gm_ms'],
            'fp16_loss':p['retained_fp16_loss'], 'fp16_model_sparsity_fraction':p['sparsity_percent']/100,
            'diagnostic_source':(folder/'diagnostics.json').relative_to(ROOT).as_posix(),
            'operations':operations(read(folder/'diagnostics.json'), p['retained_fp16_counts']),
            'skip_ablation':None})
    assert len(settings) == len({r['setting_id'] for r in settings}) == 102
    baseline = {size:next(r for r in settings if r['model']==size and r['kind']=='trained' and r['family']=='A0')
                for size in ['14M', '70M']}
    for row in settings:
        for op in OPS:
            reference = baseline[row['model']]['operations'][op]
            assert row['operations'][op]['potential_mmas'] == reference['potential_mmas']
            row['operations'][op]['base_model_bypass_fraction'] = reference['bypass_fraction']
            row['operations'][op]['bypass_difference_from_base_pp'] = 100*(
                row['operations'][op]['bypass_fraction']-reference['bypass_fraction'])
    ablations = [r for r in settings if r['skip_ablation']]
    assert len(ablations) == 35
    data = {'metric':'B_op = pooled bypassed MMA / pooled (issued + bypassed MMA)',
            'unit':'fractions in JSON, percentages in figures/tables; 1 means every potential MMA bypassed',
            'scope':'62 trained checkpoints plus 40 control-clipping settings; four p0 clipping records repeat existing weights',
            'coverage':{'validation_blocks_per_setting':338, 'documents':500, 'input_tokens':692224,
                        'excluded_tail_tokens':1444, 'trained_14m':40, 'trained_70m':22, 'clipping':40},
            'limits':['Bypass includes SIMT substitution at h/z; SIMT products retained separately.',
                      'Attention counts include causal padding; base-model bypass is retained, not subtracted from primary B.',
                      'FP16 canonical scalar counts and BF16 activation-derived lower bounds are distinct.',
                      'No model-wide average is defined; no bypass fraction is interpreted as a speedup.',
                      'Only 35 historical 14M settings have matched skip ablations.'],
            'settings':settings, 'sources_sha256':SOURCES,
            'summary':{'attention_ablation_count':len(ablations),
                       'attention_faster_count':sum(r['skip_ablation']['attention_gain']>1 for r in ablations)},
            'script':Path(__file__).name, 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = HERE/'data/operation-bypass.json'
    destination.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    lines = ['# Per-setting matrix-instruction bypass', '',
        'Each cell is 100 * bypassed / (issued + bypassed), pooled over all six layers and 338 validation blocks. '
        '100% means all potential matrix instructions are bypassed; scalar replacement work may remain. '
        'QKV is fed by a; FFN-up by m; FFN-down by h; attention-output by z.', '',
        'PV baseline bypass is 10.4167% at 14M and 5.1471% at 70M, including causal masking/padding. '
        'These raw percentages must not all be attributed to the intervention. '
        'The [JSON](data/operation-bypass.json) includes integer counts, per-operation FP16 and BF16 scalar opportunities, '
        'SIMT products, baseline differences, checkpoint identities and source hashes.', '']
    for size in ['14M', '70M']:
        for kind in ['trained', 'clipping']:
            lines += [f'## {size}: {kind} settings', '',
                      '| Setting | Recipe | Dose | '+' | '.join(LABELS)+' |',
                      '|---|---|---|'+'---:|'*6]
            for row in settings:
                if (row['model'], row['kind']) != (size, kind):continue
                dose = (f"p={row['p']:g}" if kind=='clipping' else f"kappa={row['kappa']:g}" if row['kappa'] is not None
                        else f"lambda={row['local_pressure_weight']:g}" if row['local_pressure_weight'] is not None else '-')
                lines.append('| '+row['setting_id']+' | '+row['family']+' | '+dose+' | '+
                             ' | '.join(f"{100*row['operations'][op]['bypass_fraction']:.4f}" for op in OPS)+' |')
            lines.append('')
    lines += ['## Matched 14M skip-toggle evidence', '',
              'These gains use full-model geometric mean host latency from 1,344 timings per mode. '
              'Projection gain = all skipping off / projection only; attention gain = projection only / full skipping. '
              'Values below 1 indicate a slowdown. QK and PV are toggled together. '
              'No corresponding ablation was measured for the five new 14M h-only A7 models, 70M or control clipping.', '',
              '| Setting | All off (ms) | Projection only (ms) | Full (ms) | Projection gain | Attention gain |',
              '|---|---:|---:|---:|---:|---:|']
    for row in ablations:
        a = row['skip_ablation']; t = a['graph_means_ms']
        lines.append('| '+row['setting_id']+' | '+' | '.join(f"{t[k]['candidate_graph']:.6f}" for k in ['all_off','projection_only','full'])+
                     f" | {a['projection_gain']:.6f} | {a['attention_gain']:.6f} |")
    (HERE/'TABLE_OPERATION_BYPASS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({'settings':len(settings), 'operation_records':len(settings)*6, 'ablations':len(ablations),
                      'attention_faster_count':data['summary']['attention_faster_count'], 'sources':len(SOURCES)}))


if __name__ == '__main__':
    main()
