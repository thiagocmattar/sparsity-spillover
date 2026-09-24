"""Verify the retained 31M evidence and derive the approved manuscript additions."""
import hashlib
import json
import math
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / 'runs/054-2026-09-24-pythia31m-t2-ph'
KAPPAS = [0, .01, .05, .1, .5]


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    architecture = read(RUN / 'architecture_config.json')
    initialization = read(RUN / 'prelaunch/initialization/metadata.json')
    cohorts = [read(ROOT / 'analyses/037-2026-09-24-14m-31m-70m-latency-quality/data/31m-results.json'),
               read(HERE / 'data/31m-t7-results.json')]
    for cohort in cohorts:
        for name, digest in cohort['sources_sha256'].items():
            assert sha(ROOT / name) == digest, name
        sources.update(cohort['sources_sha256'])
    points = [p for cohort in cohorts for p in cohort['points']]
    base, = [p for p in points if p['scope'] == '0']
    native = base['implementation_latency_ms']['native_graph']
    rows, identities, training_settings = [], [], []
    processes = []
    for p in points:
        attempt = ROOT / p['source_attempt']
        manifest = read(attempt / 'manifest.json')
        metrics = read(attempt / 'metrics.json')
        logical = read(attempt / 'diagnostics/logical_products.json')
        config_path = attempt / 'config.yaml'
        sources[config_path.relative_to(ROOT).as_posix()] = sha(config_path)
        config = yaml.safe_load(config_path.read_text())
        training_settings.append(config['training'])
        assert config['seeds'] == dict(model=1234, data_order=1234)
        assert manifest['completed_steps'] == 712 and manifest['input_tokens'] == 1493172224
        assert not manifest['model']['released_weights_loaded']
        assert manifest['model']['revision'] == initialization['revision']
        assert p['parameters'] == initialization['parameter_count'] == 30494720
        identities.append((manifest['initial_parameter_sha256'], manifest['data']['training_schedule_hash']))
        for coverage in (logical['coverage'], metrics['validation']['final']):
            assert coverage['sequences'] == 338 and coverage['excluded_tail_tokens'] == 1444
            assert coverage['input_tokens'] == 692224 and coverage['complete_block_coverage']
        assert metrics['validation']['final']['loss'] == p['loss']
        measured = logical['measured']
        assert measured['R_model'] == p['R_model'] == measured['block_zero_product_count'] / measured['model_product_count']
        latency = native if p['scope'] == '0' else p['latency_ms']
        row = dict(recipe={'0': 'Base', 'hz': 'T2/Ph', '7': 'T7/Ph'}[p['scope']],
                   kappa=p['kappa'], loss=p['loss'], delta_loss=p['loss']-base['loss'],
                   latency_ms=latency, delta_latency_ms=latency-native,
                   S_model_percent=100*p['R_model'],
                   zero_product_count=measured['block_zero_product_count'],
                   model_product_count=measured['model_product_count'],
                   source_attempt=p['source_attempt'], checkpoint_sha256=p['checkpoint_key'])
        rows.append(row)
        if p['scope'] != '0':
            assert p['condition']['pressure_sites'] == ['h']
            assert p['condition']['pressure_method'] == 'orthogonal_l1'
            assert p['condition']['pressure_weight'] == p['condition']['step_budget'] == 1
        run = attempt.parents[2]
        for proc in p['processes']:
            folder = run / 'latency/artifacts/attempts' / proc['attempt']
            runtime = read(folder / 'manifest.json')
            quality = read(folder / 'quality.json')
            timing = read(folder / 'timing.json')
            assert runtime['qualified'] and runtime['scientific_measurement']
            assert quality['blocks'] == 338 and all(quality['pass'].values())
            c = runtime['config']
            assert (c['timing_inputs'], c['timing_passes'], c['fresh_processes']) == (64, 7, 3)
            assert (c['logit_atol'], c['logit_rtol'], c['relative_l2_max'], c['pooled_loss_delta_max']) == (.25, .02, .02, .001)
            assert c['precision'] == 'bfloat16' and c['full_logits']
            assert c['batch_size'] == 1 and c['sequence_length'] == 2048
            for mode in ('native_graph', 'kernel_graph'):
                values = [s['host_ms'] for s in timing['samples'] if s['mode'] == mode]
                assert len(values) == 448
                gm = math.exp(math.fsum(map(math.log, values))/len(values))
                assert math.isclose(gm, p['latency_process_ms'][mode][proc['replicate']-1], rel_tol=1e-12)
            processes.append(dict(session=p['timing_session'], **proc))
    assert len(rows) == 11 and len(processes) == 33 and len(set(identities)) == 1
    assert identities[0][0] == initialization['parameter_sha256']
    assert all(s == training_settings[0] for s in training_settings)
    training = training_settings[0]
    assert training['micro_batch_size'] * training['gradient_accumulation_steps'] == 1024
    assert (training['peak_learning_rate'], training['minimum_learning_rate']) == (.001, .0001)

    L, d, f, V, T = (architecture[k] for k in ('num_hidden_layers', 'hidden_size', 'intermediate_size', 'vocab_size', 'max_position_embeddings'))
    assert (L, d, f, V, T, architecture['num_attention_heads']) == (6, 256, 1024, 50304, 2048, 8)
    block = T*(4*d*d + 2*d*f) + d*T*(T+1)
    denominator = L*block + T*d*V
    numerators = {'T0': 0, 'T1': L*T*d*f, 'T2': L*T*(d*f+d*d),
                  'T4': L*T*(4*d*d+2*d*f), 'T7': L*block}
    ceilings = {key: dict(reachable_product_count=n, model_product_count=denominator,
                          percent=100*n/denominator, unit='scalar products per full uncached sequence')
                for key, n in numerators.items()}
    assert denominator*338 == rows[0]['model_product_count']

    figure = read(HERE / 'data/scale-figure.json')
    claims = {}
    for scale in ('14M', '31M', '70M'):
        executions = [p for p in figure['executions'] if p['model'] == scale]
        groups = {scope: sorted([p for p in executions if p['scope'] == scope], key=lambda p: p['kappa']) for scope in ('hz', '7')}
        assert all([p['kappa'] for p in group] == KAPPAS for group in groups.values())
        native_base, = [p for p in executions if p['backend'] == 'PyTorch']
        assert all(a['loss'] < b['loss'] for a,b in zip(groups['hz'], groups['7']))
        assert min(p['latency_ms'] for p in groups['7']) < min(p['latency_ms'] for p in groups['hz'])
        if scale != '14M':
            assert all(p['loss'] > native_base['loss'] for g in groups.values() for p in g)
        else:
            assert all(p['loss'] < native_base['loss'] and p['latency_ms'] < native_base['latency_ms']
                       for p in groups['hz'] if p['kappa'] <= .1)
        claims[scale] = dict(t2_lower_loss_at_all_matched_kappas=True,
                             t7_lower_minimum_sweep_latency=True,
                             t2_kappa0_to_01_latency_reduction_ms=groups['hz'][0]['latency_ms']-groups['hz'][3]['latency_ms'],
                             base_references=[p for p in executions if p['scope'] == '0'])
    result = dict(architecture=architecture, initialization=initialization,
                  matched_identity=dict(initial_parameter_sha256=identities[0][0], training_schedule_sha256=identities[0][1]),
                  training=training, tokens_per_parameter=1493172224/30494720,
                  ceilings_31m=ceilings, rows_31m=rows,
                  base_kernel_latency_ms=base['latency_ms'], base_native_latency_ms=native,
                  illustrative_kappa=.1, illustrative_selection_rule='Same moderate threshold as the main 14M comparison; fixed before extracting numbers.',
                  cross_scale_claims=claims, qualified_processes=processes, source_sha256=sources,
                  source_script_sha256=sha(Path(__file__)))
    (HERE / 'data/manuscript-evidence.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print('Verified 11 checkpoints, 33 qualified processes and all 15 matched-kappa comparisons.')
    print('Tokens/parameter:', result['tokens_per_parameter'])
    print('Ceilings:', {k: round(v['percent'], 2) for k,v in ceilings.items()})
    for row in rows:
        print(row['recipe'], row['kappa'], *(f'{row[k]:.6f}' for k in ('S_model_percent', 'loss', 'delta_loss', 'latency_ms', 'delta_latency_ms')))


if __name__ == '__main__':
    main()
