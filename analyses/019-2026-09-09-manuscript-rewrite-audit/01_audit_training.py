"""Audit existing endpoints and retrospective quality budgets; no inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / 'analyses/018-2026-09-08-results-materials'
CLIPPING = ROOT / 'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nondominated(rows):
    return [r['id'] for r in rows if not any(
        q['loss'] <= r['loss'] and q['R_model'] >= r['R_model'] and
        (q['loss'] < r['loss'] or q['R_model'] > r['R_model']) for q in rows)]


def best(rows, reference, budget):
    eligible = [r for r in rows if r['loss'] <= reference['loss'] + budget]
    if not eligible:
        return None
    winner = max(eligible, key=lambda r: (r['R_model'], -r['loss'], r['id']))
    return {k: winner.get(k) for k in
            ('id', 'kind', 'scale', 'family', 'dose', 'training_parameter',
             'source_checkpoint_id', 'loss', 'R_model')} | {
        'delta_loss_vs_A0': winner['loss'] - reference['loss'],
        'sparsity_percent': 100 * winner['R_model'], 'eligible_count': len(eligible)}


def main():
    saved = read(PRIOR / 'figure_data.json')
    spec = importlib.util.spec_from_file_location('analysis018_evidence', PRIOR / 'evidence.py')
    evidence = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evidence)
    rebuilt = evidence.load_evidence()
    # A manuscript edit may change live source fingerprints, never measured fields.
    assert {k:v for k,v in saved.items() if k != 'sources'} == {
        k:v for k,v in rebuilt.items() if k != 'sources'}
    trained = saved['trained']
    clipping = read(CLIPPING)
    points = clipping['points']
    assert len(trained) == 54 and len(points) == 540
    by_id = {r['id']: r for r in trained}
    assert len(by_id) == 54
    p0 = {r['source_checkpoint_id']: r for r in points if r['dose'] == 0}
    assert set(p0) == set(by_id)
    for r in trained:
        evidence.logical_counts(r['counts'])
        assert r['counts']['model_product_count'] == 338 * r['ceiling']['model_product_count']
        assert r['identity']['training_tokens'] == 1493172224
        assert r['identity']['training']['max_steps'] == 712
        assert r['identity']['seeds'] == {'model': 1234, 'data_order': 1234}
    for r in points:
        evidence.coverage(r['coverage'])
        evidence.logical_counts(r['counts'])
        assert math.isclose(r['R_model'], r['counts']['R_model'], abs_tol=1e-14)
    for delta in saved['contrasts']:
        t, ref = by_id[delta['treatment']], by_id[delta['reference']]
        assert math.isclose(delta['delta_loss'], t['loss'] - ref['loss'], abs_tol=1e-12)
        assert math.isclose(delta['delta_R_pp'], 100 * (t['R_model'] - ref['R_model']), abs_tol=1e-12)

    endpoints, budgets, frontier_audit, operations = [], [], {}, []
    for scale in ('14M', '70M', '410M'):
        rows = [r for r in trained if r['scale'] == scale]
        clips = [r for r in points if r['scale'] == scale]
        reference = by_id[f'{scale}:A0:None']
        reach = next(r['ceiling'] for r in rows if r['family'] == 'A7-OL1')
        for r in rows:
            assert math.isclose(r['R_model'] / reach['R_model_max_fraction'],
                                r['counts']['R_block'], abs_tol=1e-12)
            endpoints.append({k:r[k] for k in ('id','scale','family','dose','loss','R_model','source')} | {
                'delta_loss_vs_A0':r['loss']-reference['loss'],
                'checkpoint_content_sha256':p0[r['id']]['checkpoint_content_sha256'],
                'S_block':r['counts']['R_block'],
                'R_arch_A7':reach['R_model_max_fraction'],
                'evaluation_precision':'FP16', 'validation_blocks':338,
                'input_tokens':692224,'prediction_tokens':691886,
                'excluded_tail_tokens':1444,'model_seed':1234,'data_order_seed':1234,
                'training_input_tokens':1493172224,'checkpoint_step':712})
        for budget in (0.05, 0.10, 0.20):
            budgets.append({'scale':scale,'budget':budget,'A0_loss':reference['loss'],
                            'best_trained':best(rows,reference,budget),
                            'best_trained_sparse':best([r for r in rows if r['family'] != 'A0'],reference,budget),
                            'best_clipped':best(clips,reference,budget),
                            'best_all_evaluated':best(rows+clips,reference,budget)})
        drift = [{'id':r['id'],'delta_loss':p0[r['id']]['loss']-r['loss'],
                  'delta_sparsity_pp':100*(p0[r['id']]['R_model']-r['R_model'])} for r in rows]
        eps_l = max(abs(r['delta_loss']) for r in drift)
        eps_s = max(abs(r['delta_sparsity_pp']) for r in drift)/100
        a0_clips = [r for r in clips if r['family'] == 'A0']
        checks = []
        for p in a0_clips:
            strict = [r['id'] for r in rows if r['loss'] <= p['loss'] and r['R_model'] >= p['R_model']
                      and (r['loss'] < p['loss'] or r['R_model'] > p['R_model'])]
            separated = [r['id'] for r in rows if r['loss'] < p['loss']-eps_l
                         and r['R_model'] > p['R_model']+eps_s]
            checks.append({'id':p['id'],'p':p['dose'],'loss':p['loss'],'R_model':p['R_model'],
                           'strict_trained_dominators':strict,'beyond_p0_drift_on_both_axes':separated})
        frontier_audit[scale] = {'p0_rerun_differences':drift,
             'empirical_max_p0_loss_drift':eps_l,'empirical_max_p0_sparsity_drift_pp':100*eps_s,
             'interpretation':'Observed p=0 rerun differences, not training uncertainty or a confidence bound.',
             'A0_clipping_comparisons':checks,
             'strict_nondominated_all_evaluations':nondominated(rows+clips)}
        for family in ('A4-OL1','A7-OL1'):
            r=by_id[f'{scale}:{family}:0.5']
            terms={op:100*v['zero_product_count']/r['counts']['model_product_count']
                   for op,v in r['counts']['per_operation'].items()}
            assert math.isclose(sum(terms.values()),100*r['R_model'],abs_tol=1e-12)
            operations.append({'id':r['id'],'scale':scale,'family':family,
                               'contributions_pp':terms,'total_percent':100*r['R_model'],
                               'attention_pp':terms['qk_scores']+terms['probability_value'],
                               'projection_pp':sum(v for k,v in terms.items() if k not in ('qk_scores','probability_value'))})
    result={'sources':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                       (PRIOR/'figure_data.json',PRIOR/'evidence.py',CLIPPING)},
            'source_reconstruction':'All Analysis 018 scientific fields reproduce exactly from original evidence.',
            'coverage':{'trained':54,'clipped':540,'paired_contrasts':29,'scale_pairs':15},
            'quality_budget_rule':'Retrospective maximum observed S_model with loss <= canonical same-size A0 + budget; no interpolation. Includes every eligible canonical trained and measured clipped endpoint, retaining actual p=0 reruns.',
            'quality_budget_latency_scope':'Clipped endpoints have no measured runtime; 70M/410M matched K050 runtime unavailable. Do not attach parent checkpoint latency to a clipped model.',
            'endpoints':endpoints,'paired_contrasts':saved['contrasts'],'scale_pairs':saved['scale_pairs'],
            'quality_budgets':budgets,'frontier_audit':frontier_audit,'operation_contributions':operations}
    (HERE/'training-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    for r in budgets:
        a,b=r['best_trained'],r['best_all_evaluated']
        print(r['scale'],r['budget'],'trained',a['id'],round(a['sparsity_percent'],4),
              'all',b['id'],round(b['sparsity_percent'],4))
    print('Verified 54 endpoints, 540 clipping records, 29 paired differences, operation sums and A7/block identity.')


if __name__ == '__main__':
    main()
