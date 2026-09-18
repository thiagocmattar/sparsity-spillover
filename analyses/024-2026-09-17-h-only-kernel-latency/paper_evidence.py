"""One checkpoint-identity join for the task.md paper figure set."""
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sparsity_research.ceilings import architecture_ceiling

KAPPAS = [0, .01, .05, .1, .5]
OPS = ['qkv_projection', 'mlp_w1', 'mlp_w2', 'attention_output_projection', 'qk_scores', 'probability_value']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    sources = {}

    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))

    tables = [('14M', HERE/'data/results.json'),
              ('70M', ROOT/'runs/035-2026-09-18-pythia70m-k050-port/results/70m-final-kernel.json')]
    rows, identities, configs = [], set(), {}
    for size, path in tables:
        timing = read(path)
        for t in timing['points']:
            assert t['qualified'] and all(r['qualified'] for r in t['replicates'])
            files = {Path(f['path']).name: f for f in t['checkpoint_files']}
            weight = files['model.safetensors']
            attempt = Path(weight['path']).parents[2]
            # Join by exact weight/config/metadata identity; paths are checked as well.
            identity = hashlib.sha256(json.dumps(sorted((name, f['sha256']) for name, f in files.items())).encode()).hexdigest()
            assert identity not in identities
            identities.add(identity)
            metrics = read(ROOT/attempt/'metrics.json')
            logical = read(ROOT/attempt/'diagnostics/logical_products.json')
            manifest = read(ROOT/attempt/'manifest.json')
            config = read(ROOT/files['config.json']['path'])
            assert sha(ROOT/files['config.json']['path']) == files['config.json']['sha256']
            final = metrics['validation']['final']
            for cover in [final, logical['coverage']]:
                for key, value in {'sequences':338, 'input_tokens':692224, 'source_tokens':693668,
                                   'excluded_tail_tokens':1444, 'complete_block_coverage':True}.items():
                    assert cover[key] == value
            counts = logical['measured']
            assert counts == t['canonical_counts']
            assert sum(o['zero_product_count'] for o in counts['per_operation'].values()) == counts['block_zero_product_count']
            assert sum(o['product_count'] for o in counts['per_operation'].values()) == counts['block_product_count']
            assert counts['model_product_count'] == counts['block_product_count']+counts['lm_head_product_count']
            family = t['family']
            scope = '4' if family.startswith('A4') else '7' if family.startswith('A7') else '0' if family == 'A0' else '1'
            pressure = 'h' if family.endswith('@h') else 'all' if '@' in family else 'none'
            if family == 'A1-H+OL1': pressure = 'h'
            if family == 'A1-H+L1': pressure = 'L1'
            comparison = scope in ['4','7'] or family in ['A0','A1-H']
            latency_key = 'k050_gm_ms' if size == '14M' else 'candidate_gm_ms'
            row = {'checkpoint_key':identity, 'model':size, 'family':family, 'scope':scope,
                   'pressure':pressure, 'kappa':t.get('kappa', t.get('kappa_or_lambda')) if scope in ['4','7'] else None,
                   'local_pressure_weight':t.get('kappa_or_lambda') if '+' in family and scope == '1' else None,
                   'comparison_cohort':comparison, 'source_attempt':attempt.as_posix(), 'checkpoint_files':files,
                   'loss':final['loss'], 'logical_pass_loss':logical['coverage']['loss'],
                   'loss_evaluation':{'precision':'FP16 autocast / FP32 parameters', 'pass':'ordinary reloaded final checkpoint',
                                      'batches':final['batches'], 'source':(attempt/'metrics.json').as_posix()},
                   'sparsity_evaluation':{'precision':'FP16', 'pass':'canonical eager logical diagnostic'},
                   'counts':counts, 'sparsity':100*counts['block_zero_product_count']/counts['model_product_count'],
                   'latency_ms':t[latency_key], 'native_latency_ms':t['native_gm_ms'],
                   'kernel':'K050' if size == '14M' else 'k050-70m-v2',
                   'timing_session':t.get('session','Run035'), 'timing_condition':t['condition'],
                   'timing_precision':'BF16', 'timing_workload':timing['protocol'],
                   'timing_device_uuid':t.get('gpu_uuid', timing.get('device_uuid')),
                   'process_latency_ms':[r[latency_key] for r in t['replicates']],
                   'timing_indices':t.get('timing_block_indices',timing.get('timing_indices')),
                   'initial_parameter_sha256':manifest['initial_parameter_sha256'],
                   'training_schedule_hash':manifest['training_schedule_hash']}
            assert math.isfinite(row['loss']) and row['latency_ms'] > 0
            rows.append(row)
            configs[size] = config
    assert len(rows) == 62
    for size, n in [('14M',32),('70M',22)]:
        selected = [r for r in rows if r['model']==size and r['comparison_cohort']]
        assert len(selected)==n
        assert len({r['initial_parameter_sha256'] for r in selected})==1
        assert len({r['training_schedule_hash'] for r in selected})==1
        base = next(r for r in selected if r['scope']=='0')
        for r in rows:
            if r['model']==size:
                r['dense_loss_difference'] = r['loss']-base['loss']
                r['dense_speedup'] = base['latency_ms']/r['latency_ms']
                r['dense_checkpoint_key'] = base['checkpoint_key']
        for scope in ['4','7']:
            for pressure in (['none','h','all'] if size=='14M' else ['h','all']):
                assert sorted(r['kappa'] for r in selected if (r['scope'],r['pressure'])==(scope,pressure)) == KAPPAS
    by_weight = {r['checkpoint_files']['model.safetensors']['sha256']:r for r in rows}
    assert len(by_weight)==62
    clip_path = ROOT/'runs/030-2026-09-08-all-models-posthoc-clipping'
    clip = read(clip_path/'results/clipping-points.json')
    verified = read(clip_path/'results/verification.json')
    assert sha(clip_path/'results/clipping-points.json')==verified['output_hashes']['results/clipping-points.json']
    inputs = read(clip_path/'input-manifest.json')
    clip_keys = {}
    for item in inputs['checkpoints']:
        w = next(f for f in item['files'] if f['name']=='model.safetensors')
        if w['sha256'] not in by_weight: continue
        r = by_weight[w['sha256']]
        assert r['source_attempt']==item['source']
        clip_keys[item['checkpoint_content_sha256']]=r
    clipping = []
    for p in clip['points']:
        r = clip_keys.get(p['checkpoint_content_sha256'])
        if not r or not r['comparison_cohort']: continue
        counts = p['counts']
        assert p['R_model']==counts['block_zero_product_count']/counts['model_product_count']
        clipping.append({'id':p['id'], 'checkpoint_key':r['checkpoint_key'], 'model':r['model'],
                         'scope':r['scope'], 'pressure':r['pressure'], 'target':p['dose'],
                         'loss':p['loss'], 'dense_loss_difference':p['loss']-(r['loss']-r['dense_loss_difference']),
                         'sparsity':100*p['R_model'], 'counts':counts, 'coverage':p['coverage'],
                         'main_clipping':r['scope'] in ['0','1'], 'source':p['source']})
    assert len(clipping)==340 and sum(p['main_clipping'] for p in clipping)==40
    p0 = []
    for key in sorted({p['checkpoint_key'] for p in clipping}):
        group = sorted([p for p in clipping if p['checkpoint_key']==key],key=lambda p:p['target'])
        assert [p['target'] for p in group]==[i/10 for i in range(10)]
        r = next(r for r in rows if r['checkpoint_key']==key)
        assert abs(group[0]['loss']-r['loss']) < 5e-4
        p0.append({'checkpoint_key':key,'loss_difference':group[0]['loss']-r['loss']})
    # Join the established instruction cohort through its source attempt, then exact checkpoint key.
    inv_root = ROOT/'analyses/021-2026-09-10-training-results-figures/investigation/data'
    instruction = read(inv_root/'checkpoints.json')
    evidence = read(ROOT/'analyses/018-2026-09-08-results-materials/figure_data.json')
    historical = {p['id']:p for p in evidence['trained']}
    by_attempt = {r['source_attempt']:r for r in rows}
    for item in instruction:
        r = by_attempt[historical[item['checkpoint_evidence_id']]['source']]
        assert r['model']=='14M'
        assert r['counts']['model_product_count']==item['model_products']
        assert math.isclose(r['sparsity'],item['s_model_percent'],abs_tol=1e-12)
        assert math.isclose(item['projection_sparse_gain'],item['all_skips_off_candidate_gm_ms']/item['projection_on_candidate_gm_ms'],rel_tol=1e-12)
        item['checkpoint_key']=r['checkpoint_key']
    assert len(instruction)==30
    coverage = {}
    for size,c in configs.items():
        coverage[size] = {scope:architecture_ceiling(topology, layers=c['num_hidden_layers'],
            hidden_size=c['hidden_size'], ffn_size=c['intermediate_size'], sequence_length=2048,
            vocabulary_size=c['vocab_size']) for scope,topology in [('4','A4-Z'),('7','A7-Z-POST')]}
    data = {'checkpoint_key_definition':'SHA256 of sorted checkpoint filename/content-SHA256 pairs; source attempts and weight hashes cross-check all joins.',
            'checkpoints':rows, 'clipping':clipping, 'clipping_p0_audit':p0, 'instruction':instruction,
            'coverage':coverage, 'sources_sha256':sources, 'comparison_counts':{'14M':32,'70M':22},
            'excluded_comparison':'Eight 14M local L1/OL1 settings remain in the identity table for the established 30-point instruction appendix only.',
            'task_sha256':sha(HERE/'task.md')}
    (HERE/'data/paper-checkpoints.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Identity table: 62 checkpoints; comparison 32+22; 340 post-hoc records; established 30-point instruction cohort.')
    return data


if __name__=='__main__':
    build()
