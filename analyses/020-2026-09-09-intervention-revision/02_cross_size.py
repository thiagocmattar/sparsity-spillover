"""Identity-checked FP16 endpoint/clipping join and observed ReLU contrasts."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=HERE/'data'
FAMILIES={'A0','A1-H','A4-OL1','A7-OL1'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    paths=[ROOT/'analyses/018-2026-09-08-results-materials/figure_data.json',
           ROOT/'analyses/019-2026-09-09-manuscript-rewrite-audit/training-audit.json',
           ROOT/'runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json']
    bundle,audit,clips=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    endpoints={r['id']:r for r in audit['endpoints']}
    baseline={r['scale']:r['loss'] for r in bundle['trained'] if r['family']=='A0'}
    a7={r['scale']:r['R_model_max_fraction'] for r in bundle['ceilings'] if r['family']=='A7'}
    rows=[]
    manifests={}
    for r in bundle['trained']:
        if r['family'] not in FAMILIES: continue
        endpoint=endpoints[r['id']]
        manifest_path=ROOT/r['source']/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['checkpoints']['final']['content_sha256']==endpoint['checkpoint_content_sha256']
        manifests[r['id']]=dict(path=manifest_path.relative_to(ROOT).as_posix(),sha256=sha(manifest_path),
                               final_checkpoint_content_sha256=manifest['checkpoints']['final']['content_sha256'])
        assert endpoint['evaluation_precision']=='FP16'
        assert all(endpoint[k]==r[k] for k in ['id','source','scale','family','dose','loss'])
        rows.append(dict(id=r['id'], checkpoint_id=r['id'], checkpoint_sha256=endpoint['checkpoint_content_sha256'],
                         scale=r['scale'],family=r['family'],trained_dose=r['dose'],clipping_p=None,kind='trained',
                         precision='FP16',source=r['source'],loss=r['loss'],delta_loss_vs_A0=r['loss']-baseline[r['scale']],
                         S_model=r['R_model'],S_block=r['R_model']/a7[r['scale']],
                         zero_products=r['counts']['block_zero_product_count'],model_products=r['counts']['model_product_count'],
                         block_products=r['counts']['block_product_count'],R_arch_actual=r['ceiling']['R_model_max_fraction'],
                         R_arch_A7=a7[r['scale']],inference_sites=','.join(r['ceiling']['active_sites']),
                         delta_loss_from_p0=None,delta_S_from_p0=None))
    originals={r['id']:r for r in rows}
    p0={r['source_checkpoint_id']:r for r in clips['points'] if r['dose']==0}
    discrepancies=[]
    for r in clips['points']:
        if r['family'] not in FAMILIES: continue
        endpoint=originals[r['source_checkpoint_id']]
        assert r['checkpoint_content_sha256']==endpoint['checkpoint_sha256']
        assert (r['scale'],r['family'],r['training_parameter'])==(endpoint['scale'],endpoint['family'],endpoint['trained_dose'])
        assert r['coverage']['batches']==338 and r['coverage']['excluded_tail_tokens']==1444
        assert sha(ROOT/r['source'])==clips['sources'][r['source']]
        base=p0[r['source_checkpoint_id']]
        row=dict(endpoint)
        row.update(id=r['id'],kind='clipped',clipping_p=r['dose'],source=r['source'],loss=r['loss'],
                   delta_loss_vs_A0=r['loss']-baseline[r['scale']],S_model=r['R_model'],S_block=r['R_model']/a7[r['scale']],
                   zero_products=r['counts']['block_zero_product_count'],model_products=r['counts']['model_product_count'],
                   block_products=r['counts']['block_product_count'],R_arch_actual=r['ceiling']['R_model_max_fraction'],
                   inference_sites=','.join(r['normalization_sites']),delta_loss_from_p0=r['loss']-base['loss'],
                   delta_S_from_p0=r['R_model']-base['R_model'])
        rows.append(row)
        if r['dose']==0:
            discrepancies.append(dict(checkpoint_id=endpoint['id'],canonical_source=endpoint['source'],clipping_source=r['source'],
                                      precision='FP16',loss_drift=r['loss']-endpoint['loss'],S_drift=r['R_model']-endpoint['S_model']))
    assert len(rows)==396 and len({r['id'] for r in rows})==396
    for r in rows:
        assert r['S_model']==r['zero_products']/r['model_products']
        assert abs(r['S_block']-r['zero_products']/r['block_products'])<1e-14
    with (DATA/'cross_size_interventions.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    relu=[]
    for scale in ['14M','70M','410M']:
        by={family:sorted([r for r in rows if r['scale']==scale and r['family']==family and r['kind']=='clipped'],key=lambda r:r['clipping_p']) for family in ['A0','A1-H']}
        paired=[]
        for g,h in zip(by['A0'],by['A1-H']):
            assert g['clipping_p']==h['clipping_p']
            paired.append(dict(p=g['clipping_p'],gelu=g,relu=h,loss_difference=h['loss']-g['loss'],
                               sparsity_difference=h['S_model']-g['S_model'],
                               incremental_loss_difference=h['delta_loss_from_p0']-g['delta_loss_from_p0'],
                               relu_dominates=h['loss']<g['loss'] and h['S_model']>=g['S_model']))
        # Exhaustive observed pairs with <=1 sparsity percentage point mismatch;
        # no interpolation or automatically selected "winner" model.
        close=[dict(gelu_id=g['id'],relu_id=h['id'],gelu_loss=g['loss'],relu_loss=h['loss'],
                    gelu_S=g['S_model'],relu_S=h['S_model'],mismatch_pp=100*(h['S_model']-g['S_model']))
               for g in by['A0'] for h in by['A1-H'] if abs(h['S_model']-g['S_model'])<=.01]
        relu.append(dict(scale=scale,base_relu_cost=originals[f'{scale}:A1-H:None']['loss']-baseline[scale],
                         matched_policy=paired,observed_pairs_within_one_pp=close))
    result=dict(sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths},rows=len(rows),trained=36,clipped=360,
                p0_discrepancies=discrepancies,relu=relu,canonical_manifest_identities=manifests,
                precision_provenance='Analysis 019 endpoint audit and Analysis 018 source reconstruction verify canonical and clipping FP16 coverage. p=0 is measured independently; differences are preserved.',
                p0_implementation='Canonical endpoints use eager exact-site evaluation; older clipping records use calibration wrappers and separate evaluations. Same checkpoint bytes, dtype and full validation coverage do not imply bitwise identical execution.')
    (DATA/'cross-size-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    dest=ROOT/'manuscript/draft/revision-v2/data';dest.mkdir(exist_ok=True)
    shutil.copyfile(DATA/'cross_size_interventions.csv',dest/'cross_size_interventions.csv')
    print('396 FP16 records: 36 canonical endpoints + 360 clipping evaluations; checkpoint identities and pooled counts verified.')
    for item in relu:
        print(item['scale'],'base cost',item['base_relu_cost'])
        for p in item['matched_policy']:
            print(' p',p['p'],'G/H loss',round(p['gelu']['loss'],4),round(p['relu']['loss'],4),
                  'G/H S%',round(100*p['gelu']['S_model'],3),round(100*p['relu']['S_model'],3),
                  'G/H dloss',round(p['gelu']['delta_loss_from_p0'],4),round(p['relu']['delta_loss_from_p0'],4))


if __name__=='__main__':main()
