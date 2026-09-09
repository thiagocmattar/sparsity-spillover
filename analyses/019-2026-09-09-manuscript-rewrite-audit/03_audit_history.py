"""Audit the five retained h-only-pressure controls without changing cohorts."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIOR=ROOT/'analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites'
RUN=ROOT/'runs/029-2026-09-07-pythia14m-matched-kernel-retrospective'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    spec=importlib.util.spec_from_file_location('analysis009_build',PRIOR/'01_build.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data=module.build_figure_data()
    assert data==read(PRIOR/'figure_data.json')
    inputs=read(RUN/'provenance/inputs.json')['checkpoints']
    historical=[p for p in inputs if p['family']=='A4+OL1@h']
    assert len(historical)==5
    rows=[]
    for p in historical:
        kappa=p['dose']
        counterparts={s:next(r for r in data['series'] if r['series_id']==s and r['kappa']==kappa)
                      for s in ('run011_a4','run012_h_only','run015_four_site')}
        h=counterparts['run012_h_only']
        assert h['realized_pressure_sites']==['h']
        canonical=p['canonical_logical_products']
        assert math.isclose(canonical['measured']['R_model'],h['R_model'],abs_tol=1e-14)
        assert math.isclose(canonical['coverage']['loss'],h['logical_product_counts']['diagnostic_validation_loss'],abs_tol=1e-12)
        assert p['source'].endswith(h['attempt_id'])
        for source in p['provenance']:
            path=RUN/source['path']
            if os.name == 'nt':
                path=Path('\\\\?\\' + str(path))
            assert path.stat().st_size==source['bytes'] and sha(path)==source['sha256']
        settings={s:{'loss':r['logical_product_counts']['diagnostic_validation_loss'],
                     'S_model_percent':100*r['R_model'],'attempt_id':r['attempt_id']}
                  for s,r in counterparts.items()}
        weight=next(f for f in p['original_files'] if f['path'].endswith('model.safetensors'))
        rows.append({'condition':p['id'],'kappa':kappa,'source':p['source'],
                     'weight_sha256':weight['sha256'],'settings':settings,
                     'four_site_minus_h_only_loss':settings['run015_four_site']['loss']-settings['run012_h_only']['loss'],
                     'four_site_minus_h_only_sparsity_pp':settings['run015_four_site']['S_model_percent']-settings['run012_h_only']['S_model_percent']})
    result={'source_hashes':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in
                            (PRIOR/'01_build.py',PRIOR/'figure_data.json',RUN/'provenance/inputs.json')},
            'matching':data['matched_identity'],'coverage':data['coverage'],
            'realization_audit':data['realization_audit'],'rows':rows,
            'verified':'Analysis 009 exactly reconstructed from original configs, manifests, counts and inherited capture code; all five Run 029 historical checkpoint identities reconciled.',
            'cohort_policy':'Remain outside the 30-condition 14M manuscript cohort and 30-checkpoint runtime summary.',
            'identification_limit':'All five historical controls retain N4 one-sided thresholding and OL1 at h only. Expanding pressure from h to four sites changes each old site-layer coefficient from 1/6 to 1/24. These controls do not populate the proposed fixed-coefficient N4/N7 by P4/P7 design.',
            'loss_convention':'This audit uses diagnostic FP16 losses paired with logical counts. Analysis 009 also preserves terminal validation losses; its terminal-loss deltas are not substituted for these diagnostic deltas.'}
    (HERE/'historical-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    for r in rows:print(r['condition'],r['kappa'],r['four_site_minus_h_only_loss'],r['four_site_minus_h_only_sparsity_pp'])
    print('Five historical controls verified; main cohort unchanged.')


if __name__=='__main__':
    main()
