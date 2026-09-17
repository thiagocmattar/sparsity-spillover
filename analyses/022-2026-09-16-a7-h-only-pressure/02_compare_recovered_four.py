"""Interim comparison of four fully retrieved conditions; kappa=0.5 is absent."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
paths={
    'none':ROOT/'runs/013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json',
    'h':ROOT/'runs/032-2026-09-16-pythia14m-a7-h-only-ol1/prelaunch/recovered-metadata-audit.json',
    'all7':ROOT/'runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/artifacts/verification.json',
}
sources={k:json.loads(p.read_text()) for k,p in paths.items()}
for arm in ('none','all7'):assert sources[arm]['status']=='verified'
hs=[r['complete_local_verification'] for r in sources['h']['conditions']]
assert len(hs)==4 and all(r and r['status']=='verified' for r in hs)
raw={'none':sources['none']['conditions'],'h':hs,'all7':sources['all7']['conditions']}
kappas=(0.0,0.01,0.05,0.1)
assert {r['condition']['gate_threshold'] for r in hs}==set(kappas)
rows=[]
identities=set()
for kappa in kappas:
    group={arm:next(r for r in records if r['condition']['gate_threshold']==kappa) for arm,records in raw.items()}
    for arm,r in group.items():
        c=r['condition']
        assert c['topology_id']=='A7-Z-POST'
        assert set(c['active_sites'])=={'a','m','h','z','q_post','k_post','v'}
        assert set(c['pressure_sites'])==set([] if arm=='none' else ['h'] if arm=='h' else c['active_sites'])
        assert r['completed_steps']==712 and r['input_tokens']==1493172224
        identities.add((r['initial_parameter_sha256'],r['training_schedule_sha256']))
    rows.append({'kappa':kappa,'arms':{arm:{'validation_loss':r['final_validation_loss'],'R_model':r['R_model'],'attempt_id':r['attempt_id']} for arm,r in group.items()},
                 'h_minus_none_loss':group['h']['final_validation_loss']-group['none']['final_validation_loss'],
                 'h_minus_none_R_model_pp':100*(group['h']['R_model']-group['none']['R_model'])})
assert len(identities)==1
result={'status':'interim_four_conditions','missing_kappa':0.5,'coverage':'Four thresholds, one matched seed; every included h-only condition passes the full original local verifier. Each final validation covers 500 documents, 338 complete 2048-token blocks, 1444 excluded tail tokens.',
        'sources':{arm:{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for arm,p in paths.items()},'rows':rows}
output=HERE/'artifacts'
output.mkdir(exist_ok=True)
(output/'recovered-four-comparison.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
lines=['| Kappa | Validation loss: none / h / all7 | R_model (%): none / h / all7 |','|---:|---:|---:|']
for row in rows:
    vals=[row['arms'][arm] for arm in ('none','h','all7')]
    losses=' / '.join(f"{v['validation_loss']:.6f}" for v in vals)
    products=' / '.join(f"{100*v['R_model']:.4f}" for v in vals)
    lines.append(f"| {row['kappa']:g} | {losses} | {products} |")
(output/'recovered-four-comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
