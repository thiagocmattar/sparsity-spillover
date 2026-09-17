"""Report recovered measurements without claiming incomplete retrieval is complete."""
import json
from pathlib import Path
from run_config import (EXPECTED_INITIAL_PARAMETER_SHA256, EXPECTED_SCHEDULE_SHA256,
                        condition_specs, load_config, run_code_identity)
from verification import (_require_train_events, _require_validation,
                          _require_diagnostics, verify_attempt)

HERE=Path(__file__).resolve().parent
config=load_config()
conditions={row['id']:row for row in condition_specs(config)}
salvage=json.loads((HERE/'prelaunch/salvage-receipt.json').read_text())
rows=[]
for recovered in salvage['conditions']:
    attempt=HERE/'artifacts/attempts'/recovered['attempt']
    manifest=json.loads((attempt/'manifest.json').read_text())
    metrics=json.loads((attempt/'metrics.json').read_text())
    condition=conditions[manifest['condition']['id']]
    assert manifest['condition']==metrics['condition']==condition
    assert manifest['status']=='completed' and manifest['completed_steps']==712
    assert manifest['input_tokens']==1493172224 and not manifest['gradient_overflow_steps']
    assert manifest['initial_parameter_sha256']==EXPECTED_INITIAL_PARAMETER_SHA256
    assert manifest['training_schedule_hash']==EXPECTED_SCHEDULE_SHA256
    assert manifest['run_code']['content_sha256']==run_code_identity()['content_sha256']
    assert manifest['code']['git_dirty'] is False
    assert manifest['activation_pressure']=={'method':'orthogonal_l1','sites':['h'],'weight':1.0,'step_budget':1.0,'eps':1e-12}
    geometry=_require_train_events(attempt,condition)
    _require_validation(metrics,attempt.name)
    activation,logical=_require_diagnostics(attempt,metrics,config)
    full=verify_attempt(condition['id']) if recovered['inventory_complete'] else None
    row={'condition':condition,'pod':recovered['pod'],'attempt':attempt.name,
         'metadata_checks':'passed','complete_local_verification':full,
         'missing_bytes':recovered['missing_bytes'],
         'final_validation_loss':metrics['validation']['final']['loss'],
         'final_train_loss':metrics['training']['task_loss_final'],
         'R_model':logical['measured']['R_model'],'R_block':logical['measured']['R_block'],
         'finished_at':manifest['finished_at'],'ol1':geometry}
    rows.append(row)
    print(json.dumps({k:row[k] for k in ['pod','metadata_checks','missing_bytes','final_validation_loss','R_model','finished_at']}))
rows.sort(key=lambda r:r['condition']['gate_threshold'])
result={'status':'partial_recovery','conditions':rows,'unretrieved_condition':'a7-h-ol1-kappa-0p5',
        'scope':'All four metadata sets pass the original train/validation/diagnostic checks. Only conditions with complete_local_verification have their entire checkpoint and artifact inventory locally verified.'}
(HERE/'prelaunch/recovered-metadata-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
