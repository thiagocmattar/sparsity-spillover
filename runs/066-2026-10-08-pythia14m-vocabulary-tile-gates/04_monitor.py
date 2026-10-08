"""Read-only status snapshot for bounded monitoring."""
import json
from statistics import median
from time import time
from run_config import RUN_DIR,PREFLIGHT_DIR,load_config

if __name__=='__main__':
    config=load_config();rows=[]
    for path in sorted((RUN_DIR/'artifacts/attempts').glob('*/manifest.json')):
        manifest=json.loads(path.read_text())
        events=path.parent/'events.jsonl'
        latest=None;updates=[];validations=set();diagnostics=set()
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue  # An active writer may have a partial last line.
                if row.get('event')=='train':latest=row;updates.append(row)
                if row.get('event')=='validation':validations.add(row['step'])
                if row.get('event')=='activation_diagnostic':diagnostics.add(row['step'])
        info={'attempt':path.parent.name,'status':manifest['status']}
        if latest:
            remaining=config['training']['max_steps']-int(latest['step'])
            step_seconds=median(r['step_wall_seconds'] for r in updates[-10:])
            calibration=PREFLIGHT_DIR/f"remote-preflight-{latest['worker_id']}.json"
            timing=json.loads(calibration.read_text()) if calibration.exists() else {}
            pending_validation=set(config['validation']['steps'])-validations
            pending_diagnostics=set(config['diagnostics']['steps'])-diagnostics
            posthoc_seconds=(len(pending_validation)*timing.get('validation_seconds',0)
                +len(pending_diagnostics)*timing.get('diagnostic_seconds',{}).get('activation',0)
                +timing.get('diagnostic_seconds',{}).get('logical',0))
            eta=remaining*step_seconds+posthoc_seconds
            if manifest['status']=='completed':eta=0
            info.update(step=latest['step'],task_loss=latest['task_loss'],
                throughput=latest['tokens_per_second'],
                training_remaining_seconds=remaining*step_seconds,
                estimated_completion_seconds=eta,
                etc_scope='training and calibrated remaining validation/diagnostics; transfer/verification reported separately',
                pending_validation_steps=sorted(pending_validation),
                pending_diagnostic_steps=sorted(pending_diagnostics),
                stale_event_seconds=time()-events.stat().st_mtime,
                elapsed_seconds=latest['elapsed_seconds'])
        rows.append(info)
    print(json.dumps(rows,indent=2))
