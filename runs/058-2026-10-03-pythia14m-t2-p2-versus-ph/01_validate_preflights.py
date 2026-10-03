"""Review the four actual preflights before a fresh scientific initialization."""
import json
from run_config import RUN_DIR,load_config,condition_specs,run_code_identity

if __name__ == '__main__':
    for condition in condition_specs(load_config()):
        path=RUN_DIR/'prelaunch'/f"remote-preflight-{condition['id']}.json"
        row=json.loads(path.read_text())
        if row['status']!='passed' or row['run_code']!=run_code_identity():
            raise RuntimeError(f'Preflight failed or source changed: {path}')
        # An unexpectedly slow placement requires investigation before committing
        # the full training budget. Historical H100 h-only median was ~3.39 s.
        if row['median_boundary_seconds']>6.8:
            raise RuntimeError(f'Placement slower than review threshold: {path}')
        if any(r['optimizer_step_skipped'] for r in row['boundary_health']):
            raise RuntimeError(f'Skipped preflight update: {path}')
    print('All four production preflights passed',flush=True)
