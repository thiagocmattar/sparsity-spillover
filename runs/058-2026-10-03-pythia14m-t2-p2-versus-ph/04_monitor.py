"""Read-only status snapshot for bounded monitoring."""
import json
from run_config import RUN_DIR

if __name__=='__main__':
    rows=[]
    for path in sorted((RUN_DIR/'artifacts/attempts').glob('*/manifest.json')):
        manifest=json.loads(path.read_text())
        events=path.parent/'events.jsonl'
        latest=None
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue  # An active writer may have a partial last line.
                if row.get('event')=='train':latest=row
        info={'attempt':path.parent.name,'status':manifest['status']}
        if latest:
            remaining=712-int(latest['step'])
            info.update(step=latest['step'],task_loss=latest['task_loss'],
                throughput=latest['tokens_per_second'],
                training_remaining_seconds=remaining*latest['step_wall_seconds'],
                elapsed_seconds=latest['elapsed_seconds'])
        rows.append(info)
    print(json.dumps(rows,indent=2))
