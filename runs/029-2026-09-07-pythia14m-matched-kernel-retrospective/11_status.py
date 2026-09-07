"""Read-only concise status, with pooled loss and workload throughput."""
import json
from datetime import datetime,timezone
from io_utils import RUN, read


def main():
    for phase in ['smoke','calibration','scientific']:
        folder=RUN/'artifacts'/phase
        if not (folder/'status.json').exists():continue
        status=read(folder/'status.json')
        if (folder/'completed.json').exists():
            completed=read(folder/'completed.json')
            if completed:
                last=read(RUN/completed[-1]['result']['path'])
                status['latest_result']={k:last.get(k) for k in ['candidate','condition','status','qualified','loss','elapsed_seconds','peak_allocated_bytes','error']}
                status['latest_timing']=last.get('timing')
        attempt=status.get('attempt')
        if attempt:
            path=RUN/'artifacts/attempts'/attempt/'status.json'
            if path.exists():
                leaf=read(path);status['leaf']=leaf
                if leaf.get('blocks') and leaf.get('elapsed_seconds',0)>0:
                    status['leaf_input_tokens_per_wall_second']=leaf['blocks']*2048/leaf['elapsed_seconds']
                status['leaf_age_seconds']=(datetime.now(timezone.utc)-datetime.fromisoformat(leaf['utc'].replace('Z','+00:00'))).total_seconds()
        print(json.dumps(status,allow_nan=False))


if __name__=='__main__':main()
