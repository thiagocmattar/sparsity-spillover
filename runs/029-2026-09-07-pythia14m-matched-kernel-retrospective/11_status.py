"""Read-only concise status, with pooled loss and workload throughput."""
import json
import argparse
from datetime import datetime,timezone
from io_utils import RUN, read


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');args=parser.parse_args()
    phases=[p for p in ['smoke','calibration','scientific'] if (RUN/'artifacts'/p/'status.json').exists()]
    for phase in phases if args.all else phases[-1:]:
        folder=RUN/'artifacts'/phase
        if not (folder/'status.json').exists():continue
        status=read(folder/'status.json')
        if status['completed'] and status['elapsed_seconds']>0:
            wall_mean=status['elapsed_seconds']/status['completed']
            status['completed_leaves_per_hour_including_lifecycle']=3600/wall_mean
            status['remaining_seconds_including_lifecycle']=(status['total']-status['completed'])*wall_mean
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
