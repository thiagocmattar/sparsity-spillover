"""Read-only per-layer joint-kernel profile breakdown, not speedup estimation."""
import argparse
import json
from pathlib import Path
from statistics import mean

RUN=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);a=p.parse_args()
    if not a.attempt.replace('-','').isalnum():p.error('Simple attempt identity')
    folder=RUN/'artifacts'/a.attempt
    result=json.loads((folder/'result.json').read_text())
    if result['status']!='complete':raise ValueError('Closed profile required')
    rows={}
    for mode in result['modes']:
        events=json.loads((folder/f'{mode}-events.json').read_text())
        events=sorted([event for event in events if event['name'].startswith('void joint<')],key=lambda event:event['start_us'])
        if not events:continue
        if len(events)!=60:raise ValueError('Expected six ordered joints in ten graph replays')
        rows[mode]=[{'layer':layer,'mean_us':mean(event['device_us'] for event in events[layer::6]),
            'min_us':min(event['device_us'] for event in events[layer::6]),
            'max_us':max(event['device_us'] for event in events[layer::6])} for layer in range(6)]
    print(json.dumps({'attempt':a.attempt,'unit':'instrumented CUDA kernel microseconds; not full-model latency','joint_by_layer':rows}))


if __name__=='__main__':main()
