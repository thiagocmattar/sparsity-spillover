"""Read-only, standard-library status for the two registered follow-up probes."""
import argparse
import json
from pathlib import Path
import time

RUN = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=['k037', 'k038', 'k039', 'k040', 'k041', 'k042', 'k043', 'k044', 'k045', 'k046', 'k047', 'k048', 'k049', 'k050'], required=True)
    parser.add_argument('--attempt-index',choices=['001','002'],default='001')
    args = parser.parse_args()
    component = 'components' if args.candidate in {'k037', 'k041', 'k044', 'k046'} else 'joint'
    if args.candidate == 'k042':
        component = 'projection'
    if args.candidate == 'k050':component='norm'
    attempts = [f'{args.candidate}-{component}-{args.attempt_index}'] + [
        f'{args.candidate}-{condition}-graph-{args.attempt_index}' for condition in ['c01', 'c11', 'c25', 'c30']]
    if args.candidate=='k047':attempts=attempts[1:]
    output = {'candidate': args.candidate, 'terminal': [], 'pending': [], 'current': None}
    for attempt in attempts:
        folder = RUN / 'artifacts' / attempt
        result_path = folder / 'result.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            output['terminal'].append({'attempt': attempt, **{
                key: result[key] for key in ['status', 'elapsed_seconds', 'qualified', 'error'] if key in result}})
            if 'loss' in result:
                output['last_native_loss'] = result['loss']['native']
            if 'timing' in result:
                references = ['native_graph', 'k036_graph', 'no_skip_graph', 'attention_dense_graph']
                if 'k038_graph' in result['timing']:
                    references.append('k038_graph')
                references += [name for name in ['k042_graph','k045_graph','k049_graph','previous_graph'] if name in result['timing']]
                output['terminal'][-1]['ratios_to_selected'] = {
                    reference: result['timing'][reference]['selected_graph']['paired_geomean_speedup']
                    for reference in references}
        else:
            output['pending'].append(attempt)
            status_path = folder / 'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                output['current'] = {'attempt': attempt, 'age_seconds': time.time() - status_path.stat().st_mtime,
                    **{key: value for key, value in status.items() if key not in {'timing', 'loss', 'qualified'}}}
                if 'loss' in status:
                    output['current']['native_loss'] = status['loss']['native']
                events_path = folder / 'events.jsonl'
                if status.get('blocks') and events_path.exists():
                    events = []
                    for line in events_path.read_text().splitlines():
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # The active writer may not have finished its last line.
                    start = next((event['elapsed_seconds'] for event in events
                        if event.get('stage') == 'validation' and 'blocks' not in event), None)
                    if start is not None and status['elapsed_seconds'] > start:
                        rate = status['blocks'] * 2048 / (status['elapsed_seconds'] - start)
                        output['current']['input_tokens_per_second'] = rate
                        output['current']['remaining_validation_seconds'] = (338 - status['blocks']) * 2048 / rate
    print(json.dumps(output))


if __name__ == '__main__':
    main()
