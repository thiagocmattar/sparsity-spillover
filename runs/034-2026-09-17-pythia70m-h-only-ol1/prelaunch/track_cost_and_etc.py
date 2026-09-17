"""Read-only Run034 spending/ETC ledger beside the existing recovery controller."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parents[1]
RECORDS = HERE / 'prelaunch'
spec = importlib.util.spec_from_file_location('cloud034_cost', HERE / '07_cloud.py')
cloud = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cloud)


def report():
    now = time.time()
    snapshot = json.loads((RECORDS / 'latest-status.json').read_text())
    observed = datetime.fromisoformat(snapshot['timestamp']).timestamp()
    assignments = json.loads((RECORDS / 'assignments.json').read_text())
    leases = {label: json.loads((RECORDS / f'lease-{label}.json').read_text()) for label in assignments}
    live = {pod['id']: pod for pod in cloud.cli('pod', 'list', '--all')}
    account = cloud.cli('user')
    completed = 0
    running = 0
    accrued_compute = 0.0
    remaining_compute = 0.0
    current_compute_rate = 0.0
    longest_remaining = 0.0
    deleted = 0
    conditions = []
    warnings = []
    if now-observed > 600:
        warnings.append('Training status is older than ten minutes; inspect the primary controller.')
    by_label = {row['label']: row for row in snapshot['pods']}
    for label, lease in leases.items():
        row = by_label[label]
        rate = float(lease['pod']['costPerHr'])
        live_pod = live.get(lease['pod']['id'])
        receipt_path = RECORDS / 'retrieval' / label / 'receipt.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        end = now
        if receipt.get('pod_deleted'):
            assert receipt.get('locally_verified')
            end = datetime.fromisoformat(receipt['deletion_confirmed_at']).timestamp()
            completed += len(assignments[label])
            deleted += 1
        elif live_pod and live_pod.get('desiredStatus') == 'RUNNING':
            current_compute_rate += float(live_pod['costPerHr'])
        elif not live_pod:
            warnings.append(f'{label}: Pod absent without a confirmed verified-deletion receipt.')
        else:
            warnings.append(f'{label}: Pod is {live_pod.get("desiredStatus")}; inspect artifact recovery.')
        if not receipt.get('pod_deleted') and not row.get('conditions'):
            warnings.append(f'{label}: condition status/ETC is unavailable; total projection is incomplete.')
        accrued_compute += max(0, end-lease['created_request_epoch'])/3600*rate
        remaining = 0.0
        for condition in row.get('conditions', []):
            event = condition.get('last_train', {})
            done = condition.get('verified', False)
            eta = 0.0 if done else max(0, condition.get('remaining_seconds_estimate', 0)-(now-observed))
            remaining = max(remaining, eta)
            if not receipt.get('pod_deleted'):
                completed += int(done)
                running += int(condition.get('attempt_status') == 'running')
            warnings.extend(f'{condition["condition"]}: {warning}' for warning in condition.get('warnings', []))
            conditions.append({'condition': condition['condition'], 'step': event.get('step'), 'total_steps': 712,
                               'task_loss': event.get('task_loss'), 'tokens_per_second': event.get('tokens_per_second'),
                               'remaining_minutes': eta/60, 'verified': done})
        if not receipt.get('pod_deleted'):
            remaining_compute += remaining/3600*rate
            longest_remaining = max(longest_remaining, remaining)
            if now+remaining+900 > lease['deadline_epoch']:
                warnings.append(f'{label}: training ETC plus 15-minute recovery reserve exceeds its stop deadline.')
    recovered = 0
    for path in (RECORDS / 'retrieval').glob('*/checkpoint-files.json'):
        recovered += sum(item['bytes'] for item in json.loads(path.read_text()).values())
    # Estimate excludes unobserved provider billing adjustments. USD5 covers the
    # retired setup attempt; USD1 covers disk, plus 15--30 minutes recovery reserve.
    projected_range = [accrued_compute+remaining_compute+6+current_compute_rate*hours for hours in [.25,.5]]
    if projected_range[1] > 200:
        warnings.append('Projected total with recovery reserve exceeds the USD200 operating allowance.')
    result = {'timestamp': datetime.now(timezone.utc).isoformat(), 'training_observed_at': snapshot['timestamp'],
              'running_conditions': running, 'verified_conditions': completed, 'deleted_pods': deleted,
              'conditions': conditions, 'remaining_training_minutes': longest_remaining/60,
              'last_training_finish_utc_estimate': datetime.fromtimestamp(now+longest_remaining, timezone.utc).isoformat(),
              'current_owned_compute_usd_per_hour': current_compute_rate,
              'current_account_total_usd_per_hour': account['currentSpendPerHr'],
              'account_balance_usd': account['clientBalance'],
              'active_allocation_accrued_compute_estimate_usd': accrued_compute,
              'remaining_training_compute_estimate_usd': remaining_compute,
              'projected_total_with_setup_disk_recovery_reserve_usd': projected_range,
              'operating_allowance_usd': 200, 'checkpoint_bytes_verified_locally': recovered,
              'cost_scope': 'estimate at recorded rates; account total rate also includes existing storage; not a final invoice',
              'warnings': warnings, 'all_owned_pods_retrieved_and_deleted': deleted == len(leases)}
    temporary = RECORDS / 'latest-cost-etc.json.tmp'
    cloud.write(temporary, result)
    temporary.replace(RECORDS / 'latest-cost-etc.json')
    with (RECORDS / 'cost-etc-history.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(result)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assignments = json.loads((RECORDS / 'assignments.json').read_text())
    end = max(json.loads((RECORDS / f'lease-{label}.json').read_text())['deadline_epoch'] for label in assignments)+1800
    while True:
        try:
            result = report()
            print(json.dumps(result), flush=True)
            if args.once or result['all_owned_pods_retrieved_and_deleted'] or time.time() >= end:
                return
        except Exception as error:
            print(json.dumps({'timestamp': datetime.now(timezone.utc).isoformat(), 'tracking_error': str(error)}), flush=True)
            if args.once:
                raise
        if time.time() >= end:
            return
        time.sleep(300)


if __name__ == '__main__':
    main()
