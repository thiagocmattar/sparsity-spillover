"""Run055 creation-to-confirmed-deletion rental estimate from retained receipts."""
from pathlib import Path
from datetime import datetime
import json
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def stamp(value):return datetime.fromisoformat(value.replace('Z','+00:00'))

def main():
    fleet=read('parallel-fleet-001.json')
    records=fleet['pods']+read('parallel-connections-001.json')
    records += [read(name) for name in ['retired-latency-pod-003.json','latency-allocation-004.json']]
    allocations={row['id']:row for row in records}
    terminated={}
    for path in HERE.glob('terminated-*.json'):
        row=read(path.name)
        if row['status']=='terminated':terminated[row['pod']]=(row,path.name)
    assert len(allocations)==10
    rows=[]
    for pod,allocation in sorted(allocations.items()):
        receipt,name=terminated[pod]
        end=next(receipt[k] for k in ['confirmed_absent_utc','verified_utc','utc'] if k in receipt)
        hours=(stamp(end)-stamp(allocation['created_at'])).total_seconds()/3600
        assert hours>=0
        rows.append(dict(pod=pod,name=allocation['name'],created_at=allocation['created_at'],
            confirmed_absent_at=end,hours_upper=hours,hourly_usd=allocation['cost_per_hour'],
            compute_usd_upper=hours*allocation['cost_per_hour'],termination_receipt=name))
    audit=read('final-resource-audit.json')
    assert not ({p['id'] for p in audit['pods']}&set(allocations))
    total=sum(row['compute_usd_upper'] for row in rows)
    result=dict(status='complete',pods=rows,compute_usd_upper=total,approved_total_envelope_usd=fleet['maximum_total_usd'],
        scope='Rental estimate at live allocation prices, from creation through confirmed absence. Includes provisioning and retry time conservatively; not an invoice. Storage and account credits are not inferred.',
        resource_audit='final-resource-audit.json',task_created_resources_remaining=[])
    assert total<fleet['maximum_total_usd']
    (HERE/'compute-closeout.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(terminated_pods=len(rows),compute_usd_upper=total)))
if __name__=='__main__':main()
