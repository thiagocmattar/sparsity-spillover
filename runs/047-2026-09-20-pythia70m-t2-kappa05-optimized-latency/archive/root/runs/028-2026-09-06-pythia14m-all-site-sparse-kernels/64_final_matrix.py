"""Run the fixed105 fresh-process measurements; preserve every outcome."""
import json
import random
import subprocess
import time
from datetime import datetime,timezone
from common import RUN,manifest,read_json,write_json,record,verify_record


def main():
    dest=RUN/'artifacts/final-matrix-001';dest.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();policy=RUN/'final-policy-001.json'
    frozen=read_json(policy)
    for row in frozen['sources']+frozen['dependencies']['files']:verify_record(row)
    rng=random.Random(2504);schedule=[]
    for replicate in range(1,4):
        identities=[row['id'] for row in manifest()['checkpoints']]
        if len(identities)!=35 or len(set(identities))!=35:raise ValueError('Expected all35 checkpoints')
        rng.shuffle(identities)
        schedule.extend({'condition':c,'replicate':replicate,'attempt':f'final-{c}-p{replicate}-001'} for c in identities)
    write_json(dest/'plan.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'policy':record(policy),'controller':record(RUN/'64_final_matrix.py'),
        'wrapper':record(RUN/'61_study.sh'),'schedule':schedule,'leaf_timeout_seconds':600})
    completed=[]
    for index,item in enumerate(schedule):
        status={'stage':'running','completed':index,'total':len(schedule),'current':item,
                'elapsed_seconds':time.monotonic()-started,'outcomes':completed}
        if completed:status['remaining_seconds']=(len(schedule)-index)*status['elapsed_seconds']/index
        write_json(dest/'status.json',status);print(json.dumps(status),flush=True)
        command=['timeout','--signal=TERM','--kill-after=30s','600','bash',str(RUN/'61_study.sh'),
                 '--condition',item['condition'],'--replicate',str(item['replicate']),'--attempt',item['attempt']]
        begin=time.monotonic()
        with (dest/f"{item['attempt']}.log").open('x') as log:
            returncode=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        result_path=RUN/'artifacts'/item['attempt']/'result.json'
        result=read_json(result_path) if result_path.exists() else {}
        completed.append({**item,'returncode':returncode,'status':result.get('status','missing_result'),
            'qualified':result.get('qualified',{}),'loss':result.get('loss',{}),
            'elapsed_seconds':time.monotonic()-begin})
        write_json(dest/'outcomes.json',completed)
    status={'status':'complete','completed':len(completed),'total':len(schedule),
            'elapsed_seconds':time.monotonic()-started,'outcomes':completed}
    write_json(dest/'status.json',status);write_json(dest/'result.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
