"""Run the fixed retrospective matrix sequentially, with durable fresh attempts."""
import argparse
from datetime import datetime, timezone
import os
import random
import subprocess
import sys
import time
from io_utils import RUN, read, write, record, verify


def jobs(cfg, catalog, checkpoints, phase):
    eligible=[r['id'] for r in catalog if r['status']=='eligible']
    if len(eligible)>cfg['maximum_configurations']:raise ValueError('Configuration budget exceeded')
    if phase=='smoke':
        rows=[{'candidate':c,'condition':'c30','replicate':1,'final':False} for c in eligible]
        rows += [{'candidate':c,'condition':d,'replicate':1,'final':True}
                 for c in ['p0','k050'] for d in cfg['retrospective_conditions'] if d!='c30']
    elif phase=='calibration':
        rows=[{'candidate':c,'condition':'c30','replicate':r,'final':c=='k050'}
              for c in ['p0','k050'] for r in range(1,4)]
    elif phase=='scientific':
        rows=[{'candidate':c,'condition':d,'replicate':r,'final':False}
              for c in eligible for d in cfg['retrospective_conditions'] for r in range(1,4)]
        rows += [{'candidate':c,'condition':d['id'],'replicate':r,'final':True}
                 for c in cfg['final_candidates'] for d in checkpoints for r in range(1,4)]
        random.Random(cfg['timing_seed']).shuffle(rows)
    else:raise ValueError(phase)
    for row in rows:
        row['key']=f"{phase}-{'final' if row['final'] else 'history'}-{row['candidate']}-{row['condition']}-r{row['replicate']}"
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['smoke','calibration','scientific'],required=True)
    p.add_argument('--deadline',required=True,help='Absolute UTC ISO8601; reserve 15 minutes for retrieval')
    a=p.parse_args()
    deadline=datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp()-900
    cfg=read(RUN/'config.json');catalog=read(RUN/'provenance/candidates.json')['configurations']
    inputs=read(RUN/'provenance/inputs.json')
    for row in read(RUN/'provenance/archive.json')['files']:verify(row['snapshot'])
    for row in [inputs['validation']]+[f for c in inputs['checkpoints'] for f in c['files']+c['provenance']]:verify(row)
    folder=RUN/'artifacts'/a.phase;folder.mkdir(parents=True,exist_ok=True)
    plan={'phase':a.phase,'jobs':jobs(cfg,catalog,inputs['checkpoints'],a.phase),
          'sources':[record(RUN/f) for f in ['02_benchmark.py','03_matrix.py','io_utils.py','replay.py','candidate_catalog.py']]+[record(RUN/'config.json'),record(RUN/'provenance/archive.json'),record(RUN/'provenance/candidates.json'),record(RUN/'provenance/inputs.json')]}
    if (folder/'plan.json').exists():
        if read(folder/'plan.json')!=plan:raise ValueError('Frozen phase plan or source changed; do not resume')
    else:write(folder/'plan.json',plan)
    # Exclusive process lock: a crash requires explicit infrastructure recovery.
    lock=folder/'controller.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.monotonic();finished=[];durations=[]
    def status(state,**fields):
        mean=sum(durations)/len(durations) if durations else None
        row={'state':state,'utc':datetime.now(timezone.utc).isoformat(),'phase':a.phase,
             'completed':len(finished),'total':len(plan['jobs']),'elapsed_seconds':time.monotonic()-started,
             'mean_leaf_seconds':mean,'leaves_per_hour':3600/mean if mean else None,
             'remaining_seconds':(len(plan['jobs'])-len(finished))*mean if mean else None,**fields}
        write(folder/'status.json',row);print(row,flush=True)
    try:
        for job in plan['jobs']:
            previous=sorted((RUN/'artifacts/attempts').glob(job['key']+'-a*/result.json'))
            if previous:
                result=read(previous[-1]);finished.append({'job':job,'result':record(previous[-1])})
                durations.append(result['elapsed_seconds']);continue
            if time.time()+cfg['leaf_timeout_seconds']>=deadline:
                status('deadline-reserve');return 2
            attempts=list((RUN/'artifacts/attempts').glob(job['key']+'-a*'))
            attempt=job['key']+f'-a{len(attempts)+1:02d}'
            command=[sys.executable,str(RUN/'02_benchmark.py'),'--candidate',job['candidate'],
                     '--condition',job['condition'],'--replicate',str(job['replicate']),'--attempt',attempt]
            if a.phase=='smoke':command.append('--smoke')
            if job['final']:command.append('--final')
            status('running',current=job,attempt=attempt)
            before=time.monotonic()
            with (folder/(attempt+'.log')).open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=process.wait(timeout=cfg['leaf_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid,signal.SIGKILL);process.wait();code=-9
            path=RUN/'artifacts/attempts'/attempt/'result.json'
            if not path.exists():
                write(path,{'status':'timeout' if code==-9 else 'process-failed','exit_code':code,
                            'arguments':job,'condition':job['condition'],'candidate':job['candidate'],
                            'elapsed_seconds':time.monotonic()-before})
            result=read(path);durations.append(result['elapsed_seconds'])
            finished.append({'job':job,'result':record(path)})
            write(folder/'completed.json',finished)
            status('running',last_result=result['status'],last_qualified=result.get('qualified'),
                   loss=result.get('loss'),last_attempt=attempt)
            if result.get('qualification',{}).get('native_graph') is False:
                status('failed-reference',attempt=attempt);return 3
            fatal=['illegal memory access','device-side assert','Pinned runtime mismatch','Exact RTX5090']
            if any(s in result.get('error','') for s in fatal):
                status('infrastructure-warning',attempt=attempt,error=result['error']);return 4
        write(folder/'completed.json',finished);status('complete');return 0
    finally:lock.unlink()


if __name__=='__main__':raise SystemExit(main())
