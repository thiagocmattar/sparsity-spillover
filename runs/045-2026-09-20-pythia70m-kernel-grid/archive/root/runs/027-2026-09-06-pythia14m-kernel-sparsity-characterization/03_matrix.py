"""Fixed finite characterization matrix; one fresh process per checkpoint/repeat."""
import subprocess
import sys
import time
import numpy as np
from run027_common import RUN,read_json,write_json

cfg=read_json(RUN/'config.json')
cohort=read_json(RUN/'prelaunch/inputs.json')['checkpoints']
started=time.monotonic()
order=[]
for replicate in range(1,cfg['fresh_processes']+1):
    ids=np.random.default_rng(cfg['cohort_order_seed']+replicate).permutation([r['id'] for r in cohort])
    order.extend((replicate,str(condition)) for condition in ids)
write_json(RUN/'runtime/matrix-order.json',order)
for done,(replicate,condition) in enumerate(order):
    attempt=f'{condition}-r{replicate}'
    subprocess.run([sys.executable,'-u',str(RUN/'01_benchmark.py'),'--condition',condition,
        '--replicate',str(replicate),'--attempt',attempt],check=True,timeout=900)
    result=read_json(RUN/'artifacts'/attempt/'result.json')
    elapsed=time.monotonic()-started
    progress={'completed':done+1,'total':len(order),'condition':condition,'replicate':replicate,
        'elapsed_seconds':elapsed,'etc_seconds':elapsed*(len(order)-done-1)/(done+1),
        'loss':result['loss'],'qualified':result['qualified'],
        'input_tokens_per_second':{mode:2048000/row['median_host_ms'] for mode,row in result['timing']['native'].items()}}
    write_json(RUN/'runtime/matrix-status.json',progress)
    print(progress,flush=True)
print('MATRIX_COMPLETE',flush=True)
