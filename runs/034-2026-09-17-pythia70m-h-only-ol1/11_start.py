"""Start each explicitly assigned condition after its own exact preflight passes."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cloud034',HERE/'07_cloud.py')
cloud=importlib.util.module_from_spec(spec);spec.loader.exec_module(cloud)


def start(label):
    assignments=json.loads((HERE/'prelaunch/assignments.json').read_text())[label]
    with cloud.connect(label) as c:
        for gpu,condition in enumerate(assignments):
            control=cloud.CONTROL+'/'+condition
            cloud.command(c,'mkdir -p '+control)
            script=f'''#!/bin/bash
set -euo pipefail
trap 'echo $? > {control}/exit-code' EXIT
while [ ! -f {cloud.CONTROL}/environment-ready ] || [ ! -f {cloud.CONTROL}/cache-ready ] || [ ! -f {cloud.CONTROL}/initialization-ready ]; do sleep 10; done
cd {cloud.REMOTE}
export PYTHONPATH={cloud.REMOTE_RUN}:{cloud.REMOTE}/src
export CUDA_VISIBLE_DEVICES={gpu}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
/opt/run034-venv/bin/python -c 'from run_config import git_identity; print(git_identity())' > {control}/source-verification.log
/opt/run034-venv/bin/python -u {cloud.REMOTE_RUN}/05_remote_preflight.py --condition {condition} > {control}/preflight.log 2>&1
touch {control}/preflight-passed
/opt/run034-venv/bin/python -u {cloud.REMOTE_RUN}/02_train.py --worker {condition} > {control}/train.log 2>&1
/opt/run034-venv/bin/python {cloud.REMOTE_RUN}/03_verify.py --condition {condition} > {control}/verification.log 2>&1
touch {control}/verified
'''
            with c.open_sftp() as s:
                with s.open(control+'/execute.sh','w') as h:h.write(script)
            cloud.command(c,f'test ! -f {control}/pid && {{ setsid bash {control}/execute.sh > {control}/orchestration.log 2>&1 < /dev/null & echo $! > {control}/pid; }}')
            print(json.dumps({'pod_label':label,'gpu_index':gpu,'condition':condition,'status':'armed_after_input_and_preflight_gates'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();start(a.label)
