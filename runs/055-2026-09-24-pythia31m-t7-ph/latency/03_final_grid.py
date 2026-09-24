"""Time hash-verified final-model files without copying optimizer history to the GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_config import condition_specs, load_config, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--condition',required=True)
    p.add_argument('--tag',required=True)
    args = p.parse_args()
    if args.condition not in {r['id'] for r in condition_specs(load_config())}:
        raise ValueError('Unknown scientific condition')
    if not args.tag.replace('-','').isalnum():
        raise ValueError('Simple unique tag required')
    rows = json.loads(args.catalog.read_text())['conditions']
    row, = [r for r in rows if r['id']==args.condition]
    checkpoint = Path(row['checkpoint'])
    if row['training_verification']['status'] != 'verified':
        raise ValueError('Source training attempt has not passed verification')
    for record in row['files']:
        path = (checkpoint/record['path']).resolve()
        if not path.is_relative_to(checkpoint.resolve()):
            raise ValueError('Checkpoint file escapes its directory')
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle,'sha256').hexdigest()
        if digest != record['sha256'] or path.stat().st_size != record['bytes']:
            raise ValueError('Final checkpoint file mismatch: '+record['path'])
    result = dict(status='running',condition=args.condition,catalog=row,replicates=[])
    output = HERE/'artifacts'/f'{args.tag}-{args.condition}-grid.json'
    if output.exists():
        raise FileExistsError('Use a fresh grid tag')
    started = time.time()
    try:
        result['hardware_before'] = subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True).stdout
        for replicate in (1,2,3):
            attempt = f'{args.tag}-{args.condition}-r{replicate}'
            log = HERE/'logs'/(attempt+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            command = [sys.executable,str(HERE/'01_benchmark.py'),'--checkpoint',str(checkpoint),
                '--condition',args.condition,'--attempt',attempt,'--replicate',str(replicate)]
            with log.open('x') as handle:
                process = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=1200)
            manifest = json.loads((HERE/'artifacts/attempts'/attempt/'manifest.json').read_text())
            result['replicates'].append(dict(attempt=attempt,returncode=process.returncode,
                qualified=manifest.get('qualified',False),status=manifest['status']))
            write_json(output,result)
            if process.returncode or not manifest.get('qualified',False):
                raise RuntimeError('Final numerical qualification failed: '+attempt)
        result['hardware_after'] = subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,check=True).stdout
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed',error=str(error))
        raise
    finally:
        result['elapsed_seconds'] = time.time()-started
        write_json(output,result)


if __name__=='__main__':
    main()
