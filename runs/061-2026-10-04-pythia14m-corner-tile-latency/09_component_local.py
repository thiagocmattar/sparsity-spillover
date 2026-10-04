"""Transfer only component sources to the existing isolated local WSL GPU."""
import io
import argparse
import json
import subprocess
import tarfile
from io_utils import RUN, record, write, fs

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--tag',default='001')
    args=parser.parse_args()
    if not args.tag.isalnum():
        parser.error('Simple unique tag required')
    name='native' if args.native else 'component'
    linux='/home/researcher/sparsity-spillover/run061-'+name+'-'+args.tag
    files=[RUN/n for n in ['00_component.py','kernel.py','joint.cu','tile_gate.py','io_utils.py']]
    files += [p for p in (RUN/'archive/root/src').rglob('*.py')]
    if args.native:
        files += [RUN/n for n in ['10_native_smoke.py','topology.py','replay.py','config.json','provenance/inputs.json']]
        inputs=__import__('io_utils').read(RUN/'provenance/inputs.json')
        files += [RUN/inputs['validation']['path']]
        files += [RUN/r['path'] for c in inputs['checkpoints'] for r in c['files']]
        files += [RUN/r['snapshot']['path'] for r in __import__('io_utils').read(RUN/'provenance/archive.json')['files']]
    files=sorted(set(files))
    inventory=[record(p) for p in files]
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for path in files:
            archive.add(fs(path),arcname=path.relative_to(RUN).as_posix(),recursive=False)
        data=json.dumps(inventory).encode()
        entry=tarfile.TarInfo('upload-inventory.json');entry.size=len(data)
        archive.addfile(entry,io.BytesIO(data))
    subprocess.run(['wsl','-d','SparsityGPU','--exec','mkdir',linux],check=True)
    subprocess.run(['wsl','-d','SparsityGPU','--exec','tar','-xf','-','-C',linux],input=payload.getvalue(),check=True)
    code="import hashlib,json,pathlib; r=pathlib.Path('.'); rows=json.loads((r/'upload-inventory.json').read_text()); assert all((r/x['path']).stat().st_size==x['bytes'] and hashlib.sha256((r/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in rows); print('Verified',len(rows),'source files')"
    script='10_native_smoke.py' if args.native else '00_component.py'
    command='source /opt/sparsity-gpu/activate.sh; cd '+linux+'; python -c '+__import__('shlex').quote(code)+' && exec timeout --kill-after=10s 600s python -u '+script
    log=RUN/f'prelaunch/{name}-local-{args.tag}.log';log.parent.mkdir(exist_ok=True)
    with log.open('xb') as stream:
        result=subprocess.run(['wsl','-d','SparsityGPU','--exec','bash','-c',command],stdout=stream,stderr=subprocess.STDOUT,timeout=660)
    if result.returncode:
        raise RuntimeError('Local component failed; see persistent log')
    output='native-cuda-smoke.json' if args.native else 'component-cuda.json'
    response=subprocess.run(['wsl','-d','SparsityGPU','--exec','cat',linux+'/prelaunch/'+output],capture_output=True,check=True)
    data=json.loads(response.stdout)
    local_output=output if args.tag=='001' else output.replace('.json','-'+args.tag+'.json')
    (RUN/'prelaunch'/local_output).write_bytes(response.stdout)
    write(RUN/f'prelaunch/{name}-local-receipt-{args.tag}.json',{'sources':inventory,'result':record(RUN/'prelaunch'/local_output),
          'local_scope':'bounded development qualification, no research timing cohort','linux_root':linux})
    print(json.dumps({'status':data['status'],'checks':len(data.get('checks',data.get('rows',[]))),'seconds':data['seconds'],'gpu':data['gpu']}))


if __name__=='__main__':
    main()
