"""Copy a hash-sealed minimal checkout to isolated WSL for bounded GPU qualification."""
import argparse
import io
import json
import subprocess
import tarfile
from io_utils import RUN, REPO, SOURCE, read, record, write, verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--full',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--trace',action='store_true')
    args=parser.parse_args()
    if not args.tag.isalnum():raise ValueError('Simple unique tag required')
    root='/home/researcher/sparsity-spillover/run068_head-'+args.tag
    rows=read(RUN/'provenance/upload-inventory.json')
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for row in rows:
            path=verify(row)
            archive.add(path,arcname=row['path'],recursive=False)
        path=RUN/'provenance/upload-inventory.json'
        archive.add(path,arcname=path.relative_to(REPO).as_posix(),recursive=False)
    subprocess.run(['wsl','-d','SparsityGPU','--exec','mkdir',root],check=True)
    subprocess.run(['wsl','-d','SparsityGPU','--exec','tar','-xf','-','-C',root],input=payload.getvalue(),check=True)
    relative=RUN.relative_to(REPO).as_posix()
    training=SOURCE.relative_to(REPO).as_posix()
    commands=[f'python -u {relative}/05_verify.py --inputs-only --development',
              f'python -u {training}/24_cuda_smoke.py --tag {args.tag}',
              f'python -u {relative}/00_component.py --tag {args.tag}']
    if args.trace:commands.append(f'python -u {relative}/10_trace_numerics.py --tag {args.tag}')
    if args.full or args.smoke:
        for c in read(RUN/'provenance/inputs.json')['checkpoints']:
            commands.append(f'python -u {relative}/02_benchmark.py --model {c["id"]} --replicate 1 --attempt local-{c["id"]}-{args.tag} --development'+(' --smoke' if args.smoke else ''))
    command=f'source /opt/sparsity-gpu/activate.sh; cd {root}; export PYTHONPATH={root}/src; export HF_HUB_OFFLINE=1; export MAX_JOBS=2; set -e; '+'; '.join(commands)
    log=RUN/f'prelaunch/local-{args.tag}.log';log.parent.mkdir(exist_ok=True)
    with log.open('xb') as stream:
        result=subprocess.run(['wsl','-d','SparsityGPU','--exec','bash','-c',command],stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    # Copy failed evidence too. Local logs and GPU result files are part of prelaunch evidence.
    evidence=subprocess.run(['wsl','-d','SparsityGPU','--exec','tar','-C',root,'-cf','-',relative+'/prelaunch',relative+'/artifacts'],cwd=REPO,capture_output=True)
    destination=RUN/'prelaunch'/f'local-{args.tag}'
    destination.mkdir()
    cuda=subprocess.run(['wsl','-d','SparsityGPU','--exec','cat',root+'/'+training+f'/prelaunch/cuda-smoke-{args.tag}.json'],capture_output=True)
    if cuda.returncode==0:(destination/'cuda-smoke.json').write_bytes(cuda.stdout)
    elif result.returncode==0:raise RuntimeError('Reduced CUDA training receipt missing')
    if evidence.stdout:
        with tarfile.open(fileobj=io.BytesIO(evidence.stdout)) as archive:
            for member in archive.getmembers():
                if not member.isfile():continue
                if not member.name.startswith(relative+'/'):raise ValueError('Unexpected local artifact path')
                name=member.name[len(relative)+1:]
                if '..' in name.split('/'):raise ValueError('Unsafe local artifact path')
                path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    write(RUN/f'prelaunch/local-receipt-{args.tag}.json',{'exit_code':result.returncode,'linux_root':root,
          'sources':rows,'log':record(log),'scope':'development GPU qualification; local timing is not accepted RTX5090 evidence'})
    if result.returncode:raise RuntimeError('Local GPU qualification failed; inspect retained log and evidence')
    print({'status':'passed','tag':args.tag,'full_validation':args.full},flush=True)


if __name__=='__main__':main()
