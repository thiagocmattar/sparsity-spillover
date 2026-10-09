"""Stage the sealed runtime into isolated WSL, run smoke checks, retrieve evidence."""
import argparse
import io
import subprocess
import tarfile
from common import RUN, REPO, read, record, verify, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
    if not args.tag.isalnum():raise ValueError('Simple unique tag required')
    root='/home/researcher/sparsity-spillover/run069-'+args.tag
    relative=RUN.relative_to(REPO).as_posix();rows=read(RUN/'provenance/upload-inventory.json')
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for row in rows:archive.add(verify(row),arcname=row['path'],recursive=False)
        path=RUN/'provenance/upload-inventory.json';archive.add(path,arcname=path.relative_to(REPO).as_posix(),recursive=False)
    base=['wsl.exe','-d','SparsityGPU','--exec']
    subprocess.run(base+['mkdir',root],check=True)
    subprocess.run(base+['tar','-xf','-','-C',root],input=payload.getvalue(),check=True)
    log=RUN/f'prelaunch/local-{args.tag}.txt';log.parent.mkdir(exist_ok=True)
    with log.open('xb') as stream:
        process=subprocess.run(base+['bash',root+'/'+relative+'/09_local.sh',root],stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    exported=subprocess.run(base+['tar','-C',root,'-cf','-',relative+'/prelaunch'],capture_output=True,check=True)
    destination=RUN/'prelaunch'/('local-'+args.tag);destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(exported.stdout)) as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            if not member.name.startswith(relative+'/prelaunch/'):raise ValueError('Unexpected local evidence')
            suffix=member.name[len(relative+'/prelaunch/'):]
            if '..' in suffix.split('/'):raise ValueError('Unsafe evidence path')
            path=destination/suffix;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    receipt={'exit_code':process.returncode,'linux_root':root,'inventory':record(RUN/'provenance/upload-inventory.json'),
             'files':[record(p) for p in sorted(destination.rglob('*')) if p.is_file()],
             'scope':'local5070Ti component/development qualification; not accepted5090 latency evidence'}
    write(RUN/f'prelaunch/local-receipt-{args.tag}.json',receipt)
    print({'exit_code':process.returncode,'evidence':destination.as_posix()},flush=True)
    if process.returncode:raise RuntimeError('Local smoke failed; inspect retrieved evidence')


if __name__=='__main__':main()
