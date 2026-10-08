"""Inventory and archive scientific outputs before any Pod teardown."""
import argparse
import tarfile
from io_utils import RUN, read, record, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if read(RUN/f'artifacts/verification-{args.tag}.json')['status']!='passed':raise ValueError('Verified cohort required')
    paths=sorted(p for parent in (RUN/'artifacts',RUN/'prelaunch') for p in parent.rglob('*') if p.is_file() and p.suffix!='.tmp')
    rows=[record(p,RUN) for p in paths]
    inventory=RUN/f'transfer/inventory-{args.tag}.json';write(inventory,rows)
    archive=RUN/f'transfer/results-{args.tag}.tar.gz'
    with tarfile.open(archive,'w:gz') as handle:
        for row in rows:handle.add(RUN/row['path'],arcname=row['path'],recursive=False)
        handle.add(inventory,arcname=inventory.relative_to(RUN).as_posix(),recursive=False)
    write(RUN/f'transfer/seal-{args.tag}.json',{'archive':record(archive,RUN),'files':len(rows),'total_bytes':sum(r['bytes'] for r in rows)})
    print({'status':'sealed','files':len(rows)},flush=True)


if __name__=='__main__':main()
