"""Verify every byte in a received scientific archive before approving teardown."""
import argparse
import hashlib
import io
import tarfile
from pathlib import Path
from io_utils import RUN, read, record, verify, write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--seal',type=Path,required=True);parser.add_argument('--tag',required=True);args=parser.parse_args()
    expected=read(args.seal)['archive'];actual=record(args.archive,args.archive.parent)
    if any(actual[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('Archive SHA256/byte mismatch')
    with tarfile.open(args.archive,'r:gz') as archive:
        members=archive.getmembers()
        if any(not m.isfile() or m.name.startswith('/') or '..' in m.name.split('/') for m in members):raise ValueError('Unexpected archive members')
        if len({m.name for m in members})!=len(members):raise ValueError('Duplicate archive members')
        inventory_name=f'transfer/inventory-{args.tag}.json'
        raw=archive.extractfile(inventory_name).read()
        rows=__import__('json').loads(raw)
        if {m.name for m in members}!={r['path'] for r in rows}|{inventory_name}:raise ValueError('Archive membership differs from inventory')
        for row in rows:
            payload=archive.extractfile(row['path']).read()
            if len(payload)!=row['bytes'] or hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Archived member differs')
            destination=RUN/row['path']
            if destination.exists() and destination.read_bytes()!=payload:raise ValueError('Local evidence would be overwritten')
        for row in rows:
            destination=RUN/row['path'];destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.extractfile(row['path']).read())
        destination=RUN/inventory_name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    for row in rows:verify(row,RUN)
    write(RUN/f'transfer/receipt-{args.tag}.json',{'status':'passed','files':len(rows),'archive':actual,'inventory':record(RUN/inventory_name,RUN)})
    print({'status':'passed','verified_files':len(rows)},flush=True)


if __name__=='__main__':main()
