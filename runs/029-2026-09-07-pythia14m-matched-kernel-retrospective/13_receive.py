"""Verify a retrieved output bundle, then materialize only its inventoried files."""
import argparse
import json
from pathlib import PurePosixPath
import tarfile
from io_utils import RUN, read, write, verify, fs, record


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag.replace('-','').isalnum():raise ValueError('Simple tag required')
    receipt=read(RUN/f'bundles/outputs-{a.tag}.receipt.json')
    bundle=verify(receipt)
    dest=RUN/'artifacts/retrieval'/a.tag;dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(bundle,'r') as tar:
        invname=f'bundles/outputs-{a.tag}-inventory.json'
        inventory=json.load(tar.extractfile(invname))
        expected={r['path']:r for r in inventory['files']}
        if len(expected)!=len(inventory['files']):raise ValueError('Duplicate inventory paths')
        members=tar.getmembers()
        if len({m.name for m in members})!=len(members) or {m.name for m in members}!=set(expected)|{invname}:
            raise ValueError('Archive members differ from inventory')
        for m in members:
            path=PurePosixPath(m.name)
            if (not m.isfile() or path.is_absolute() or '..' in path.parts or '\\' in m.name or ':' in m.name
                or not m.name.startswith(('artifacts/','runtime/','bundles/'))):raise ValueError('Unsafe archive member')
            target=dest/m.name;fs(target.parent).mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,fs(target).open('wb') as output:
                import shutil
                shutil.copyfileobj(source,output)
        for row in inventory['files']:verify(row,dest)
        # Complete leaf data is append-only. Mutable controller snapshots are
        # also retained in the receipt directory before updating the local view.
        for row in inventory['files']:
            target=RUN/row['path'];source=dest/row['path']
            if fs(target).exists() and '/attempts/' in row['path'] and record(target)!=row:
                raise ValueError('Refuse different completed scientific artifact')
            fs(target.parent).mkdir(parents=True,exist_ok=True)
            import shutil
            shutil.copyfile(fs(source),fs(target))
    write(RUN/f'results/retrieval-{a.tag}.json',{'bundle':receipt,'files_verified':len(expected),
          'bytes_verified':sum(r['bytes'] for r in expected.values()),'inventory':record(dest/invname),
          'verification':'Archive SHA256, exact member set, and every local file byte count/SHA256 passed.'})
    print({'tag':a.tag,'verified_files':len(expected),'bytes':inventory['bytes']})


if __name__=='__main__':main()
