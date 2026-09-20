"""Verify every recovered member before extraction; never overwrite records."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import json
from io_utils import RUN, read, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    assert args.tag.isalnum()
    staging = RUN / 'retrieval'
    receipt = read(staging / f'receipt-{args.tag}.json')
    archive = staging / f'output-{args.tag}.tar.gz'
    assert archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members)
        for member in members:
            assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
            assert (RUN / member.name).resolve().is_relative_to(RUN)
        inventory_name = f'transfer/inventory-{args.tag}.json'
        inventory = json.load(tar.extractfile(inventory_name))
        expected = {r['path']: r for r in inventory['files']}
        assert {m.name for m in members} == set(expected) | {inventory_name}
        for member in members:
            content = tar.extractfile(member).read()
            if member.name in expected:
                row = expected[member.name]
                assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
            target = RUN / member.name
            if target.exists():
                assert target.read_bytes() == content, f'Existing record mismatch: {member.name}'
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    write(RUN / f'artifacts/retrieval-verification-{args.tag}.json',
          {'all_bytes_verified': True, 'archive': receipt, 'files': len(expected)})
    print(f'{len(expected)} transferred files verified', flush=True)


if __name__ == '__main__': main()
