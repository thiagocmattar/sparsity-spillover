"""Verify all transferred bytes; report incomplete science separately."""
import hashlib
import json
from pathlib import Path
import tarfile
from io_utils import RUN, read, write
from controls import MODES


def main():
    staging = RUN / 'retrieval'
    receipt = read(staging / 'receipt-001.json')
    archive = staging / 'output-001.tar.gz'
    assert archive.stat().st_size == receipt['bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len({m.name for m in members}) == len(members)
        for member in members:
            assert member.isfile() and not member.name.startswith('/')
            assert '..' not in Path(member.name).parts
            assert (RUN / member.name).resolve().is_relative_to(RUN)
        inventory = json.load(tar.extractfile('transfer/inventory-001.json'))
        expected = {r['path']: r for r in inventory['files']}
        assert {m.name for m in members} == set(expected) | {'transfer/inventory-001.json'}
        for member in members:
            content = tar.extractfile(member).read()
            if member.name in expected:
                row = expected[member.name]
                assert len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256']
            path = RUN / member.name
            if path.exists():
                assert path.read_bytes() == content, f'Local source/record mismatch: {member.name}'
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
    completed = []
    for mode in MODES:
        for rep in [1, 2, 3]:
            folder = RUN / 'artifacts/attempts' / f'scientific-{mode}-r{rep}-001'
            if not (folder / 'result.json').exists(): continue
            result = read(folder / 'result.json')
            if result['status'] != 'complete' or not result.get('qualified'): continue
            quality = read(folder / 'quality.json')
            assert not result['arguments']['smoke']
            assert quality['blocks'] == 338 and quality['documents'] == 500
            assert quality['excluded_tail_tokens'] == 1444 and quality['prediction_tokens'] == 691886
            assert result['qualification'] == quality['pass']
            if rep == 1: assert (folder / 'diagnostics.json').is_file()
            completed.append({'mode': mode, 'replicate': rep, 'gpu_uuid': result['runtime']['device_uuid']})
    assert len({r['gpu_uuid'] for r in completed}) <= 1
    report = {'archive': receipt, 'verified_files': len(expected), 'qualified_processes': len(completed),
              'all_transferred_bytes_verified': True, 'complete_scientific_matrix': len(completed) == 30}
    write(RUN / 'artifacts/verification.json', report)
    print(json.dumps(report))


if __name__ == '__main__': main()
