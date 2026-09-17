"""Verify a downloaded archive and each member before accepting Pod teardown."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent


def main():
    staging=HERE/'retrieval'
    receipt=json.loads((staging/'receipt-001.json').read_text())
    archive=staging/'output-001.tar.gz'
    assert archive.stat().st_size==receipt['bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        for member in members:
            assert member.isfile() and not member.name.startswith('/')
            assert '..' not in Path(member.name).parts
            assert (HERE/member.name).resolve().is_relative_to(HERE)
        inventory=json.load(bundle.extractfile('transfer/inventory-001.json'))
        expected={r['path']:r for r in inventory['files']}
        assert {m.name for m in members}==set(expected)|{'transfer/inventory-001.json'}
        for member in members:
            content=bundle.extractfile(member).read()
            if member.name in expected:
                row=expected[member.name]
                assert len(content)==row['bytes']
                assert hashlib.sha256(content).hexdigest()==row['sha256']
            path=HERE/member.name
            if path.exists():assert path.read_bytes()==content, f'Local source/record mismatch: {member.name}'
            else:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    scientific=[]
    for condition in [f'c{i}' for i in range(36,41)]:
        for replicate in range(1,4):
            folder=HERE/'artifacts/attempts'/f'scientific-{condition}-r{replicate}-001'
            result=json.loads((folder/'result.json').read_text())
            quality=json.loads((folder/'quality.json').read_text())
            assert result['status']=='complete' and not result['arguments']['smoke']
            assert result['candidate']=='k050' and result['condition']==condition
            assert quality['blocks']==338 and quality['documents']==500
            assert quality['excluded_tail_tokens']==1444 and quality['prediction_tokens']==691886
            assert result['qualification']==quality['pass']
            if replicate==1:assert (folder/'diagnostics.json').is_file()
            scientific.append({'attempt':folder.name,'qualified':result['qualified'],'loss':quality['loss'],
                               'gpu_uuid':result['runtime']['device_uuid']})
    assert len({r['gpu_uuid'] for r in scientific})==1
    report={'archive':receipt,'verified_files':len(expected),'scientific_processes':scientific,
            'qualified_processes':sum(r['qualified'] for r in scientific),
            'all_required_artifacts_verified':True}
    (HERE/'artifacts/verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified_files':len(expected),'processes':len(scientific),
                      'qualified':report['qualified_processes'],'archive_sha256':receipt['sha256']}))


if __name__=='__main__':main()
