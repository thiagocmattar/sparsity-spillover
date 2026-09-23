"""Stage explicit immutable sources and weights; no Windows drive mounting."""
import io
import json
import subprocess
import tarfile
from pathlib import Path
from local_support import RUN, load, sha, write

REPO = RUN.parents[1]
BASE = REPO / 'runs/049-2026-09-22-pythia70m-short-row-limits'
GRID = REPO / 'runs/051-2026-09-22-pythia70m-frozen-kernel-grid'
LINUX = '/home/researcher/sparsity-spillover/' + RUN.name

def main():
    files = {}
    def add(source, target, expected=None):
        source = source.absolute()
        row = {'path': target, 'bytes': source.stat().st_size, 'sha256': sha(source),
               'source': source.relative_to(REPO).as_posix()}
        if expected:
            assert (row['bytes'], row['sha256']) == (expected['bytes'], expected['sha256']), str(source)
        if target in files:
            assert files[target] == row
        files[target] = row
    # The archive is the frozen transitive dependency set from Run049.
    for folder in ('archive', 'candidates', 'base70', 'kernel', 'provenance'):
        for source in (BASE / folder).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts:
                add(source, 'deps/run049/' + source.relative_to(BASE).as_posix())
    for source in [*BASE.glob('*.py'), BASE / 'config.json']:
        add(source, 'deps/run049/' + source.name)
    catalog = load(GRID / 'provenance/inputs.json')
    rows = []
    for cid in load(RUN / 'config.json')['conditions']:
        row = next(r for r in catalog['checkpoints'] if r['id'] == cid)
        origin = BASE if cid != 'c26' else REPO / row['catalog_source']
        for item in row['files']:
            add(origin / item['path'], item['path'], item)
        for item in row['provenance']:
            add(GRID / item['path'], item['path'], item)
        rows.append(row)
    old_inputs = load(BASE / 'provenance/inputs.json')
    for key in ('development', 'validation'):
        item = old_inputs[key]
        add(BASE / item['path'], item['path'], item)
    write(RUN / 'provenance/inputs.json', {'checkpoints': rows,
          'development': old_inputs['development'], 'validation': old_inputs['validation'],
          'validation_metadata': old_inputs['validation_metadata']})
    for source in [*RUN.glob('*.py'), RUN / 'config.json', RUN / 'provenance/inputs.json']:
        add(source, source.relative_to(RUN).as_posix())
    inventory = {'linux_root': LINUX, 'files': list(files.values()),
                 'bytes': sum(r['bytes'] for r in files.values())}
    write(RUN / 'prelaunch/transfer-inventory.json', inventory)
    state = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
             'scoped_status': subprocess.check_output(['git', 'status', '--short', '--', str(RUN)], cwd=REPO, text=True)}
    write(RUN / 'prelaunch/source-state.json', state)
    subprocess.run(['wsl.exe', '-d', 'SparsityGPU', '--exec', 'mkdir', '-p', LINUX], check=True)
    proc = subprocess.Popen(['wsl.exe', '-d', 'SparsityGPU', '--exec', 'tar', '-xf', '-', '-C', LINUX], stdin=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdin, mode='w|') as archive:
            for row in files.values():
                archive.add(REPO / row['source'], arcname=row['path'], recursive=False)
            for name, value in [('input-transfer.json', inventory), ('source-state.json', state)]:
                data = (json.dumps(value, indent=2) + '\n').encode()
                member = tarfile.TarInfo(name); member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    finally:
        proc.stdin.close()
    assert proc.wait() == 0
    verify = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'input-transfer.json').read_text())
for row in manifest['files']:
 p=root/row['path']
 with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
 assert p.stat().st_size==row['bytes'] and digest==row['sha256'],row['path']
print(json.dumps({'verified_files':len(manifest['files']),'bytes':manifest['bytes']}))
'''
    subprocess.run(['wsl.exe', '-d', 'SparsityGPU', '--exec', 'python3', '-', LINUX], input=verify.encode(), check=True)
    print(json.dumps({'linux_root': LINUX, 'staging': 'verified'}))

if __name__ == '__main__':
    main()
