"""Freeze the approved sources and input identities; no GPU or infra operations."""
import argparse
import shutil
import subprocess
from io_utils import RUN, REPO, ARCHIVE, read, write, record, verify, fs


def copy_exact(source, dest):
    source, dest = fs(source), fs(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if source.read_bytes() != dest.read_bytes():
            raise ValueError(f'Refuse to overwrite different archived input: {dest}')
    else:
        shutil.copyfile(source, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', action='store_true')
    args = parser.parse_args()
    originals = {n: next((REPO/'runs').glob(f'{n:03d}-*')) for n in range(25,29)}
    paths = set((REPO/'src').rglob('*.py'))
    suffixes = {'.py','.cu','.cuh','.h','.hpp','.cpp','.json','.md','.sh'}
    for n, folder in originals.items():
        paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in suffixes)
        for sub in ['candidates','kernels','upstream','autoresearch/candidates','autoresearch/dense_probe']:
            paths.update(p for p in (folder/sub).rglob('*') if p.is_file() and
                         (p.suffix in suffixes or p.name.startswith('LICENSE')) and '__pycache__' not in p.parts)
        paths.update(p for p in (folder/'autoresearch').glob('*.py'))
    policy = read(originals[28]/'final-policy-002.json')
    for row in policy['sources'] + policy['dependencies']['files']:
        paths.add(verify(row, REPO))
    paths.add(originals[28]/'runtime/vendor/inventory.json')
    vendor = read(originals[28]/'runtime/vendor/inventory.json')
    for row in vendor['files']:
        paths.add(verify(row, REPO))
    # Prove the selected historical candidate source bytes still match execution.
    source_checks = []
    progress = read(originals[28]/'results/search-progress-001.json')
    checked = set()
    for point in progress['points']:
        if point['stage'] != 'development':
            continue
        manifest_row = next(r for r in point['sources'] if r['path'].endswith('/manifest.json'))
        manifest = read(verify(manifest_row, REPO))
        for row in manifest['sources']:
            if '/candidates/' in row['path'] and (row['path'],row['sha256']) not in checked:
                verify(row, REPO)
                checked.add((row['path'],row['sha256']))
                source_checks.append({'attempt':point['attempt'], **row})
    archive = []
    for source in sorted(paths):
        dest = ARCHIVE/source.relative_to(REPO)
        copy_exact(source, dest)
        archive.append({'origin':record(source, REPO), 'snapshot':record(dest)})
    write(RUN/'provenance/archive.json', {'files':archive,
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'historical_candidate_checks':source_checks, 'note':'Byte-identical snapshots; no original file edited.'})
    manifest_path = originals[27]/'prelaunch/inputs.json'
    manifest = read(manifest_path)
    checkpoints = []
    for row in manifest['checkpoints']:
        item = dict(row)
        item['original_checkpoint'] = row['checkpoint']
        item['checkpoint'] = f'inputs/checkpoints/{row["id"]}'
        item['original_files'] = row['files']
        item['files'] = []
        for source_row in row['files']:
            source = verify(source_row, REPO)
            dest = RUN/item['checkpoint']/source.name
            if args.inputs:
                copy_exact(source,dest)
            item['files'].append({**source_row,'path':(dest.relative_to(RUN)).as_posix()})
        # Small provenance records stay with the reproducible package, not external runs.
        provenance = []
        for source_row in row['provenance']:
            source = verify(source_row, REPO)
            dest = RUN/'provenance/originals'/source.relative_to(REPO)
            copy_exact(source,dest)
            provenance.append(record(dest))
        item['provenance'] = provenance
        checkpoints.append(item)
    token_source = verify(manifest['inputs']['validation'], REPO)
    token_dest = RUN/'inputs/validation.int32.bin'
    if args.inputs:
        copy_exact(token_source,token_dest)
    write(RUN/'provenance/inputs.json', {'source_manifest':record(manifest_path,REPO),
          'checkpoints':checkpoints,'validation':{**manifest['inputs']['validation'],'path':'inputs/validation.int32.bin'},
          'validation_metadata':read(verify(manifest['inputs']['validation_metadata'],REPO))})
    print({'archived_files':len(archive),'historical_checks':len(source_checks),'checkpoints':len(checkpoints),
           'checkpoint_bytes':sum(f['bytes'] for c in checkpoints for f in c['files'])})


if __name__ == '__main__':
    main()
