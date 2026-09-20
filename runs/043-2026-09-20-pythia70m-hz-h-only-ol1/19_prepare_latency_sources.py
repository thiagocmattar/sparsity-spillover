"""Freeze the qualified Run035 70M kernel with Run041's four-HZ measurement harness.

The only kernel-side adaptation is the existing HZ topology allowlist bridge.
All CUDA, headers, launch settings and 70M counters remain byte-identical.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[1]
R35 = REPO / 'runs/035-2026-09-18-pythia70m-k050-port'
R41 = REPO / 'runs/041-2026-09-20-pythia14m-hz-h-only-ol1/latency'
DEST = RUN / 'latency'


def fs(path):
    return Path('\\\\?\\' + str(path.resolve())) if os.name == 'nt' else path


def digest(path):
    return hashlib.sha256(fs(path).read_bytes()).hexdigest()


def record(path, root):
    return dict(path=path.relative_to(root).as_posix(), bytes=fs(path).stat().st_size,
                sha256=digest(path))


def copy(source, target):
    fs(target.parent).mkdir(parents=True, exist_ok=True)
    if fs(target).exists():
        if digest(source) != digest(target):
            raise ValueError(f'Existing source snapshot differs: {target}')
    else:
        shutil.copyfile(fs(source), fs(target))


def main():
    DEST.mkdir(exist_ok=True)
    origin = json.loads((R35 / 'artifacts/frozen-final-port.json').read_text())
    sources = []
    archive = json.loads((R35 / 'provenance/archive.json').read_text())
    for row in archive['files']:
        relative = row['snapshot']['path']
        source = R35 / relative
        assert record(source, R35) == row['snapshot']
        copy(source, DEST / relative)
    for relative, row in origin['sources'].items():
        source = R35 / relative
        assert record(source, R35) == row
        target = DEST / relative
        if relative == 'kernel/candidate.py':
            original = source.read_text()
            changed = original.replace("replay.R27/'adapter.py'", "replay.RUN/'hz_adapter.py'")
            assert changed != original
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(changed, encoding='utf-8', newline='\n')
        else:
            copy(source, target)
        sources.append({'origin':record(source, REPO), 'snapshot':record(target, DEST)})
    for name in ['01_prepare.py','02_benchmark.py','03_execute.py','04_setup.sh',
                 '05_execute.sh','08_collect.py','09_verify_retrieval.py','io_utils.py',
                 'hz_adapter.py']:
        content = (R41 / name).read_text().replace('Run041','Run043').replace('run041','run043').replace('RUN041','RUN043')
        if name in ['02_benchmark.py','03_execute.py','09_verify_retrieval.py']:
            content = content.replace("'k050'", "'k050-70m-v2'")
        if name == '02_benchmark.py':
            content = content.replace("replay.R28/'115_hybrid_diagnostics.py'", "RUN/'diagnostics.py'")
        if name == '01_prepare.py':
            content = content.replace("    inputs=read(RUN/'provenance/inputs.json')", "    rows += [r['snapshot'] for r in read(RUN/'provenance/port-origin.json')['sources']]\n    inputs=read(RUN/'provenance/inputs.json')")
            content = content.replace("    paths += list((RUN/'provenance')", "    paths += [RUN/r['snapshot']['path'] for r in read(RUN/'provenance/port-origin.json')['sources']]\n    paths += list((RUN/'provenance')")
            content = content.replace("        archive.add(RUN.parent/'run043_topology.py',arcname='run043_topology.py',recursive=False)\n", '')
        if name == '08_collect.py':
            content = content.replace("    paths=sorted(set(paths))", "    paths += [p for p in (HERE/'kernel').rglob('*') if p.is_file()]\n    paths=sorted(set(paths))")
        (DEST / name).write_text(content, encoding='utf-8', newline='\n')
    for name in ['replay.py','diagnostics.py']:
        copy(R35 / name, DEST / name)
    copy(RUN / 'run043_topology.py', DEST / 'run043_topology.py')
    for name in ['archive.json','pip-freeze.txt','candidates.json']:
        copy(R35 / 'provenance' / name, DEST / 'provenance' / name)
    copy(R35 / 'artifacts/frozen-final-port.json', DEST / 'provenance/run035-frozen-final-port.json')
    config = json.loads((R35 / 'config.json').read_text())
    config.update(run_id=RUN.name, budget_usd=2, maximum_gpu_hours=1.5,
        conditions=['c00','c01','c02','c03'],
        authorization='User authorized the matched four-condition 70M promotion, parallel RunPod execution, and final specialized-kernel diagnostics.')
    (DEST / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name in ['hz_adapter.py','replay.py','diagnostics.py','run043_topology.py']:
        source = {'hz_adapter.py':R41/name, 'run043_topology.py':RUN/name}.get(name, R35/name)
        sources.append({'origin':record(source, REPO), 'snapshot':record(DEST/name, DEST)})
    provenance = dict(candidate='k050-70m-v2', sources=sources,
        original_freeze=record(R35/'artifacts/frozen-final-port.json',REPO),
        adaptation='kernel/candidate.py redirects the original adapter to hz_adapter.py; the latter adds only HZ to the topology allowlist. CUDA, launch policy and 70M counters unchanged.')
    (DEST/'provenance/port-origin.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'archived_files':len(archive['files']), 'port_files':len(sources), 'candidate':'k050-70m-v2'}))


if __name__ == '__main__':
    main()
