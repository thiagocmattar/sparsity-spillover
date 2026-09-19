"""Collect stopped/completed outputs, including failures, before Pod teardown."""
import json
import tarfile
from io_utils import RUN, record, write


def main():
    if (RUN / 'artifacts/controller.lock').exists():
        raise RuntimeError('Controller still active: do not package changing artifacts')
    target = RUN / 'transfer'
    target.mkdir(exist_ok=True)
    archive = target / 'output-001.tar.gz'
    if archive.exists(): raise FileExistsError('Never overwrite a retrieval archive')
    paths = []
    for folder in ['artifacts', 'provenance', 'results', 'candidate']:
        paths.extend(p for p in (RUN / folder).rglob('*') if p.is_file())
    paths += [p for p in (RUN / 'runtime').iterdir() if p.is_file() and p.suffix in ['.txt', '.log', '.exit']]
    paths += list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json']
    paths = sorted(set(paths))
    inventory = {'files': [record(p) for p in paths]}
    write(target / 'inventory-001.json', inventory)
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths + [target / 'inventory-001.json']:
            tar.add(path, arcname=path.relative_to(RUN).as_posix(), recursive=False)
    receipt = record(archive)
    write(target / 'receipt-001.json', receipt)
    print(json.dumps({'archive': receipt, 'files': len(paths),
                      'payload_bytes': sum(r['bytes'] for r in inventory['files'])}))


if __name__ == '__main__': main()
