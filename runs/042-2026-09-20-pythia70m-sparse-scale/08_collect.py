"""Inventory complete or failed artifacts before stopping cloud execution."""
import argparse
import json
import tarfile
from io_utils import RUN, record, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='001')
    args = parser.parse_args()
    assert args.tag.isalnum()
    if (RUN / 'artifacts/controller.lock').exists():
        raise RuntimeError('Active controller; do not package changing files')
    target = RUN / 'transfer'
    target.mkdir(exist_ok=True)
    archive = target / f'output-{args.tag}.tar.gz'
    if archive.exists(): raise FileExistsError('Use a new recovery tag')
    paths = []
    for folder in ('artifacts', 'provenance', 'results', 'kernel', 'base70', 'candidates'):
        paths.extend(p for p in (RUN / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths += [p for p in (RUN / 'runtime').glob('*') if p.is_file() and p.suffix in ('.txt', '.log', '.exit')]
    paths += list(RUN.glob('*.py')) + list(RUN.glob('*.sh')) + [RUN / 'config.json']
    paths = sorted(set(paths))
    inventory = target / f'inventory-{args.tag}.json'
    write(inventory, {'files': [record(p) for p in paths]})
    with tarfile.open(archive, 'w:gz') as tar:
        for path in paths + [inventory]:
            tar.add(path, arcname=path.relative_to(RUN).as_posix(), recursive=False)
    receipt = record(archive)
    write(target / f'receipt-{args.tag}.json', receipt)
    print(json.dumps({'archive': receipt, 'files': len(paths)}), flush=True)


if __name__ == '__main__': main()
