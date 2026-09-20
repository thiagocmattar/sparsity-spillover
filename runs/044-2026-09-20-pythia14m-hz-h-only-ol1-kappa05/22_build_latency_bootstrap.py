"""Bundle frozen source and environment only; no model or scientific result."""
import importlib.util
from pathlib import Path
import sys
import tarfile

RUN=Path(__file__).resolve().parent
LATENCY=RUN/'latency'
sys.path.insert(0,str(LATENCY))
from io_utils import fs,read,record,verify,write

def main():
    paths=list(LATENCY.glob('*.py'))+list(LATENCY.glob('*.sh'))+[LATENCY/'config.json']
    for item in read(LATENCY/'provenance/archive.json')['files']:
        paths.append(verify(item['snapshot']))
    paths += [LATENCY/'provenance'/name for name in ['archive.json','candidates.json','pip-freeze.txt']]
    paths=sorted(set(paths))
    target=RUN/'bundles/latency-bootstrap-001.tar.gz'
    if target.exists():raise FileExistsError(target)
    with tarfile.open(target,'w:gz',compresslevel=1) as bundle:
        for path in paths:bundle.add(fs(path),arcname=path.relative_to(LATENCY).as_posix(),recursive=False)
    receipt=record(target,RUN)
    write(RUN/'prelaunch/latency-bootstrap-001.receipt.json',receipt)
    print(receipt)

if __name__=='__main__':main()
