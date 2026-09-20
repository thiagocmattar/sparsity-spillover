"""Bundle frozen source and environment only, without a model or timing result."""
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
    for item in read(LATENCY/'provenance/port-origin.json')['sources']:
        paths.append(verify(item['snapshot']))
    paths += [p for p in (LATENCY/'provenance').iterdir() if p.is_file()]
    target=RUN/'bundles/latency-bootstrap-001.tar.gz'
    if target.exists():raise FileExistsError(target)
    with tarfile.open(target,'w:gz',compresslevel=1) as bundle:
        for path in sorted(set(paths)):
            bundle.add(fs(path),arcname=path.relative_to(LATENCY).as_posix(),recursive=False)
    receipt=record(target,RUN)
    write(RUN/'prelaunch/latency-bootstrap-001.receipt.json',receipt)
    print(receipt)


if __name__=='__main__':main()
