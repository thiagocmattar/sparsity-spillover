"""Build a complete post-stage output archive; no model/input/credential files."""
import argparse,tarfile
from support import RUN,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    if not a.tag.replace('-','').isalnum():raise ValueError('Simple transfer tag required')
    paths=[q for folder in ('artifacts','results','provenance') for q in (RUN/folder).rglob('*') if q.is_file() and '__pycache__' not in q.parts]
    # Training/kernel selections and diagnostics are retained; original model
    # weights already reside locally and are not sent back unnecessarily.
    paths+=list(RUN.glob('*.py'))+list(RUN.glob('*.cu'))+list(RUN.glob('*.sh'))+[RUN/'config.json',RUN/'README.md']
    paths=sorted(set(paths));dest=RUN/'bundles'/f'output-{a.tag}.tar.gz';dest.parent.mkdir(exist_ok=True)
    if dest.exists():raise FileExistsError(dest)
    manifest={'files':[{'path':q.relative_to(RUN).as_posix(),'bytes':q.stat().st_size,'sha256':sha(q)} for q in paths]}
    inventory=dest.with_suffix('.inventory.json');write(inventory,manifest)
    with tarfile.open(dest,'w:gz',compresslevel=1) as archive:
        for q in paths:archive.add(q,arcname=q.relative_to(RUN).as_posix(),recursive=False)
    write(dest.with_suffix('.receipt.json'),{'bytes':dest.stat().st_size,'sha256':sha(dest),'files':len(paths)})
    print({'path':str(dest),'files':len(paths),'bytes':dest.stat().st_size,'sha256':sha(dest)})


if __name__=='__main__':main()
