"""Seal the existing trained sources and the new run; do not rewrite Run068."""
import argparse
import tarfile
from common import RUN, REPO, SOURCE, config, read, record, verify, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args()
    config()
    rows=read(SOURCE/'provenance/trained-upload-inventory.json')
    rows += [record(SOURCE/'provenance/trained-upload-inventory.json')]
    rows += [record(p) for p in sorted(RUN.glob('*')) if p.suffix in ('.py','.cu','.sh','.json','.md') and not p.name.startswith('test_')]
    rows=sorted({r['path']:r for r in rows}.values(),key=lambda r:r['path'])
    for row in rows:verify(row)
    write(RUN/'provenance/upload-inventory.json',rows)
    if args.bundle:
        archive=REPO/'tmp/run069-upload.tar.gz'
        archive.parent.mkdir(parents=True,exist_ok=True)
        with tarfile.open(archive,'w:gz') as stream:
            for row in rows:stream.add(REPO/row['path'],arcname=row['path'],recursive=False)
            inventory=RUN/'provenance/upload-inventory.json'
            stream.add(inventory,arcname=inventory.relative_to(REPO).as_posix())
        write(RUN/'prelaunch/bundle.json',{'archive':record(archive),'members':len(rows)+1})
    print({'verified_files':len(rows),'new_training':False},flush=True)


if __name__=='__main__':main()
