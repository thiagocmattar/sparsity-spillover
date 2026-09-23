"""Verify every portable input/output member before extraction or teardown."""
import argparse,hashlib,tarfile
from pathlib import Path,PurePosixPath
from support import read,write,sha


def verify(archive,inventory,receipt):
    expected=read(inventory)['files'];receipt=read(receipt);archive=Path(archive)
    assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
    by_name={r['path']:r for r in expected};assert len(by_name)==len(expected)
    seen=set();total=0
    with tarfile.open(archive,'r:gz') as source:
        for member in source:
            path=PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert '\\' not in member.name and ':' not in member.name
            assert member.name in by_name and member.name not in seen
            row=by_name[member.name];assert member.size==row['bytes']
            digest=hashlib.sha256()
            with source.extractfile(member) as stream:
                for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
            assert digest.hexdigest()==row['sha256'],member.name
            seen.add(member.name);total+=member.size
    assert seen==set(by_name),'Missing archive members'
    return {'verified':True,'files':len(seen),'uncompressed_bytes':total,'archive_bytes':archive.stat().st_size,'sha256':receipt['sha256']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--inventory',required=True)
    p.add_argument('--receipt',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    result=verify(a.archive,a.inventory,a.receipt);write(a.report,result);print(result)
