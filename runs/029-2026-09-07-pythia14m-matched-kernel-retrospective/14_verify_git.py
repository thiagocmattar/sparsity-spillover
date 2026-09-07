"""Verify the staged archive bytes and exclude data, secrets and live outputs."""
import hashlib
import subprocess
from io_utils import RUN, REPO, fs


def main():
    prefix=RUN.relative_to(REPO).as_posix()
    raw=subprocess.check_output(['git','ls-files','--stage','-z','--',prefix],cwd=REPO)
    count=total=0
    forbidden={'inputs','runtime','bundles','launch-control','artifacts','__pycache__'}
    for entry in raw.split(b'\0'):
        if not entry:continue
        metadata,name=entry.split(b'\t',1);name=name.decode('utf-8')
        path=REPO/name;relative=path.relative_to(RUN)
        # Historical archived source trees legitimately contain runtime/vendor
        # and provenance originals contain historical artifact paths.
        if relative.parts[0] in forbidden or '__pycache__' in relative.parts or path.suffix in {'.safetensors','.bin','.pyc'}:
            raise ValueError(f'Forbidden staged input/live artifact: {relative}')
        contents=fs(path).read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
        if actual!=metadata.split()[1].decode():raise ValueError(f'Staged bytes differ: {relative}')
        count+=1;total+=len(contents)
    print({'staged_run_files_byte_identical':count,'bytes':total})


if __name__=='__main__':main()
