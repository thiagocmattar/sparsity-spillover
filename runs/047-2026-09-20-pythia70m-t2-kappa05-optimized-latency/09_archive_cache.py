"""Cache only immutable vendor sources locally before any measurement starts."""
import shutil
import time
from pathlib import Path
from io_utils import RUN, read, sha, write


def main():
    if (RUN/'artifacts/controller.lock').exists():
        raise RuntimeError('Cache setup must precede the measurement controller')
    source=RUN/'archive/root'
    local=Path('/tmp/run047-archive-root')
    original=source.with_name('root-persistent-original')
    assert not source.is_symlink() and not local.exists() and not original.exists()
    started=time.monotonic()
    shutil.copytree(source,local)
    frozen=read(RUN/'provenance/archive.json')['files']
    for entry in frozen:
        row=entry['snapshot']
        path=local/Path(row['path']).relative_to('archive/root')
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    source.rename(original)
    source.symlink_to(local,target_is_directory=True)
    write(RUN/'provenance/local-archive-cache.json',dict(status='verified',frozen_files=len(frozen),
          seconds=time.monotonic()-started,persistent_original_preserved=True,
          implementation_sources_unchanged=True,reason='Avoid repeated vendor metadata reads; all hash checks remain enabled.'))


if __name__=='__main__':main()
