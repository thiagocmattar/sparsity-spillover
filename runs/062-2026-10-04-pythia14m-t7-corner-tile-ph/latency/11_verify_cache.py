"""Verify byte-identical fast vendor/source copies before importing them."""
from io_utils import RUN, ARCHIVE, read, record, write

rows=read(RUN/'provenance/archive.json')['files']
for entry in rows:
    expected=entry['snapshot']
    relative=expected['path'].removeprefix('archive/root/')
    actual=record(ARCHIVE/relative,ARCHIVE)
    if actual['bytes']!=expected['bytes'] or actual['sha256']!=expected['sha256']:
        raise ValueError('Fast source cache mismatch: '+relative)
write(RUN/'runtime/source-cache.json',{'status':'verified','files':len(rows),
      'fast_path':str(ARCHIVE),'persistent_originals':str(RUN/'archive/root'),
      'scientific_bytes_changed':False})
print('Verified fast source cache',len(rows),'files')
