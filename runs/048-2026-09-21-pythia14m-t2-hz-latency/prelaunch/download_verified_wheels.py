"""Fetch identical pinned wheels in verified byte ranges from official hosts."""
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request as request

RUN = Path('/workspace/run048')
ROOT = Path('/tmp/run048-wheels')
ROOT.mkdir(exist_ok=True)
rows = json.loads((RUN/'provenance/download-mirror-probes.json').read_text())
started = time.monotonic()
jobs = []
for row in rows:
    # NVIDIA mirrors publish the identical PyPI digest. The Triton mirror differs;
    # retain the original PyPI wheel for that package.
    mirrors = [url for url in row['mirrors'] if url.endswith('#sha256='+row['sha256'])]
    row['selected_url'] = mirrors[0] if mirrors else row['probes'][0]['url']
    folder = ROOT/(row['filename']+'.parts')
    folder.mkdir(exist_ok=True)
    for index, start in enumerate(range(0, row['bytes'], 8*1024**2)):
        end = min(start+8*1024**2, row['bytes'])-1
        jobs.append((row, index, start, end))


def fetch(job):
    row, index, start, end = job
    path = ROOT/(row['filename']+'.parts')/f'{index:04d}'
    with request.urlopen(request.Request(row['selected_url'], headers={'Range':f'bytes={start}-{end}'}), timeout=60) as response:
        assert response.status == 206
        assert response.headers['Content-Range'] == f"bytes {start}-{end}/{row['bytes']}"
        with path.open('wb') as stream:
            while block := response.read(1024**2):
                stream.write(block)
    assert path.stat().st_size == end-start+1
    return path.stat().st_size


done = 0
last = started
with futures.ThreadPoolExecutor(max_workers=12) as pool:
    for result in pool.map(fetch, jobs):
        done += result
        now = time.monotonic()
        if now-last >= 30:
            print({'downloaded_bytes':done,'total_bytes':sum(r['bytes'] for r in rows),
                   'seconds':now-started},flush=True)
            last = now
for row in rows:
    target = ROOT/row['filename']
    digest = hashlib.sha256()
    with target.open('wb') as output:
        for part in sorted((ROOT/(row['filename']+'.parts')).iterdir()):
            with part.open('rb') as stream:
                while block := stream.read(8*1024**2):
                    output.write(block)
                    digest.update(block)
    assert target.stat().st_size == row['bytes'] and digest.hexdigest() == row['sha256']
    print('Verified wheel',row['filename'],row['sha256'],flush=True)
(RUN/'provenance/verified-wheel-downloads.json').write_text(json.dumps({
    'status':'verified','seconds':time.monotonic()-started,'wheels':rows},indent=2)+'\n')
