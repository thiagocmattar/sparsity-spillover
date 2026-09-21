"""Probe official mirrors for the four remaining exact wheels; no environment mutation."""
import remote
client = remote.connect()
try:
    print(remote.execute(client, '''python3 - <<'PY'
import concurrent.futures as f,html,json,pathlib,re,time,urllib.parse as p,urllib.request as u
pins=dict(line.strip().split('==') for line in pathlib.Path('/workspace/run048/provenance/pip-freeze.txt').read_text().splitlines() if '==' in line)
def probe(name):
 version=pins[name]
 data=json.load(u.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=15))
 files=[d for d in data['urls'] if 'x86_64' in d['filename'] and 'linux' in d['filename'] and ('-py3-' in d['filename'] or '-cp312-' in d['filename'])]
 assert len(files)==1,(name,[d['filename'] for d in files])
 wheel=files[0]
 page=u.urlopen(f'https://download.pytorch.org/whl/cu128/{name}/',timeout=15).read().decode()
 links=[html.unescape(x) for x in re.findall(r'href="([^"]+)"',page)]
 mirrors=[p.urljoin(f'https://download.pytorch.org/whl/cu128/{name}/',link) for link in links if p.unquote(p.urlsplit(link).path).endswith('/'+wheel['filename'])]
 row={'name':name,'filename':wheel['filename'],'sha256':wheel['digests']['sha256'],'bytes':wheel['size'],'mirrors':mirrors,'probes':[]}
 for url in [wheel['url']]+mirrors[:1]:
  start=time.monotonic()
  try:
   with u.urlopen(u.Request(url,headers={'Range':'bytes=0-262143'}),timeout=12) as stream:
    block=stream.read(262144)
   row['probes'].append({'url':url,'bytes':len(block),'seconds':time.monotonic()-start})
  except Exception as e:row['probes'].append({'url':url,'error':str(e),'seconds':time.monotonic()-start})
 return row
with f.ThreadPoolExecutor(max_workers=4) as pool:
 rows=list(pool.map(probe,['nvidia-cudnn-cu12','nvidia-nccl-cu12','nvidia-nvshmem-cu12','triton']))
pathlib.Path('/workspace/run048/provenance/download-mirror-probes.json').write_text(json.dumps(rows,indent=2)+'\\n')
print(json.dumps(rows,indent=2))
PY''',timeout=60))
finally: client.close()
