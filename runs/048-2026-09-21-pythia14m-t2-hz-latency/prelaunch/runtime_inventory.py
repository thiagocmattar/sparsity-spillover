"""Read-only comparison of base-image packages with the exact runtime pins."""
import remote
client = remote.connect()
try:
    print(remote.execute(client, '''python3 - <<'PY'
import importlib.metadata as m,json,pathlib
pins=dict(line.strip().split('==') for line in pathlib.Path('/workspace/run048/provenance/pip-freeze.txt').read_text().splitlines() if '==' in line)
rows=[]
for name,version in pins.items():
 try:
  d=m.distribution(name)
  if d.version==version:
   rows.append({'name':name,'version':version,'bytes':sum(f.size or 0 for f in d.files or []),'location':str(d.locate_file(''))})
 except m.PackageNotFoundError:pass
print(json.dumps({'exact_matches':sorted(rows,key=lambda r:r['bytes'],reverse=True)},indent=2))
PY
pgrep -af 'uv pip install' ''',timeout=60))
finally: client.close()
