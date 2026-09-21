"""Apply a tighter per-process timeout before execution, retaining its source audit."""
from pathlib import Path
import remote
RUN = Path(__file__).resolve().parent.parent
client = remote.connect()
try:
    print(remote.execute(client, '''python3 - <<'PY'
import hashlib,json,pathlib
r=pathlib.Path('/workspace/run048')
assert not (r/'artifacts/controller.lock').exists() and not (r/'artifacts/attempts').exists()
p=r/'03_execute.py';old=p.read_bytes();text=old.decode()
assert text.count('timeout = 1200')==1
new=text.replace('timeout = 1200',"timeout = 1200 if args.phase == 'smoke' else 600").encode()
(r/'provenance/controller-original.py').write_bytes(old)
p.write_bytes(new)
record={'original_sha256':hashlib.sha256(old).hexdigest(),'revised_sha256':hashlib.sha256(new).hexdigest(),
 'change':'Final fresh processes have a600s rather than1200s timeout; smoke retains1200s. No benchmark, input, timing, validation, or diagnostic setting changes.',
 'reason':'Known full-process durations fit600s; avoid rejecting work merely because a1200s upper bound overlaps the20min retrieval reserve.'}
(r/'provenance/controller-timeout-revision.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY'''))
finally: client.close()
