"""Precompile unchanged K050 dependencies outside scientific process timing."""
import json
import time
import replay_hz as replay
from io_utils import RUN,module,write

started=time.monotonic();rows=[]
for name in ['k050','k042','k035','k049']:
    candidate=module('run059_precompile_'+name,replay.R28/'candidates'/name/'candidate.py')
    candidate.extension()
    row={'compiled':name,'elapsed_seconds':time.monotonic()-started};rows.append(row)
    write(RUN/'runtime/precompile-status.json',rows);print(json.dumps(row),flush=True)
rope=module('run059_precompile_rope',replay.R26/'autoresearch/candidates/k019/candidate.py')
rope.extension()
rows.append({'compiled':'k019-rope','elapsed_seconds':time.monotonic()-started})
write(RUN/'runtime/precompile-status.json',rows)
(RUN/'runtime/precompile-ready').touch()
