"""Compile the actual fixed dependency chain before model graph capture."""
import time
import replay
from io_utils import RUN,module,write

started=time.monotonic();rows=[]
for name in ['k050','k042','k035']:
    candidate=module('run061_precompile_'+name,replay.R28/'candidates'/name/'candidate.py')
    candidate.extension()
    rows.append({'compiled':name,'elapsed_seconds':time.monotonic()-started})
    write(RUN/'runtime/precompile-status.json',rows)
    print(rows[-1],flush=True)
rope=module('run061_precompile_rope',replay.R26/'autoresearch/candidates/k019/candidate.py')
rope.extension()
from kernel import extension
extension()
rows.append({'compiled':'k019-rope and run061-joint','elapsed_seconds':time.monotonic()-started})
write(RUN/'runtime/precompile-status.json',rows)
(RUN/'runtime/precompile-ready').touch()
