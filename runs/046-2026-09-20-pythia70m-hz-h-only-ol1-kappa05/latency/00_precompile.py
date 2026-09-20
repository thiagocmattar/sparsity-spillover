"""Compile the unchanged 70M kernel builders before the checkpoint arrives."""
import json
import time
import replay
from io_utils import RUN, module

started=time.monotonic()
sources=[(name,RUN/'kernel'/name/'candidate.py') for name in ['norm','projection']]
sources += [('rope',replay.R26/'autoresearch/candidates/k019/candidate.py')]
sources += [(name,RUN/'kernel'/name/'candidate.py') for name in ['attention','joint']]
for name,path in sources:
    candidate=module('run046_precompile_'+name,path)
    candidate.extension()
    print(json.dumps({'compiled':name,'elapsed_seconds':time.monotonic()-started}),flush=True)
(RUN/'runtime/precompile-ready').touch()
