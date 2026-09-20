"""Compile unchanged K050 dependencies before the final checkpoint arrives."""
import json
import time
import replay
from io_utils import RUN, module

started=time.monotonic()
for name in ['k042','k049','k050']:
    candidate=module('run044_precompile_'+name,replay.R28/'candidates'/name/'candidate.py')
    candidate.extension()
    print(json.dumps({'compiled':name,'elapsed_seconds':time.monotonic()-started}),flush=True)
(RUN/'runtime/precompile-ready').touch()
