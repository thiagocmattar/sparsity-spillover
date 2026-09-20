"""Dense head scheduling only on the qualified composed parent."""
from types import MethodType
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    verify(spec['shared_head_cuda'])
    base=module('run042_opt056_parent',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    head=module('run042_opt056_head',HERE/'head.py')
    linear=model.embed_out
    linear._run042_head=head.Head(linear,3)
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {**metadata,'identity':'opt056','head_schedule':(64, 64, 32, 32, 32, 3),'dense_head_change':True}
