"""Dense full-logit scheduling only; exact parent sparse components."""
from types import MethodType
from pathlib import Path
from io_utils import RUN, module, read, verify
HERE = Path(__file__).resolve().parent
def install(model):
    spec = read(HERE/'spec.json')
    parent = read(verify(spec['parent_manifest']))
    for row in parent['files']: verify(row)
    base = module('run042_opt033_parent', RUN/'candidates/opt032/candidate.py')
    metadata = base.install(model)
    head = module('run042_opt033_head', HERE/'head.py')
    linear = model.embed_out
    linear._run042_head = head.Head(linear, 0)
    linear.forward = MethodType(lambda obj, x: obj._run042_head(x), linear)
    return {**metadata, 'identity': 'opt033', 'triton_version': head.triton.__version__,
            'head_schedule': (32, 128, 64, 4, 3), 'dense_head_change': True}
