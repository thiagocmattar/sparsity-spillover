"""Full vocabulary GEMM tile (128, 256, 32); sparse h/z unchanged."""
from pathlib import Path
from types import MethodType
from io_utils import RUN,module
HERE=Path(__file__).resolve().parent
def install(model):
    base=module('run042_opt026_base',RUN/'base70/candidate.py')
    metadata=base.install(model)
    head=module('run042_opt026_head',HERE/'head.py')
    linear=model.embed_out
    linear._run042_head=head.Head(linear,1)
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {**metadata,'identity':'opt026','head_tile':(128, 256, 32),
             'head_precision':'BF16 operands/output, FP32 accumulation, full50304 vocabulary',
             'dense_head_change':True,'sparse_sites':['h','z']}
