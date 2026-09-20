"""Full vocabulary GEMM tile (256, 128, 64); sparse h/z unchanged."""
from pathlib import Path
from types import MethodType
from io_utils import RUN,module
HERE=Path(__file__).resolve().parent
def install(model):
    base=module('run042_opt027_base',RUN/'base70/candidate.py')
    metadata=base.install(model)
    head=module('run042_opt027_head',HERE/'head.py')
    linear=model.embed_out
    linear._run042_head=head.Head(linear,2)
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {**metadata,'identity':'opt027','head_tile':(256, 128, 64),
             'head_precision':'BF16 operands/output, FP32 accumulation, full50304 vocabulary',
             'dense_head_change':True,'sparse_sites':['h','z']}
