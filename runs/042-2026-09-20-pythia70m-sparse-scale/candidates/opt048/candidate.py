"""Dense a/m scheduling; all sparse, attention and head components retained."""
from types import MethodType
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_opt048_parent',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    projection=module('run042_opt048_projection',HERE/'projection.py')
    for layer in model.gpt_neox.layers:
        for linear in (layer.attention.query_key_value,layer.mlp.dense_h_to_4h):
            linear._run042_dense_projection=projection.Projection(linear)
            linear.forward=MethodType(lambda obj,x:obj._run042_dense_projection(x),linear)
    return {**metadata,'identity':'opt048','dense_am_change':True,
            'projection_schedule':(64, 128, 64, 4, 3),'input_projection_skipping':False,
            'input_projections':'Dense Triton BF16 products, FP32 accumulation and bias epilogue; original gates.'}
