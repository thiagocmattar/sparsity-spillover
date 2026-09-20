"""Parallel sparse-row inspection on the qualified composed parent."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_opt050_base',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    joint=module('run042_opt050_joint',HERE/'joint.py')
    for layer in model.gpt_neox.layers:layer._run026_joint=joint.Joint(layer._run026_joint)
    return {**metadata,'identity':'opt050','parallel_inspection_threads':64,
            'counter_note':'Same M8/N256 conceptual fallback potential; exact issued MMA and scalar counts. Fresh inspection and both launches are timed.'}
