"""Test two native dense SDPA backends with the qualified sparse components."""
from io_utils import RUN, write, record


def main():
    for identifier, backend in [('opt061','CUDNN_ATTENTION'), ('opt062','EFFICIENT_ATTENTION')]:
        folder = RUN/'candidates'/identifier
        (folder/'attention').mkdir(parents=True, exist_ok=False)
        attention = f'''import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

class Attention:
    native=True
    def __init__(self,skip=False,shortcut=False):
        assert not skip and not shortcut
    def __call__(self,q,k,v,scale):
        assert not torch.is_grad_enabled() and q.dtype==torch.bfloat16
        with sdpa_kernel(SDPBackend.{backend}):
            return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
'''
        (folder/'attention/candidate.py').write_text(attention,newline='\n')
        candidate = f'''"""Native dense {backend} attention; unchanged gates and sparse h/z."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_{identifier}_parent',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    attention=module('run042_{identifier}_attention',HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=attention.Attention()
    return {{**metadata,'identity':'{identifier}','attention_backend':'{backend}',
            'attention_activation_skipping':False,'attention':'Native dense SDPA backend; no changed operands, gates, mask or scaling.'}}
'''
        (folder/'candidate.py').write_text(candidate,newline='\n')
        write(folder/'spec.json', {'kind':'native-dense-attention-backend','backend':backend,
              'parent_manifest':record(RUN/'candidates/opt032/manifest.json'),
              'hypothesis':'Attention contributes 0.324ms; compare available native scheduling implementations without changing the model or attributing dense gains to sparsity.',
              'qualification':'Causal/synthetic operator checks and original native-eager full-model bounds; no final input exposure during selection.'})
        write(folder/'manifest.json', {'candidate':identifier,
              'files':[record(p) for p in sorted(folder.rglob('*')) if p.is_file()],
              'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
