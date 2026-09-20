"""Preserve qualified attention arithmetic while writing the next layer's layout."""
import shutil
from io_utils import RUN, write, record


def main():
    folder = RUN/'candidates/opt064'
    shutil.copytree(RUN/'candidates/opt024/attention', folder/'attention')
    py = (folder/'attention/candidate.py').read_text().replace('run042_opt024_', 'run042_opt064_')
    assert 'self.out=torch.empty_like(q)' in py
    py = py.replace('self.out=torch.empty_like(q)',
                    'self.out=torch.empty((1,2048,8,64),device=q.device,dtype=q.dtype).transpose(1,2)')
    (folder/'attention/candidate.py').write_text(py.rstrip()+'\n', newline='\n')
    cu = (folder/'attention/kernel.cu').read_text()
    cu = cu.replace('for(const auto& x:{q,k,v,out})', 'for(const auto& x:{q,k,v})')
    anchor = '    for(const auto& x:{lse,lse_accum,o_accum})'
    assert anchor in cu
    cu = cu.replace(anchor, '    TORCH_CHECK(out.device()==q.device() && out.scalar_type()==q.scalar_type() && out.strides()==at::IntArrayRef({1048576,64,512,1}),"Token-major output required");\n'+anchor)
    cu = cu.replace('p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=64;',
                    'p.q_row_stride=p.k_row_stride=p.v_row_stride=64;p.o_row_stride=512;')
    cu = cu.replace('p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=2048*64;',
                    'p.q_head_stride=p.k_head_stride=p.v_head_stride=2048*64;p.o_head_stride=64;')
    (folder/'attention/kernel.cu').write_text(cu.rstrip()+'\n', newline='\n')
    code = '''"""Qualified parallel sparse h/z plus token-major dense attention output."""
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent

def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_opt064_parent',RUN/'candidates/opt063/candidate.py')
    metadata=base.install(model)
    attention=module('run042_opt064_attention',HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention=attention.Attention(skip=False,shortcut=False)
    return {**metadata,'identity':'opt064','attention_output_layout':'T,H,D',
            'attention_activation_skipping':False,'attention_change':'Output layout only; unchanged dense attention arithmetic.'}
'''
    (folder/'candidate.py').write_text(code,newline='\n')
    write(folder/'spec.json', {'kind':'dense-attention-token-major-output',
          'parent_manifest':record(RUN/'candidates/opt063/manifest.json'),
          'source_attention':record(RUN/'candidates/opt024/manifest.json'),
          'hypothesis':'Avoid six attention-output transpose copies without changing arithmetic, operands, masks, scaling or thresholds.',
          'qualification':'Synthetic native comparison and causal checks, then original full-model bounds on fixed development inputs only.'})
    write(folder/'manifest.json', {'candidate':'opt064',
          'files':[record(p) for p in sorted(folder.rglob('*')) if p.is_file()],
          'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
