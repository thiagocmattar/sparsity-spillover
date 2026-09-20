"""Bounded dense-attention scheduling variants, retaining sparse M8/N256 h/z."""
import subprocess
import sys
from io_utils import RUN, write, record


def main():
    origin = RUN/'kernel/attention'
    choices = [('opt007',64,64,4),('opt008',64,128,4),('opt009',128,64,4),
               ('opt010',64,256,4),('opt011',128,128,8)]
    for identifier, m, n, warps in choices:
        folder = RUN/'candidates'/identifier
        attention = folder/'attention'
        attention.mkdir(parents=True,exist_ok=False)
        for name in ('sparse_gemm.h','flash_fwd_kernel.h'):
            text = (origin/name).read_text()
            if name == 'flash_fwd_kernel.h':
                text = text.replace('((bidh*16+m_block)*4+tidx/32)*4',
                    f'((bidh*{2048//m}+m_block)*{warps}+tidx/32)*4')
            (attention/name).write_text(text,newline='\n')
        (attention/'LICENSE-FLASH.md').write_text((origin/'LICENSE-FLASH').read_text(),newline='\n')
        cu = (origin/'kernel.cu').read_text()
        cu = cu.replace('<64,128,128,4,false,false,cutlass::bfloat16_t>',
                        f'<64,{m},{n},{warps},false,false,cutlass::bfloat16_t>')
        cu = cu.replace('dim3(16,1,8),128,smem',f'dim3({2048//m},1,8),{warps*32},smem')
        cu = cu.replace('TORCH_CHECK(!shortcut,', 'TORCH_CHECK(!skip && !count && !shortcut,')
        # This search is for dense attention scheduling. No bypass counters are claimed.
        cu = cu.replace('if(skip){if(count)run035_flash::launch<true,true>(p,stream);else run035_flash::launch<true,false>(p,stream);}\n    else{if(count)run035_flash::launch<false,true>(p,stream);else run035_flash::launch<false,false>(p,stream);}',
                        'run035_flash::launch<false,false>(p,stream);')
        (attention/'kernel.cu').write_text(cu,newline='\n')
        py = (origin/'candidate.py').read_text().split('\ndef install(')[0]
        py = py.replace("name='run035_attention_'+digest",f"name='run042_{identifier}_attention_'+digest")
        py = py.replace("'--expt-extended-lambda'", "'--expt-extended-lambda','--ptxas-options=-v'")
        py = py.replace('class Attention:\n', 'class Attention:\n    native = True  # Dense schedule; no sparse-attention counters.\n')
        py = py.replace('def __init__(self,skip=True,shortcut=True):','def __init__(self,skip=False,shortcut=False):')
        (attention/'candidate.py').write_text(py,newline='\n')
        source = f'''"""Sparse h/z plus dense attention query/key tile {m}/{n}, {warps} warps."""
from pathlib import Path
from io_utils import RUN, module
HERE = Path(__file__).resolve().parent

def install(model):
    base = module('run042_{identifier}_base', RUN/'base70/candidate.py')
    metadata = base.install(model)
    attention = module('run042_{identifier}_attention', HERE/'attention/candidate.py')
    for layer in model.gpt_neox.layers:
        layer.attention._run028_attention = attention.Attention(skip=False,shortcut=False)
    return {{**metadata,'identity':'{identifier}','sparse_sites':['h','z'],
             'attention_activation_skipping':False,'attention_tile':[{m},{n}],
             'attention_warps':{warps},'attention_change':'Dense scheduling; separate from sparsity gain'}}
'''
        (folder/'candidate.py').write_text(source,newline='\n')
        write(folder/'spec.json',{'kind':'dense-attention','M':m,'N':n,'warps':warps,
              'sparse_joint':'Run040 M8 N256, unchanged',
              'origin':record(origin/'kernel.cu'),
              'numerical_note':'Changing key tile can change softmax reduction order; original correctness bounds retained.'})
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__ == '__main__': main()
