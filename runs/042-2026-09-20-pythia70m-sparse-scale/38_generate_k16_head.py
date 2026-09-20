"""Dense head occupancy variants with one immutable shared CUDA compilation."""
from io_utils import RUN, write, record


def main():
    configs = [(128,128,16,64,64,3), (128,128,16,64,64,4),
               (128,128,16,64,64,2), (64,128,16,32,64,3),
               (128,256,16,64,64,3)]
    source = (RUN/'candidates/opt025/head.cu').read_text()
    source = source.replace('template<int M,int N,int K,int WM,int WN>',
                            'template<int M,int N,int K,int WM,int WN,int Stages>')
    source = source.replace('GemmIdentityThreadblockSwizzle<>,3,8,8,false',
                            'GemmIdentityThreadblockSwizzle<>,Stages,8,8,false')
    start = source.index('  case 0:')
    end = source.index('  default:', start)
    cases = ''.join(f'  case {i}:launch<{",".join(map(str,c))}>(x,w,out,stream);break;\n' for i,c in enumerate(configs))
    source = source[:start] + cases + source[end:]
    for index, config in enumerate(configs):
        identifier = f'opt{65+index:03d}'
        folder = RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        if index == 0:
            (folder/'head.cu').write_text(source, newline='\n')
        head = (RUN/'candidates/opt025/head.py').read_text()
        head = head.replace("source=Path(__file__).with_name('head.cu')", "from io_utils import read,verify\n    source=verify(read(Path(__file__).with_name('spec.json'))['shared_head_cuda'])")
        (folder/'head.py').write_text(head.rstrip()+'\n', newline='\n')
        code = f'''"""Dense head scheduling only on the qualified composed parent."""
from types import MethodType
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    verify(spec['shared_head_cuda'])
    base=module('run042_{identifier}_parent',RUN/'candidates/opt063/candidate.py')
    metadata=base.install(model)
    head=module('run042_{identifier}_head',HERE/'head.py')
    linear=model.embed_out
    linear._run042_head=head.Head(linear,{index})
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {{**metadata,'identity':'{identifier}','head_schedule':{config},'dense_head_change':True}}
'''
        (folder/'candidate.py').write_text(code, newline='\n')
        write(folder/'spec.json', {'kind':'dense-head-cutlass-occupancy','index':index,
              'schedule':config,'parent_manifest':record(RUN/'candidates/opt063/manifest.json'),
              'shared_head_cuda':record(RUN/'candidates/opt065/head.cu'),
              'hypothesis':'K16 pipeline tiles may improve dense-head occupancy by lowering shared-memory use; unsupported compilation is recorded as failure.',
              'constraints':'BF16 operands/output, FP32 accumulation, unchanged full50304 logits and sparse components.'})
        write(folder/'manifest.json', {'candidate':identifier,
              'files':[record(p) for p in sorted(folder.glob('*'))],
              'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
