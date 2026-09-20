"""Dense a/m products, bias epilogue and input refresh under original operator bounds."""
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import replay
from io_utils import RUN, module, read, verify, record, write


def main(identifier):
    folder = RUN/'candidates'/identifier
    manifest = read(folder/'manifest.json')
    for row in manifest['files']: verify(row)
    implementation = module('operator_'+identifier, folder/'projection.py')
    result = {'status':'running','candidate':manifest,'checker':record(__file__),'checks':[]}
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.inference_mode():
            for width in (1536, 2048):
                linear = torch.nn.Linear(512,width,device='cuda',dtype=torch.bfloat16)
                op = implementation.Projection(linear)
                for case in ('zero','dense','sparse','changed_input'):
                    x = torch.randn(1,2048,512,device='cuda',dtype=torch.bfloat16)
                    if case == 'zero': x.zero_()
                    if case == 'sparse': x.masked_fill_(x.abs()<.5,0)
                    if case == 'changed_input': x.mul_(2)
                    expected, actual = linear(x), op(x).clone()
                    error = (actual.float()-expected.float()).abs()
                    passed = bool(torch.isfinite(actual).all()) and bool((error<=.015625+.02*expected.float().abs()).all())
                    row = {'case':case,'width':width,'maximum_absolute_error':float(error.max()),
                           'bitwise_equal':bool(torch.equal(actual,expected)),'pass':passed}
                    result['checks'].append(row)
                    print(row,flush=True)
                    assert passed,row
            result['status'] = 'passed'
    except Exception:
        result['status'] = 'failed'
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        write(RUN/'artifacts/development'/f'operator-{identifier}.json',result)


if __name__ == '__main__':
    main(sys.argv[1])
