"""Pin Run061 diagnostics after frozen imports prepend their legacy paths.

Transport/bootstrap retry only: the sealed gate, measurement and diagnostic
sources are unchanged. Fail closed if the intended modules are not selected.
"""
import inspect
import os
from pathlib import Path
import sys
import traceback

try:
    run = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(run))
    from io_utils import module, record, write
    import replay
    kernel = module('kernel', run / 'kernel.py')
    diagnostics = module('diagnostics', run / 'diagnostics.py')
    assert Path(diagnostics.__file__).resolve() == run / 'diagnostics.py'
    assert 'blocks' in inspect.signature(diagnostics.collect).parameters
    assert diagnostics.gated_value is kernel.gated_value
    write(run / 'artifacts/bootstrap-retry-002' / f'{os.getpid()}.json', {
        'status': 'verified', 'argv': sys.argv, 'pid': os.getpid(),
        'source_files': [record(__file__), record(diagnostics.__file__), record(kernel.__file__)],
        'diagnostic_signature': str(inspect.signature(diagnostics.collect)),
        'purpose': 'module resolution only; sealed scientific inputs unchanged',
    })
except BaseException:
    traceback.print_exc()
    os._exit(78)
