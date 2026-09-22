"""Infrastructure retry: bind intended diagnostics after archived import setup."""
from pathlib import Path
import runpy
import sys

RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN))
import replay
from io_utils import module, record


def bind():
    modules = {}
    for name in ('frozen_diagnostics', 'tile_oracle', 'parallel_work_oracle', 'diagnostics', 'profiling'):
        value = module(name, RUN / (name + '.py'))
        assert Path(value.__file__).resolve() == RUN / (name + '.py')
        modules[name] = record(Path(value.__file__))
    return modules


if __name__ == '__main__':
    import json
    print(json.dumps({'infrastructure_import_bindings': bind()}), flush=True)
    runpy.run_path(str(RUN / '02_benchmark.py'), run_name='__main__')
