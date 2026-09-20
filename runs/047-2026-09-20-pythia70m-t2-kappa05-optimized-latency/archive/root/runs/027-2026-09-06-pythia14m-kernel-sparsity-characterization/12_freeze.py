"""Freeze the evaluated scientific implementation once, before the matrix."""
from datetime import datetime,timezone
from run027_common import RUN,ROOT,R25,read_json,write_json,record

target=RUN/'prelaunch/frozen.json'
if target.exists():raise ValueError('Do not overwrite a frozen execution record')
paths=[RUN/name for name in ['01_benchmark.py','03_matrix.py','adapter.py','kernel.cu','diagnostics.py','run027_common.py','config.json']]
paths += list((ROOT/'src').rglob('*.py'))
paths += [R25/name for name in ['autoresearch/dense_probe/probe.py','measurement.py','run025_common.py','config.json']]
records=[record(p) for p in paths]+read_json(RUN/'prelaunch/inputs.json')['kernel_sources']
write_json(target,{'utc':datetime.now(timezone.utc).isoformat(),
    'files':records,'bytes':sum(r['bytes'] for r in records),
    'scope':'All model, kernel, validation and timing implementation; no runtime tuning or selection.'})
print(record(target))
