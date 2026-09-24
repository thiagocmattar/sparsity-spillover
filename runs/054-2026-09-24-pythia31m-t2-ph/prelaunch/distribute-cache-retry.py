import importlib.util,pathlib,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
p=pathlib.Path('/workspace/run054-control/distribute-cache.py')
spec=importlib.util.spec_from_file_location('distribute',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m.ssh
m.ssh=lambda node:[part.replace('/workspace/run054-control/transfer-key','/root/.ssh/run054-transfer-key') for part in original(node)]
old=m.pipeline_command
m.pipeline_command=lambda node,index:'set -e; '+old(node,index).replace(' && ','; ')+' echo $! > /workspace/run054-control/pipeline.pid'
with ThreadPoolExecutor(max_workers=3) as pool:
 for future in as_completed([pool.submit(m.transfer_and_start,i) for i in (0,2,3)]):future.result()
pathlib.Path('/root/.ssh/run054-transfer-key').unlink()
m.emit(stage='complete',private_transfer_key_removed=True)
(m.CONTROL/'distribution-ready').touch()
