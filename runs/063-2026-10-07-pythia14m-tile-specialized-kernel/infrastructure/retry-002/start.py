"""Resume the unchanged Run063 pipeline after a pinned-wheel network timeout."""
import importlib.util
from pathlib import Path
import shlex

RUN=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("run063_transport",RUN/"infrastructure/retry-001/transport.py")
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
client,lease=transport.connect()
try:
    remote="""from pathlib import Path
control=Path('/workspace/run063-control')
assert (control/'pipeline.exit').read_text().strip()=='1'
assert Path('/workspace/run063/artifacts/attempts/setup-001/pipeline.log').exists()
for suffix in ['log','exit','pid']:
    source=control/('pipeline.'+suffix)
    destination=control/('pipeline-001.'+suffix)
    assert not destination.exists()
    source.rename(destination)
source=Path('/opt/run063-fast/venv');destination=Path('/opt/run063-fast/failed-venv-001')
assert source.resolve().is_relative_to(Path('/opt/run063-fast'))
assert not destination.exists()
source.rename(destination)
print('Preserved failed setup; prepared fresh application venv with unchanged pins.')
"""
    print(transport.original.execute(client,"python3 -c "+shlex.quote(remote)),flush=True)
    wrapper=(f"cd /workspace/run063; export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5 UV_CONCURRENT_DOWNLOADS=2; "
        f"bash /workspace/run063/04_setup.sh || exit $?; "
        f"bash /workspace/run063/05_execute.sh smoke {lease['deadline_epoch']} 002 || exit $?; "
        f"bash /workspace/run063/05_execute.sh scientific {lease['deadline_epoch']} 002")
    wrapper=f"({wrapper}); code=$?; printf '%s\\n' \"$code\" > /workspace/run063-control/pipeline.exit"
    print(transport.original.execute(client,
        "nohup setsid bash -c "+shlex.quote(wrapper)+
        " > /workspace/run063-control/pipeline.log 2>&1 < /dev/null & echo $! > /workspace/run063-control/pipeline.pid"),flush=True)
    transport.original.write(RUN/"prelaunch/retry-002.json",{
        "status":"started","pod_id":lease["pod"]["id"],"scientific_tag":"002",
        "network_environment":{"UV_HTTP_TIMEOUT":300,"UV_HTTP_RETRIES":5,"UV_CONCURRENT_DOWNLOADS":2},
        "source_script":transport.original.record(__file__),
        "scientific_bundle_unchanged":True,"deadline_epoch":lease["deadline_epoch"]})
finally:client.close()
