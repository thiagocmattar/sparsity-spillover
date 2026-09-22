"""Use the existing verified SSH identity without changing the Run049 record."""
import importlib.util
from pathlib import Path

BASE=Path(__file__).resolve().parents[2]/'049-2026-09-22-pythia70m-short-row-limits'
spec=importlib.util.spec_from_file_location('run050_existing_transport',BASE/'prelaunch/remote.py')
remote=importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)
connect=remote.connect
execute=remote.execute
