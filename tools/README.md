# Operational Helpers

- `estimate_etc.py` performs transparent ETC and optional cost arithmetic from a
  representative calibration record.
- `watch_run.py` polls a manifest/event stream read-only with bounded sleeps.

These tools do not approve, schedule, launch, retry, or terminate experiments.

## Local GPU environment

The isolated WSL/CUDA development setup and its current readiness are documented
in [local_gpu/README.md](local_gpu/README.md). It does not modify prior runs or
the Windows CPU environment.
