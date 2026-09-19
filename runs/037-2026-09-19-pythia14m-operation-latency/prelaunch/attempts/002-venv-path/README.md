# Infrastructure retry 002: executable path

The first execution exited before compiling any extension or evaluating model
inputs: PyTorch could not find Ninja. The exact pinned Ninja package was already
installed in the run-local venv, but its bin directory was absent from PATH.
The retry adds that existing directory to PATH in its detached worker. No
scientific code, weights, thresholds, data, flags or numerical bounds change.
The original execution log remains in `runtime/execution-001.log`; this retry
writes separate logs under `artifacts/infrastructure/002-venv-path/`.
