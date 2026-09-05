# Attempt 001 outcome

The detached worker stopped during `official_upstream_positive_control`, before
any Run-024 Pythia-410M measurement. Both pinned environments and both static
preflights had completed successfully.

The upstream virtual environment contained `ninja==1.13.2`, but the worker
invoked its Python executable by absolute path without placing the virtual
environment's `bin` directory on `PATH`. PyTorch's JIT extension loader uses an
executable search for `ninja`, so the official positive control failed closed
with `RuntimeError: Ninja is required to load C++ extensions`.

This is an execution-environment path defect, not a scientific-input or kernel
correctness failure. Attempt 002 reuses the unchanged sealed repository and
checkpoints, appending `/workspace/run024-state/venv-upstream/bin` to `PATH`
for the worker process. Attempt 001's partial artifacts remain on the Pod.
