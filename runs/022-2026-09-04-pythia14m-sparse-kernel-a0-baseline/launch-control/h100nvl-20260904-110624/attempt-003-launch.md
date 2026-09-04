# Attempt 003 launch adjustment

Attempt 002 stopped before timing because `torch.utils.cpp_extension` searches
for the `ninja` executable through `PATH`. The package and executable were
present at `/workspace/run022-state/venv-upstream/bin/ninja`, but invoking the
virtual-environment Python by absolute path did not activate that directory.

Attempt 003 retains the same pinned code, model, workload, and configuration.
It starts the unchanged worker after exporting:

```text
PATH=/workspace/run022-state/venv-upstream/bin:/usr/local/cuda/bin:$PATH
CUDA_HOME=/usr/local/cuda
```

Before launch, the remote host resolved `ninja` to the upstream virtual
environment (version `1.13.2.git.kitware.jobserver-pipe-1`) and `nvcc` to CUDA
12.8 (`V12.8.93`).
