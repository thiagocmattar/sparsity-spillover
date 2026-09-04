# Attempt 003 launch adjustment

Attempt 003 retains the pinned code, model checkpoints, workload,
configuration, validation coverage, timing design, and kernel derivative. It
applies `benchmark-inference-context.patch` to the remote extracted benchmark
harness. The patch enters `torch.inference_mode()` around the linear-primitive
and separate attention-composition calls, which consume tensors intentionally
captured under that same mode. It does not change inputs, arithmetic,
repetitions, timing boundaries, tolerances, or reported estimands.

The unchanged worker is launched with the Attempt-002 `PATH`, `CUDA_HOME`, and
`CUDA_PATH` correction. Patch applicability, patched-file compilation, and the
exact remote patch SHA-256 must pass before launch.
