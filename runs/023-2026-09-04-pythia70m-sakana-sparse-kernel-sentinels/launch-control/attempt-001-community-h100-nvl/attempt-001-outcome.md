# Attempt 001 outcome

The disconnect-safe worker started as PID 266 and completed setup, then stopped
before checkpoint benchmarking during the official upstream positive control.
The upstream virtual environment contained `ninja`, but invoking its Python by
absolute path did not add the virtual environment's `bin` directory to `PATH`.
PyTorch's extension loader therefore reported `Ninja is required to load C++
extensions`. The worker recorded exit code 1 and
`status.txt=failed:official_upstream_positive_control` at
2026-09-04T17:16:00Z. No timing result or scientific result was produced.

This reproduces the infrastructure-only PATH condition documented by Run 022.
The failed attempt is retained unchanged under the remote output root.
