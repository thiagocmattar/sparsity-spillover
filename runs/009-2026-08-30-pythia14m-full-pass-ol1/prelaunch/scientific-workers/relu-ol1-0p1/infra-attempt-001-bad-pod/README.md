# Lambda 0.1 infrastructure attempt 001

Pod `wr8s9u2geswh5o` in `US-KS-2` repeatedly stalled while installing the
pinned PyTorch 2.11.0+cu128 environment. The original setup and setup retry 002
each stopped making log progress during pip installation, with the Python
installer waiting on HTTPS sockets in `CLOSE-WAIT`. No dataset cache or
scientific attempt was started on this Pod.

Both setup logs are retained here. The Pod is replaced as an infrastructure
retry without changing source, config, condition, data, or scientific inputs.
