# Attempt 007: verification-only recovery

This supersedes failed Recovery 006, whose isolated verifier directory omitted
`config.yaml`. It includes that immutable config, reapplies the same all-zero RMS
verifier correction, verifies the unchanged Attempt 005 benchmark outputs, and
packages them if all fail-closed checks pass. It performs no GPU benchmark work.
