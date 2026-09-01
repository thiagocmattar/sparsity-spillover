# Canonical Pythia-410M initialization

This directory records the one locally generated seed-1234 random-pretraining
initialization used by every Run 019 calibration and scientific condition.
Released Pythia weights were not loaded.

`metadata.json` records the constructor, Pythia small-init/Wang-init recipe,
parameter hash, tensor-key hash, tensor count/bytes, generator runtime, and the
strict safetensors round trip. `verification.json` records two additional fresh
CPU constructions and strict loads; both reproduced the same parameter bytes
and the same Python, NumPy, and Torch post-initialization RNG probes.

The 1.621 GB safetensors file and the RNG `.pt` file are intentionally ignored
by Git. Their exact byte counts and SHA-256 values are pinned in `config.yaml`,
`metadata.json`, `verification.json`, and `prelaunch/input-manifest.json`.
Remote setup must possess and verify those exact files before any CUDA model
transfer. Regeneration is not an acceptable substitute for a missing file.
