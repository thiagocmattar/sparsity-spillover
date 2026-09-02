# Run 019 H200 scientific launch record

The approved fastest matched-SKU execution uses one `NVIDIA H200` Pod per
condition, preferring Community capacity at $3.59/GPU-hour and falling back to
Secure capacity at $4.59/GPU-hour without changing the GPU SKU. The live H200
calibration passed all required checks and projects:

| Conditions | Count | ETC each | Community cost each | Secure cost each | Guard |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0, A1-H plus TEAL | 2 | 5.861 h | $21.04 | $26.90 | 10.8 h |
| A4-OL1, all `kappa` | 5 | 11.117 h | $39.91 | $51.03 | 20.0 h |
| A7-OL1, all `kappa` | 5 | 12.192 h | $43.78 | $55.96 | 21.9 h |

The twelve-way central projection is 12.192 hours and 128.269 aggregate
GPU-hours: $460.49 if all capacity is Community or $588.76 if all is Secure.
The account balance observed immediately after calibration was $401.53, so
scientific Pods must not be launched until the balance covers the selected
capacity mix plus runtime margin.

Each Pod gets an isolated 60 GB volume, the pinned image digest, an explicit
SSH port, a platform termination deadline where supported, and an independent
local deletion guard. The source bundle and input payload are SHA-verified
before extraction. Worker setup then rechecks all seven scientific inputs and
strict-loads the canonical initialization twice. Training again checks the CPU
parameter hash and its CUDA round trip before optimizer construction.

`build_seed_payload.sh` creates the immutable seven-file transport archive.
`prepare_worker.sh` verifies source and payload identities and performs pinned
environment setup. `verify_worker.sh` is the final fail-closed pre-training
gate. `start_worker.sh` runs exactly one assigned condition, and
`finalize_worker.sh` verifies and packages only its terminal attempt plus local
control records for hash-checked retrieval.
