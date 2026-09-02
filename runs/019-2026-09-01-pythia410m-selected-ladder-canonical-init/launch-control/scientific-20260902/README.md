# Run 019 mixed-GPU scientific launch record

The final approved capacity-first execution uses one Pod per condition and tries
GPU SKUs in this order: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `NVIDIA
A100-SXM4-80GB`, `NVIDIA H100 80GB HBM3`, and `NVIDIA H200`. For each SKU it
prefers Community and then Secure capacity. The scientific conditions remain
matched; the realized hardware is recorded as execution provenance and creates
a numerical-reproducibility limitation because kernels may differ by device.

Direct calibration supplies initial ETCs for A100 and H200. RTX PRO 6000 and
H100 start with the conservative A100 projection and are recalibrated from live
complete optimizer boundaries:

| Conditions | Count | Initial A100 ETC each | Measured H200 ETC each |
| --- | ---: | ---: | ---: |
| A0, A1-H plus TEAL | 2 | 9.24 h | 5.861 h |
| A4-OL1, all `kappa` | 5 | 17.54 h | 11.117 h |
| A7-OL1, all `kappa` | 5 | 19.46 h | 12.192 h |

At the approval refresh the balance was $399.99. All-A100 expected compute is
$282.86--$323.56, while all-H200 is $460.49--$588.76. Mixed realized cost is
updated from the assigned SKU, tier, and live ETC. A balance warning is raised
before projected remaining cost consumes the available balance; the human has
stated that additional balance can be added if required.

Each Pod gets an isolated 60 GB volume, the pinned image digest, an explicit
SSH port, a 35-hour platform termination deadline where supported, and an
independent local deletion guard. The source bundle and input payload are SHA-verified
before extraction. Worker setup then rechecks all seven scientific inputs and
strict-loads the canonical initialization twice. Training again checks the CPU
parameter hash and its CUDA round trip before optimizer construction.

`build_seed_payload.sh` creates the immutable seven-file transport archive.
`prepare_worker.sh` verifies source and payload identities and performs pinned
environment setup. `verify_worker.sh` is the final fail-closed pre-training
gate. `start_worker.sh` runs exactly one assigned condition, and
`finalize_worker.sh` verifies and packages only its terminal attempt plus local
control records for hash-checked retrieval.
