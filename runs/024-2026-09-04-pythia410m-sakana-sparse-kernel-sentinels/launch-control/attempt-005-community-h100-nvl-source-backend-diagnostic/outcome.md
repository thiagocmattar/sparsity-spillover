# Attempt 005 source-backend diagnostic outcome

Run 019's archived SDPA reference for A7-OL1 `kappa=0.5` came from an
NVIDIA A100-SXM4-80GB, whereas Run 024 runs on an H100 NVL. Three alternating
H100 SDPA validations were identical at `5.1206965474687385`; two H100 eager
validations were identical at `5.120631906407825`. Both are within the
unchanged `0.0002` source-identity tolerance of Run 019's canonical eager
endpoint (`5.120691668705122`). The H100 SDPA value is consistently
`0.00020468164477804862` away from the A100 SDPA value.

A second source-only diagnostic forced eager attention for all six sentinels.
All covered all 338 complete blocks and passed against their canonical eager
endpoints; the largest difference was `0.00013801508400934637` for A4-OL1
`kappa=0.5`. These diagnostics contain no sparse-kernel or runtime result.

The evidence supports treating eager validation as the cross-device checkpoint
identity estimand already paired with canonical `R_model`, while retaining
H100 Flash-SDPA for BF16 runtime equivalence and timing. It does not support
widening the tolerance.

