# Run 017 - Pythia-70M selected ladder with portable initialization

## Status

Stopped at the non-evidence remote preflight on 2026-09-01; no scientific
attempt was created and no training boundary ran. Four approved Secure H200
Pods were provisioned, but the first A7 preflight model realized initialization
SHA-256 `10a47983e13f9a6590bd853abf38690e2e1508a1c7e0715859f7aafa5e7725b4`
under the pinned Linux CUDA wheel instead of the locally pinned CPU-wheel hash
`e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`.
The failure occurred after cache and schedule verification and before the first
boundary. All four Pods were deleted after the failure records were retrieved;
the complete operational record is in `prelaunch/execution-record.json`.

This run carries forward the approved Run 016 scientific design after live H200
and A100 checks proved that Run 016's CUDA-realized initialization hash was
GPU-specific while A40 stock was unavailable. Run 016 remains unchanged and
has no scientific attempts.

## Question and matched design

Does the relationship between measured logical-product opportunity (`R_model`)
and complete-validation loss observed at Pythia-14M persist for the selected
ladder at randomly initialized Pythia-70M?

The twelve conditions, seed 1234, model architecture, one-pass 712-boundary
schedule, 1,493,172,224 training tokens, optimizer, FP16 boundary, activation
topologies, gates, OL1 sites and trust budget, checkpoints, diagnostics, and
A0/A1-H post-hoc TEAL frontiers are identical to Run 016. There is one
implementation delta: the model is constructed and initialized in FP32 on CPU,
the parameter bytes must hash to
`e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`,
and only then is it moved to CUDA before optimizer construction. This makes the
matched initial draw independent of the training GPU. Released weights are
never loaded.

The sentinel remains:

- `a0-gelu`;
- `a1h-relu`;
- `a4-ol1-kappa-0`;
- `a7-ol1-kappa-0`.

The remaining eight conditions are the nonzero-kappa A4/A7 points and still
require a separate post-sentinel decision. This is a ladder promotion, not a
larger-model ablation.

## Validation, diagnostics, and retained state

Step 1 and the reloaded final checkpoint each evaluate all 500 MiniPile
validation documents: all 338 complete 2,048-token blocks and 692,224 input
tokens, with the 1,444-token tail excluded and reported. Counts are pooled as
integers before division.

Final diagnostics retain exact and near-zero activation counts, RMS/L2 and
finite counts per layer/site, all named weight norms, full logical-product
integer counters and `R_block`/`R_model`, and the declared-topology
`R_model_max` counts. Every OL1 boundary stores task/pressure norms, dot product,
cosine, conflict, raw/final ratios, trust scale, clipping, capture identity, and
overflow behavior. The final recovery checkpoint includes model, optimizer,
loss scaler, schedule identity, and Python/NumPy/Torch CPU/all-CUDA RNG states.

After verification, A0 and A1-H run the Analysis 005/006 uniform TEAL protocol
over sites `a,m,h,z`, calibration on the first ten complete source-order
training blocks, target sparsities 0.0 through 0.9, and all 338 validation
blocks per point. The zero-threshold loss must reproduce the source within
`5e-4`.

## Identities and interpretation

- training schedule SHA-256:
  `d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`;
- portable CPU initialization SHA-256:
  `e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`;
- initial implemented config identity:
  `e84b5dc0684d5dbc4d1ebdcdf786478da011263ae6749d1038d7c5794cad96d5`;
- run-code identity:
  `6122e7a66a355ca43fc07cbee6d6b8a9a7cfa5a30a0d9cce4854b0263eaa0d43`.

At one seed, support means that promoted endpoints/frontiers retain a useful
validation-loss versus measured-`R_model` tradeoff. Loss collapse, negligible
measured opportunity, or qualitatively reversed ordering would refute that
expectation. `R_model` and `R_model_max` are logical-product opportunities, not
runtime speedups. The single seed and one larger scale remain interpretation
limits.

## Local verification

The focused Run 017 suite constructs the exact 70,426,624-parameter graph,
checks the CPU initialization hash, verifies CPU-before-CUDA lifecycle wiring,
condition identities, OL1 capture dimensions, schedule/config invariants,
ceilings, and TEAL layer mapping. It currently passes 17/17 tests. The local
Torch 2.11 CPU environment cannot execute the CUDA transfer or representative
training boundary; the launch begins with the non-evidence exact-H200 preflight
before any scientific attempt.

## Proposed RunPod launch

The live 2026-09-01 candidate is four independent Secure H200 Pods at
`$4.59/GPU-hour`, one condition per GPU and no DDP. Each uses the pinned RunPod
PyTorch image, 30 GB container disk, a 25 GB Pod volume at `/workspace`, and an
independent 6.5-hour deletion guard. The hard sentinel maximum is 26 GPU-hours,
`$119.34` compute plus approximately `$0.21` prorated Pod storage (`$119.55`
total), against the last posted `$127.71` balance. Live H200 stock is low, so
price, balance, resources, and per-data-center stock must be refreshed
immediately before creation.

All four Pods are secured first. Repository setup runs concurrently; the
hash-verified 5.97 GB token cache is copied in slowest-condition order
(`a7`, `a4`, `a1`, `a0`). The A7 Pod first runs `05_remote_preflight.py`, which
creates no attempt and must reproduce the CPU hash on the CUDA-enabled Torch
wheel, complete five exact A0 and A7 boundaries, full validation, diagnostics,
checkpoint serialization, and the 10% VRAM-headroom check. Only a passed
preflight permits detached scientific workers to start.

Monitoring is every five minutes and reports step, loss, throughput, and
refreshed ETC. Warnings are stale events over ten minutes, non-finite metrics,
overflow/skipped boundaries, capture mismatch, schedule/hash/runtime drift,
VRAM over 90%, disk risk, or ETC beyond the guard. A0/A1 TEAL runs on their
checkpoint-bearing Pods. Complete attempts, final recovery checkpoints,
diagnostics, logs, pip freeze, cache/code identities, and transfer inventories
are copied and hash-verified locally before deletion. Teardown then confirms
zero unintended Pods; the pre-existing `EUR-IS-1` network volume remains a
separate continuing charge.

See `DEPLOYMENT_PLAYBOOK.md` for the exact control flow.
