# 001 — Local Pythia-14M learning-rate pipeline calibration

**State:** completed

## Request and detailed interpretation

Run four independently initialized-from-config Pythia-14M pretraining
conditions locally for a combined target of approximately 30 minutes. Vary only
the peak learning rate, reduce the reference global batch from 128 to 64
sequences, and use the exercise to validate the new repository's end-to-end
training, validation, diagnostics, checkpoint, and artifact pipeline.

The user approved the independent factor-two grid centered on the original
Pythia-14M recipe: `5e-4`, `1e-3`, `2e-3`, and `4e-3`.

## Goal / hypothesis

- **Question:** Can the bootstrap execute a matched four-condition random
  Pythia-14M pretraining cohort on the local GPU with complete validation and
  internally consistent durable artifacts?
- **Expect:** finite training/validation losses and gradients, decreasing loss
  over the short horizon, correct schedules, matched input identities,
  reloadable checkpoints, and stable local throughput with VRAM headroom.
- **Supports if:** all four conditions complete with identical initialization
  and data-order identities, exact validation coverage, finite metrics, and
  reconciled manifests/checkpoints/diagnostics.
- **Refutes if:** any condition OOMs, becomes nonfinite, consumes a different
  schedule, misses coverage, fails checkpoint reload, or publishes inconsistent
  artifacts.

This is an operational shakedown with a descriptive short-horizon LR ranking.
It is not a horizon-independent learning-rate selection.

## Manuscript relationship

- **Affected goal:** prerequisite reliability for later randomly initialized
  Pythia-14M comparisons.
- **Empirical claim tested:** none of the sparsity-spillover or logical-compute
  claims; this run uses the unsparsified control topology.
- **Operational definition:** random pretraining under the pinned architecture,
  full MiniPile validation contract, count-first diagnostics, and immutable
  attempt artifacts.
- **New/material difference from the draft:** none.
- **Possible consequence:** a pipeline-validity observation and a provisional
  LR ranking only; no TeX or manuscript claim follows automatically.

## Interpretation choices

- The LR grid is independent and centered on the original Pythia-14M `1e-3`
  recipe rather than on historical results from the source repository.
- The 30-minute request is interpreted as driver start through all four final
  artifacts. Implementation and environment preparation occur beforehand.
- All four conditions receive one identical fixed optimizer-step count. A
  wall-time stop is not used because it could give conditions unequal budgets.
- The exact common step count is locked after representative local timing and
  returned for separate launch approval.
- Conditions run serially in ascending LR order on one local GPU.

## Design

### Varies

Peak learning rate only: `5e-4`, `1e-3`, `2e-3`, `4e-3`.

### Matched controls

Architecture/config revision, random initialization seed and resulting initial
parameter hash, complete training-block order, batch, token budget, optimizer,
schedule shape, precision, validation coverage, diagnostics, and checkpoints.

### Model and initialization

- `EleutherAI/pythia-14m-deduped` architecture revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`.
- Construct with `AutoModelForCausalLM.from_config`; do not load released
  weights.
- Model seed `0`; FP32 parameters and AdamW state; BF16 CUDA autocast.

### Data, order, batch, budget, optimizer, and schedule

- MiniPile revision `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0`.
- Tokenizer revision `7386d9a4ae45aef494a6e704910394def3037fc5`;
  no added special tokens and append EOS per document.
- Sequence length 2,048. Use the verified full training token cache and one
  seed-0 permutation of complete blocks.
- Global batch 64 sequences = 131,072 input tokens/update, realized as
  microbatch 4 × accumulation 16.
- One common step count for all conditions: `449`, locked from the
  representative local calibration so the p90 cohort ETC is below 27 minutes,
  leaving approximately three minutes of headroom inside the requested
  30-minute target. Each condition consumes 58,851,328 training input tokens;
  the four-condition cohort consumes 235,405,312.
- Warmup is 5 updates. The realized seed-0 schedule contains 28,736 distinct
  complete blocks with no wrap and has SHA-256
  `d61d355668223d092d2d0f1b04daf9c614c45d6bffe2670ab4a6c63b1ae47523`.
- AdamW `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, global gradient clip
  `1.0`; 1% linear warmup and cosine decay to 10% of peak LR.

### Full validation

Every named pass covers all 500 validation documents in deterministic source
order: 338 complete sequences, 692,224 input tokens, and the 1,444-token
excluded tail. Evaluate after update 1 and from the reloaded final checkpoint;
validation batch size is 4.

### Topology, gate, and pressure

- Topology: `A0`.
- Gate: none; stock GELU at `h`.
- Pressure method/sites/weight: `none` / empty / `0.0`.

### Seeds and uncertainty

Model seed `0` and data-order seed `0` for every LR. With `n=1`, the run has no
seed uncertainty estimate. Sequential execution can also retain residual
thermal/order effects.

### Diagnostics and post-hoc retention

- Every optimizer boundary: step, tokens, elapsed/step time, effective LR,
  task loss, task-gradient norm before/after clipping, clipping flag,
  throughput, and peak allocated/reserved GPU memory.
- Final complete validation: exact/near-zero counts at `0`, `1e-3`, and `1e-2`
  plus activation RMS/L2 statistics for `a`, `m`, and `h` by layer and pooled.
- Final checkpoint: weight statistics by named parameter and role.
- Retain all four final model checkpoints and hashes; omit optimizer state.
- Gradient conflict is inapplicable without pressure. Logical-product counters
  and clipping sweeps are deliberately omitted; the retained checkpoints and
  exact cache identity permit later evaluation-only diagnostics.

### Interpretation limits

- The preferred LR can depend on batch and horizon; this short run cannot set a
  manuscript-wide LR.
- This does not exercise `l1_naive`, `orthogonal_l1`, gated topologies, logical
  counters, or post-hoc clipping end to end.
- Logical opportunity and measured runtime speedup are not outcomes here.

## Approval record

- Design approved by user: 2026-08-28; four independent recipe-centered LRs,
  matched random Pythia-14M conditions, reduced local batch, complete
  validation, and an approximately 30-minute serial cohort.
- Launch approved by user: 2026-08-28; exact local 449-step/four-condition
  envelope in `prelaunch/launch-plan.json`, with the measurements and retention
  policy above.

## Implementation

- `config.yaml` is the resolved four-condition scientific and operational
  definition.
- `lr_run_config.py` owns config validation, cache identities, the matched data
  schedule, batch materialization, and content hashes.
- `lr_training.py` executes AdamW boundaries, complete validation, final
  activation/weight diagnostics, checkpoint reload, and immutable attempts.
- `lr_calibration.py` owns the production-shaped timing sample and transparent
  common-step/ETC arithmetic. `pipeline.py` is only their small public facade.
- `01_calibrate.py` runs the production-shaped non-evidence local timing sample
  and writes its measurements under `prelaunch/`.
- `02_train.py` is the launch entry point. It runs the four LRs serially in
  ascending order and creates one immutable attempt per condition.
- The exact prior cache bytes are exposed in this checkout by NTFS hard links;
  no 5.97 GB duplicate was created. The run verifies SHA-256 before training.

## Verification

- Run-focused tests: `6 passed`; they cover the approved grid/batch/
  validation lock, one-field condition resolution, common p90 step budgeting,
  and explicit block-to-microbatch materialization.
- Full bootstrap suite immediately before launch: `54 passed` in 3.56 s.
- Python compilation: all six executed run-local Python files compiled
  successfully.
- Production-shaped GPU calibration, repeated twice: two startup boundaries
  discarded and eight full optimizer boundaries measured per sample; two
  ordinary and two diagnostic complete-validation passes per sample; all losses
  finite and identical across repeated no-update validation within each sample;
  18 layer-level and three pooled activation diagnostic rows; 76 weight rows;
  checkpoint save, content hash, reload, and CUDA placement succeeded.
- The tests use the repository `.venv`; it intentionally remains CPU-only. The
  GPU calibration and proposed launch use the system Python 3.12 environment,
  whose pinned package versions match `pyproject.toml` and whose Torch build is
  `2.11.0+cu128`.

## ETC and execution plan

- **Calibration evidence:**
  two repeated samples under `prelaunch/`. The conservative repeat is
  `calibration-20260828-190736.json`: cache verification 3.21 s, eight
  timed boundaries after two discarded startup boundaries 0.854--0.883 s,
  ordinary full validation 1.64 s,
  diagnostic full validation 2.99--3.03 s, checkpoint save/hash/reload 0.320 s,
  and per-condition model/optimizer setup 0.569 s.
- **Local fit:** RTX 5070 Ti Laptop GPU, 12,227 MiB reported by `nvidia-smi`
  (12,820,480,000 bytes via Torch). Peak allocated/reserved memory was
  6,263,572,992 / 7,769,948,160 bytes. With the 2,974 MiB preflight WDDM/
  desktop usage conservatively added to the measured process reservation,
  approximately 1.93 GB (1.80 GiB) remains; this exceeds the 1 GiB warning
  floor but assumes no other GPU workload starts.
- **ETC:** 449 common updates; median 1,582.97 s (26m23.0s), p90 1,617.14 s
  (26m57.1s), eight complete validation passes, and about three minutes of
  headroom to the 30-minute target.
- **Execution definition:** exact resolved values are also recorded in
  `prelaunch/launch-plan.json`.
- **Proposed location:** local single GPU, serial conditions.
- **Cloud cost:** none.
- **Storage:** expected four final model checkpoints of approximately 54 MiB
  each plus small JSON artifacts; no transfer is required.
- **Monitoring:** inspect every-step events at approximately 60-second
  intervals; warn on nonfinite metrics, OOM, cache/hash mismatch, missing or
  stale events, throughput degradation above 25%, or less than 1 GiB free VRAM
  headroom at the observed peak.
- **Teardown:** local process only; release CUDA memory at exit. No billable
  resources exist.

## Attempts

The non-evidence timing artifacts are `prelaunch/calibration-20260828-190158.json`
and `prelaunch/calibration-20260828-190736.json`.

| Attempt | Condition | Status | Command / note |
| --- | --- | --- | --- |
| `001-20260828-201539-fe6bf60a` | `lr-5e-4` | completed | 449 updates; final full-validation loss 6.257393. |
| `002-20260828-202212-f9a5b7c7` | `lr-1e-3` | completed | 449 updates; final full-validation loss 5.839303. |
| `003-20260828-202838-391f8ac5` | `lr-2e-3` | completed | 449 updates; final full-validation loss 5.542678. |
| `004-20260828-203504-94ec830c` | `lr-4e-3` | completed | 449 updates; final full-validation loss 5.418346. |

Terminal verification is in `artifacts/verification.json`. It recomputed every
checkpoint inventory/hash and reconciled events, configs, manifests,
diagnostics, and coverage.

## Results

All four conditions completed. Total cohort wall time was 1,547.52 seconds
(25m47.5s), below the approved 30-minute envelope.

| Peak LR | Final train loss | Final validation loss |
| ---: | ---: | ---: |
| `5e-4` | 6.312839 | 6.257393 |
| `1e-3` | 5.915539 | 5.839303 |
| `2e-3` | 5.627964 | 5.542678 |
| `4e-3` | 5.504993 | 5.418346 |

The descriptive short-horizon ranking is monotone over the tested grid and
ends at the upper boundary. It does not establish an optimum. See
`observations/O001-local-lr-pipeline-calibration.md`.

Terminal evidence is labeled **provisional** because all detached attempt
manifests contain null Git commit/dirty fields. Exact resolved configs and the
identical six-file run-code content hash `29160e3c…` are present, so the
executed code and inputs remain byte-identifiable.

## Candidate findings

None. This pipeline shakedown is not promoted into a scientific finding.

## Where we stopped

Run completed and terminal artifacts verified. The next operational repair is
to make detached launches populate Git commit/dirty fields before relying on
the same artifact path for definitive evidence. Await user direction on whether
to repair that provenance issue or design the next experiment.
