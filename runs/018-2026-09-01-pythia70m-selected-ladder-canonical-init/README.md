# Run 018 - Pythia-70M selected ladder with canonical initialization

## Status

Implemented and locally verified on 2026-09-01; awaiting explicit launch
approval. No Run 018 cloud resource or scientific attempt has been created.

Run 017 stopped before its first training boundary because identical source
initialization code produced different parameter bytes under the local CPU and
remote Linux CUDA Torch wheels. Run 018 preserves the approved Run 017 science
but removes that environment-dependent draw from the remote lifecycle: one
locally generated initialization and its post-initialization CPU RNG state are
serialized, byte/hash pinned, transferred, strictly loaded on CPU, checked
against the approved parameter hash, and only then moved to the training GPU.

## Question and matched design

Does the relationship between measured logical-product opportunity (`R_model`)
and complete-validation loss observed at Pythia-14M persist for the selected
ladder at randomly initialized Pythia-70M?

The experiment has twelve independent conditions at seed 1234:

- A0/GELU and A1-H/ReLU;
- A4-OL1 at kappa 0, 0.01, 0.05, 0.1, and 0.5;
- A7-OL1 at the same five kappa values.

All conditions use the pinned 70,426,624-parameter Pythia-70M architecture,
the same canonical initial parameter bytes, the same 712-boundary one-pass
schedule over 1,493,172,224 MiniPile training tokens, global batch 1,024 as
4 x 256 accumulation, AdamW, the GPT-NeoX-v1 pre-step learning-rate schedule,
FP16 dynamic loss scaling, and a final recovery checkpoint. Dropout is zero.
Only topology/gates and the declared OL1 sites differ. This is a selected
ladder promotion, not a broad larger-model ablation.

The implementation reuses the already verified Run 004 scientific lifecycle
through the frozen Run 017 70M adaptation. It does not reimplement training,
diagnostics, OL1 boundaries, logical products, or TEAL. Run 018 adds only the
checked-in architecture constructor, canonical initialization generator and
loader, exact artifact/RNG validation, and thin entry-point wrappers. Released
model weights are never requested or loaded.

The four-condition sentinel is A0, A1-H, A4 kappa 0, and A7 kappa 0. The eight
nonzero-kappa A4/A7 conditions remain a second wave and require a new explicit
decision after the sentinel is reviewed.

## Validation and measurements

Step 1 and the reloaded final checkpoint each evaluate all 500 MiniPile
validation documents: 338 complete 2,048-token blocks and 692,224 input tokens;
the excluded 1,444-token tail is recorded. Integer counts are pooled before any
rate is computed.

Final diagnostics retain exact and near-zero activation counts at thresholds
0, 0.001, and 0.01; activation RMS/L2/finite counts; per-layer parameter norms;
full logical-product integers, `R_block`, `R_model`, and declared-topology
`R_model_max`; and all OL1 task/pressure norms, dot products, cosines, conflicts,
raw/final ratios, trust scales, clipping, capture identities, and overflow
behavior. A4 captures 24 pressure sites and A7 captures 42. The final recovery
checkpoint retains model, optimizer, loss scaler, schedule identity, and all
Python/NumPy/Torch CPU/CUDA RNG states.

A0 and A1-H additionally run the Analysis 005/006 post-hoc uniform TEAL
protocol over sites `a,m,h,z`. Thresholds are calibrated on the first ten
complete source-order training blocks for target sparsities 0.0 through 0.9;
every point evaluates all 338 validation blocks. The zero-threshold loss must
reproduce the source within `5e-4`.

## Initialization identity

- schedule SHA-256:
  `d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`;
- model artifact: 281,715,344 bytes, SHA-256
  `024e01975e1a52ead00340afd7a5c3f0b7c2fa0542d9dd5998e648ec14f73501`;
- post-initialization RNG artifact: 14,823 bytes, SHA-256
  `ff839f490cbbbec528181113451802f52c734fb45ae693fc800991bc2be36762`;
- strict-load parameter SHA-256:
  `e8b8d8e48880f8ff25e421ed29b04a81eb417300f2b4a01a8c4d56f2591a1062`.

The safetensors artifact has 76 tensors and 281,706,496 tensor bytes. The two
binary artifacts are launch inputs excluded from Git; their tracked provenance
is `prelaunch/initialization/metadata.json`. Each Pod must verify both files
before the non-evidence preflight or a worker can run.

## Interpretation

At one seed, support means that promoted endpoints/frontiers retain a useful
validation-loss versus measured-`R_model` tradeoff. Loss collapse, negligible
measured opportunity, or qualitatively reversed ordering would refute that
expectation. `R_model` and `R_model_max` are logical-product opportunities, not
runtime speedups. One seed and one promoted larger scale limit generalization;
the design estimates persistence, not a scaling law or causal mediation.

The operational definitions in `research/METHODS.md` and
`research/METRICS.md` execute. No new design-material manuscript difference is
introduced. The result can affect the manuscript's proposed scale-persistence
claim and the corresponding `R_model` versus validation-loss figure/table only
after evidence is complete and the user approves consolidation.

## Verification and launch boundary

The focused suite checks the 12-condition partition, inherited schedule and
70M ceilings, artifact/config fail-closed behavior, exact files and metadata,
strict model loading, post-initialization CPU RNG restoration, CPU-before-CUDA
worker wiring, OL1 capture dimensions, and TEAL semantics. The full bootstrap
suite is also required before launch. On 2026-09-01 the focused suite passed
16/16 and the full bootstrap suite passed 183/183. The implemented config
SHA-256 is
`1bee5a9a9a7a667180fe4782bbfab5a14504f49bdac11d94617ee3be2c827c00`;
the run-code content SHA-256 is
`a79689e8ba78d8cf57ec5c4107101b39061cf8e41a80bed6cb8eb548b1efdb83`.

The local CPU environment can verify construction and exact artifact loading
but cannot execute the CUDA transfer or representative training. The proposed
remote workflow therefore begins with a non-evidence A7 H200 preflight that
must load this artifact and complete the exact boundary/validation/diagnostic
probe before any scientific attempt. See `DEPLOYMENT_PLAYBOOK.md`. Live price,
stock, balance, maximum cost, and storage will be refreshed for the launch
approval; this README is not launch authorization.

## Live sentinel proposal

The read-only 2026-09-01 11:19 UTC audit found zero Pods and only the intended
pre-existing 100 GB Standard network volume in `EUR-IS-1`, continuing at the
posted `$0.01/hour`. Secure H200 is `$4.59/GPU-hour`, 141 GB, with low stock in
four pinned candidate data centers. The proposal is four independent H200 Pods,
one sentinel condition each, for `$18.36/hour` combined. Each has 30 GB container
disk, 25 GB Pod volume, no network-volume attachment, and an independent
6.5-hour deletion guard.

The hard new-resource envelope is 26 GPU-hours: `$119.34` compute plus about
`$0.21` Pod disk, or `$119.55`. Including up to `$0.065` of the unrelated retained
volume during the same window gives a `$119.62` maximum account outflow. The
live balance was `$120.84`, leaving only about `$1.22` guard headroom. Actual
completion is expected before the guard, but the exact ETC is intentionally
deferred to the five-boundary/full-validation preflight because no valid 70M
boundary timing exists yet.

Each Pod receives 6,251,350,503 bytes (5.822 GiB) of hash-pinned cache and
initialization files, plus the committed source bundle. The expected final
recovery checkpoint is approximately 0.85 GB per condition (model plus AdamW
state), with diagnostics, logs, identities, and A0/A1-H TEAL records retrieved
in addition. Hash verification is required at both ends.

Twelve simultaneous H200 Pods are technically unnecessary and operationally
poor here: the approved design needs a scientific sentinel, live H200 stock is
low, and twelve Pods would cost `$55.08/hour` while duplicating roughly 70 GiB
of input transfer. After a successful sentinel, the eight remaining conditions
can run independently on eight GPUs in parallel (`$36.72/hour` at the current
H200 price), subject to a fresh price, stock, balance, and launch decision.
