# Manuscript experiment control

Last updated: 2026-08-31

This is the paper-facing control surface for experiments that may contribute to
the manuscript. Inclusion here is not design approval, launch approval, or a
commitment to run every condition. Execution status comes from verified run
artifacts; manuscript status changes only through an approved finding or an
explicit paper decision.

## How to use this control

Each ladder step is one experiment family. Its intervention parameters are
aggregated inside that family; for example, `A1-H-L1` contains its complete
lambda grid rather than appearing as four separate ladder rows.

Use these independent status dimensions:

- **Paper intent:** `candidate`, `selected`, `required`, or `dropped`.
- **Execution:** `not designed`, `designed`, `launch approved`, `running`,
  `partial`, `completed`, `invalid`, or `superseded`.
- **Evidence scale:** `pilot` or `paper-scale full pass`.
- **Analysis:** `not started`, `in progress`, `completed`, or `not needed`.
- **Manuscript:** `not cited`, `draft cited`, or `finding-backed`.

A checked condition below means that the paper-scale full-pass condition is
complete and verified, not merely that a pilot exists. Pilot evidence is noted
separately. Before changing a checkbox, reconcile the run config, terminal
verification, complete validation coverage, and artifact inventory.

## Scope dashboard

Cells remain paper candidates until explicitly selected or dropped. The
Pythia-14M A4-OL1 cell is complete with corrected evidence from Run 015;
Finding F001 remains discarded because it was based on Run 012, which actually
applied OL1 only at `h`. Run 015 has a run-local candidate observation but no
promoted finding or manuscript claim.
The A7 cell is finding-backed by tentative Finding F002. A7-OL1 is a
completed, descriptively reported candidate with no promoted finding. Their
paper intent remains `candidate` pending explicit subset selection; this does
not select the same cells at larger scales. If every grid point were executed,
the inventory
would contain 30
unique conditions per model, 90 conditions total, 64,080 optimizer boundaries,
and 134,385,500,160 training input tokens. These totals are planning scale, not
an instruction to execute the full matrix.

| Step | Parameter count per model | Pythia-14M | Pythia-70M | Pythia-410M |
| --- | ---: | --- | --- | --- |
| `A0` | 1 | full pass complete: Run 004 | not started | not started |
| `A1-H` | 1 | full pass complete: Run 004 | not started | not started |
| `A1-H-L1` | 4 | 4/4 full pass complete: Run 004 | 0/4 | 0/4 |
| `A1-H-OL1` | 4 | 4/4 full pass complete: Run 009 | 0/4 | 0/4 |
| `A4` | 5 | 5/5 full pass: Run 011; F002 source | 0/5 | 0/5 |
| `A4-OL1` | 5 | 5/5 full pass: corrected Run 015 | 0/5 | 0/5 |
| `A7` | 5 | 5/5 full pass: Run 013; F002 source | 0/5 | 0/5 |
| `A7-OL1` | 5 | 5/5 full pass: Run 014; pilot: Run 010 | 0/5 | 0/5 |
| **Total** | **30** | **30/30 intended full passes complete** | **0/30** | **0/30** |

Runs 006 and 007 remain pilot precedents for A4 and A4-OL1. Run 007 completed
`kappa` values `0`, `0.01`, `0.05`, and `0.1`; its `kappa=0.5` pilot failed
twice from local OOM. Run 011 supersedes the A4 pilot. Run 012 does not complete
the A4-OL1 cell: it realized A4-Z gates plus OL1@h. Run 015 is the separately
numbered four-site correction and now completes all five full-pass conditions.
Run 013 supersedes Run 008 for A7, and Run 014 supersedes Run 010 for A7-OL1.

### Historical Run 011 versus Run 012 digest (not four-site A4-OL1)

Runs 011 and 012 share the full-pass seed, initialization, realized data order,
optimizer schedule, validation cache, and diagnostic implementation. However,
Run 012's pressure objective captured only `h`, not `a,m,h,z`. Analysis 007
reconciles the output counts for that realized A4-Z + OL1@h intervention;
Finding F001 is discarded and this table is retained only as historical data.

| `kappa` | A4 loss | A4 `R_model` | historical A4-Z + OL1@h loss | historical `R_model` | pressure loss delta | pressure `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | 7.2120% | 5.215749 | 8.4094% | -0.254749 | +1.1974 pp |
| 0.01 | 5.466500 | 7.4137% | 5.204504 | 8.5858% | -0.261996 | +1.1721 pp |
| 0.05 | 5.434110 | 8.2059% | 5.195590 | 9.0356% | -0.238520 | +0.8297 pp |
| 0.1 | 5.419642 | 8.9530% | 5.228687 | 9.3435% | -0.190955 | +0.3905 pp |
| 0.5 | 5.659680 | 10.2155% | 5.722666 | 10.2274% | +0.062986 | +0.0119 pp |

Every endpoint covers all 338 complete validation blocks, but none supports a
four-site A4-OL1 effect. Corrected evidence is provided separately by Run 015.

### Current corrected Pythia-14M A4-OL1 result digest

Runs 011 and 015 share the full-pass seed, initialization, realized data order,
optimizer schedule, validation cache, topology, gates, and diagnostic
implementation. Run 015 proves that the differentiated pressure objective
contains all 24 `{a,m,h,z}.layer_{0..5}` tensors on every boundary. It has a
run-local candidate observation; a new numbered frontier analysis and any
finding or manuscript decision remain pending.

| `kappa` | A4 loss | A4 `R_model` | corrected A4-OL1 loss | corrected `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | 7.2120% | 5.458170 | 7.8100% | -0.012327 | +0.5980 pp |
| 0.01 | 5.466500 | 7.4137% | 5.458277 | 8.5867% | -0.008222 | +1.1730 pp |
| 0.05 | 5.434110 | 8.2059% | 5.489803 | 10.4564% | +0.055693 | +2.2506 pp |
| 0.1 | 5.419642 | 8.9530% | 5.548323 | 11.3943% | +0.128681 | +2.4413 pp |
| 0.5 | 5.659680 | 10.2155% | 6.037987 | 12.7134% | +0.378307 | +2.4979 pp |

The effect is threshold-dependent: corrected OL1 improves both matched
outcomes at `kappa=0` and `0.01`; larger thresholds add more logical
opportunity at increasing validation-loss cost. This is descriptive Run 015
evidence, not a restoration of discarded Finding F001.

### Current Pythia-14M A7-OL1 result digest

Runs 013 and 014 share the full-pass seed, initialization, realized data order,
optimizer schedule, validation cache, topology, gates, and diagnostic
implementation. Run 014 adds seven-site post-gate OL1 with `lambda=1` and
trust budget `1`. Analysis 008 now includes its endpoints in the numbered
count-reconciled full-pass frontier and Status Report Number 1 includes the
matched all-site table and corrected single-panel figure; no A7/A7-OL1 finding
has been promoted.

| `kappa` | A7 loss | A7 `R_model` | A7-OL1 loss | A7-OL1 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.468401 | 7.2177% | 5.480184 | 7.0542% | +0.011783 | -0.1634 pp |
| 0.01 | 5.458822 | 7.6176% | 5.475797 | 7.7234% | +0.016975 | +0.1057 pp |
| 0.05 | 5.437888 | 9.1269% | 5.462811 | 9.8630% | +0.024923 | +0.7361 pp |
| 0.1 | 5.428681 | 10.4250% | 5.429497 | 11.7968% | +0.000816 | +1.3717 pp |
| 0.5 | 5.702923 | 15.3868% | 5.829390 | 27.4827% | +0.126466 | +12.0959 pp |

The effect is threshold-dependent: OL1 adds new nondominated `kappa=0.1` and
`0.5` endpoints, but the zero-threshold row regresses. This does not establish
replication, scale transfer, or measured speedup.

## Paper-scale global training contract

These are the current control defaults. They extend the mapped Pythia recipe
used in Runs 004 and 009 to one MiniPile pass. They must still be restated and
confirmed in every numbered run design. Hardware, cloud price, and launch scope
are deliberately not global parameters.

### Model scale and learning rate

| Model | Model ID for architecture config | `(L, d, heads, d_head, V)` | Peak LR | Minimum LR | Status |
| --- | --- | --- | ---: | ---: | --- |
| Pythia-14M | `EleutherAI/pythia-14m-deduped` | `(6, 128, 4, 32, 50,304)` | `1.0e-3` | `1.0e-4` | executed in Runs 004/009/011/012/013/014/015 |
| Pythia-70M | `EleutherAI/pythia-70m-deduped` | `(6, 512, 8, 64, 50,304)` | `1.0e-3` | `1.0e-4` | proposed; exact revision must be pinned |
| Pythia-410M | `EleutherAI/pythia-410m-deduped` | `(24, 1,024, 16, 64, 50,304)` | `3.0e-4` | `3.0e-5` | proposed; exact revision must be pinned |

The widths, head layouts, common 2M-token batch, and peak learning rates follow
the official Pythia recipes. The minimum is 10% of peak, matching the mapped
Runs 004/009/011/012 schedule. Models are always constructed from the pinned architecture
config with random initialization; released weights are not loaded. The mapping
uses Pythia `small_init` and `wang_init`, but is not claimed to be an exact
GPT-NeoX/DeepSpeed reproduction.

Sources:

- <https://github.com/EleutherAI/pythia>
- <https://github.com/EleutherAI/pythia/blob/main/models/14M/pythia-14m.yml>
- `runs/004-2026-08-29-pythia14m-full-pass-l1n/config.yaml`
- `runs/009-2026-08-30-pythia14m-full-pass-ol1/config.yaml`
- `runs/011-2026-08-30-pythia14m-full-pass-a4z/config.yaml`
- `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/config.yaml`
- `runs/013-2026-08-30-pythia14m-full-pass-a7/config.yaml`
- `runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/config.yaml`
- `runs/015-2026-08-31-pythia14m-corrected-a4-ol1/config.yaml`

### Shared scientific settings

| Setting | Selected value | Rationale |
| --- | --- | --- |
| Dataset | `JeanKaddour/minipile` at `18ad1b0c701eaa0de03d3cecfdd769cbc70ffbd0` | Fixed smaller-scale corpus for intervention comparisons. |
| Tokenizer | Pythia tokenizer, append EOS, no added encode-time special tokens; current pinned 14M tokenizer revision `7386d9a4ae45aef494a6e704910394def3037fc5` | Keeps the established cache, vocabulary, sequence semantics, and cross-condition data identity. Confirm the shared tokenizer contract before the first 70M/410M run. |
| Sequence length | `2,048` | Matches Pythia and the ladder's architecture-ceiling workload. |
| Training cache | 1,000,000 documents; 1,491,711,416 tokens; 728,374 complete blocks; 1,464-token excluded tail | Uses the complete pinned MiniPile training cache. |
| Global batch | `1,024` sequences = `2,097,152` input tokens/update | Matches the Pythia 2M-token recipe. Microbatch decomposition is a hardware decision. |
| Horizon | `712` optimizer boundaries | `ceil(728,374 / 1,024)` gives one complete shuffled pass plus 714 wrapped blocks: 729,088 sequences and 1,493,172,224 input tokens per condition. |
| Microbatch / accumulation | 14M reference: `32 x 32` on one GPU; 70M/410M: calibrate, while preserving `world_size x microbatch x accumulation = 1,024` | Preserves the scientific global batch while allowing safe memory fit. A changed decomposition and its numerical implications must be disclosed. |
| Seeds | model `1234`; data order `1234` | Matches Pythia/Run 004 and enables paired initialization and ordered block starts within each model size. |
| Optimizer | AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`; exclude bias and LayerNorm from decay | Mapped Pythia recipe used by the completed full-pass runs. |
| Gradient clipping | global L2 norm `1.0` immediately before AdamW | Pythia recipe and operational contract. For OL1, the clipped task gradient is the AdamW/task reference direction. |
| LR schedule | GPT-NeoX-v1 pre-step semantics; 1% linear warmup, cosine decay to 10% of peak | Matches Runs 004/009 while scaling peak/minimum LR by model size. |
| Precision | dynamic FP16 autocast; FP32 parameters and optimizer state | Matches Pythia and the full-pass 14M controls. BF16 pilots are useful but are not numerically interchangeable with this target. |
| Dropout | hidden `0.0`; attention `0.0` | Pythia recipe and deterministic paired comparison. |
| Attention | SDPA/Flash for training; eager, uncached causal attention for logical diagnostics | Training throughput and exact operand observability require different tested paths. |
| Validation | all 500 documents; 338 complete sequences; 692,224 tokens; report 1,444-token tail; evaluate at step 1 and final | Repository complete-validation contract and paired quality measurement. |
| Checkpoints | model at `0,1,2,4,8,16,32,64,128,256,512,712`; optimizer at `256,512,712`; retain final | Matches the full-pass source runs and preserves trajectory/post-hoc options. Reconfirm storage for larger scales before launch. |
| Logging | every optimizer boundary | Required for matched loss, throughput, ETC, gradient, and failure auditing. |

### Required measurements

Unless a design explicitly narrows them, every selected paper-scale condition
should retain:

- final task loss under complete validation and the matched step-1 loss;
- count-first exact-zero and near-zero (`epsilon in {0.001, 0.01}`) activation
  statistics, RMS/L2, sums, and finite/nonfinite counts for all relevant sites
  among `a,m,h,q_post,k_post,v,z,attention_output`;
- per-layer/role weight norms with named parameters and element counts;
- exact logical-product integer counts, `R_block`, `R_model`, and the integer
  numerator/denominator for `R_model_max`;
- task/pressure gradient interaction for pressure runs, including OL1 conflict,
  projection, adaptive direction, raw ratio, trust scale, and final ratio;
- training schedule, initialization, config, code, environment, cache, and
  checkpoint identities;
- the final checkpoint and enough cache/code identity for approved post-hoc
  work.

These logical-product quantities are opportunities, not measured sparse-kernel
speedups.

## Intervention families and parameter grids

All gate thresholds use the operational boundary convention: equality survives.
`A4` maps to operational topology `A4-Z`; `A7` maps to `A7-Z-POST`. One common
`kappa` is used across all active sites in A4/A7. The pressure suffix is not a
topology.

| Step | Gates | Pressure | Aggregated parameter set | `R_model_max` for 14M / 70M / 410M |
| --- | --- | --- | --- | --- |
| `A0` | stock GeLU at `h`; no active gate sites | none | singleton | `0.0% / 0.0% / 0.0%` |
| `A1-H` | ReLU at `h` | none | singleton | `4.3% / 12.4% / 24.9%` |
| `A1-H-L1` | ReLU at `h` | naive L1 at `h` | `lambda in {0.05, 0.1, 0.5, 1.0}` | `4.3% / 12.4% / 24.9%` |
| `A1-H-OL1` | ReLU at `h` | OL1 at `h` | `lambda in {0.05, 0.1, 0.5, 1.0}`, `step_budget=1.0` | `4.3% / 12.4% / 24.9%` |
| `A4` | one-sided threshold at `a,m,h,z` | none | `kappa in {0, 0.01, 0.05, 0.1, 0.5}` | `12.8% / 37.1% / 74.8%` |
| `A4-OL1` | same A4 gates | OL1 at `a,m,h,z` | `kappa in {0, 0.01, 0.05, 0.1, 0.5}`, `lambda=1.0`, `step_budget=1.0` | `12.8% / 37.1% / 74.8%` |
| `A7` | one-sided at `a,m,h,z`; symmetric at `q_post,k_post,v` | none | `kappa in {0, 0.01, 0.05, 0.1, 0.5}` | `30.0% / 49.4% / 87.2%` |
| `A7-OL1` | same mixed A7 gates | OL1 at all seven active sites | `kappa in {0, 0.01, 0.05, 0.1, 0.5}`, `lambda=1.0`, `step_budget=1.0` | `30.0% / 49.4% / 87.2%` |

Parameter provenance:

- A1-H L1 lambdas: Run 004.
- A1-H OL1 lambdas: Run 009; same four-lambda grid and reused Run 004 controls.
- A4 kappas: Run 006 pilot and Run 011 full pass.
- A7 kappas: Run 008.
- A4-OL1 fixed OL1 settings: Run 007 pilot and completed corrected Run 015
  (`lambda=1.0`, `step_budget=1.0`). Historical Run 012 declared those settings
  but actually captured only `h`; it is not an A4-OL1 full pass. Run 010
  independently executed the same fixed settings for the A7-OL1 pilot.

OL1 pressure uses the unweighted mean across captured site/layer tensors. This
means expanding pressure from one to four or seven sites changes the set of
equally weighted tensors; it is not an element-count-weighted pressure.

## Detailed execution checklist

### Pythia-14M

Paper intent remains `candidate` for every step. Manuscript status is
`finding-backed` for A7 through tentative Finding F002. Finding F001 is
discarded, so A4-OL1 is not finding-backed. A7-OL1 is reported descriptively in Status
Report Number 1 without a promoted finding; every other step remains `not
cited` unless updated explicitly.

#### `A0` - stock GeLU control

- [x] Singleton - paper-scale full pass completed and verified in Run 004.
- Analysis: paired analysis pending; paper inclusion undecided.

#### `A1-H` - ReLU control

- [x] Singleton - paper-scale full pass completed and verified in Run 004.
- Analysis: paired analysis pending; reused by Run 009 without rerunning.

#### `A1-H-L1` - naive L1 at `h`

- [x] `lambda=0.05` - Run 004.
- [x] `lambda=0.1` - Run 004.
- [x] `lambda=0.5` - Run 004.
- [x] `lambda=1.0` - Run 004.
- Analysis: Analysis 003 provides the full-pass matched L1/OL1 comparison;
  paper selection remains open.

#### `A1-H-OL1` - OL1 at `h`

- [x] `lambda=0.05`, `step_budget=1.0` - Run 009.
- [x] `lambda=0.1`, `step_budget=1.0` - Run 009.
- [x] `lambda=0.5`, `step_budget=1.0` - Run 009.
- [x] `lambda=1.0`, `step_budget=1.0` - Run 009.
- Analysis: Analysis 003 completed the matched L1/OL1
  quality--logical-opportunity and gradient-interaction comparison.

#### `A4` - one-sided threshold at `a,m,h,z`

- [x] `kappa=0` - paper-scale full pass completed in Run 011.
- [x] `kappa=0.01` - paper-scale full pass completed in Run 011.
- [x] `kappa=0.05` - paper-scale full pass completed in Run 011.
- [x] `kappa=0.1` - paper-scale full pass completed in Run 011.
- [x] `kappa=0.5` - paper-scale full pass completed in Run 011.
- Analysis: Analyses 004 and 007 contain the count-reconciled frontier; Run
  006 remains a seed-0, BF16 pilot and is not pooled with full-pass evidence.

#### `A4-OL1` - A4 plus OL1 at `a,m,h,z`

- [x] `kappa=0`, `lambda=1.0`, `step_budget=1.0` - corrected paper-scale full pass completed and verified in Run 015.
- [x] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0` - corrected paper-scale full pass completed and verified in Run 015.
- [x] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0` - corrected paper-scale full pass completed and verified in Run 015.
- [x] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0` - corrected paper-scale full pass completed and verified in Run 015.
- [x] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0` - corrected paper-scale full pass completed and verified in Run 015.
- Historical note: Run 012 completed five A4-Z + OL1@h conditions, not this
  four-site cell. Analysis 007 retains that historical comparison; F001 is
  discarded. Run 015 has a verified run-local observation; a new numbered
  corrected frontier analysis is pending. Analysis 002 remains the separate
  pilot comparison.

#### `A7` - mixed threshold at seven sites

- [x] `kappa=0` - paper-scale full pass completed and verified in Run 013.
- [x] `kappa=0.01` - paper-scale full pass completed and verified in Run 013.
- [x] `kappa=0.05` - paper-scale full pass completed and verified in Run 013.
- [x] `kappa=0.1` - paper-scale full pass completed and verified in Run 013.
- [x] `kappa=0.5` - paper-scale full pass completed and verified in Run 013.
- Pilot identity: seed 0, global batch 64, 581 updates, BF16.
- Paper-scale identity: seed 1234, global batch 1,024, 712 updates, dynamic
  FP16; five condition-parallel Secure A100 workers completed in Run 013.
- Analysis: Analysis 008 provides the numbered count-reconciled A7/A7-OL1
  table and expanded full-pass frontier. Its A4/A7 comparison supports
  tentative Finding F002; the A7/A7-OL1 comparison remains descriptive.

#### `A7-OL1` - mixed A7 plus OL1 at all seven sites

- [x] `kappa=0`, `lambda=1.0`, `step_budget=1.0` - paper-scale full pass completed and verified in Run 014.
- [x] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0` - paper-scale full pass completed and verified in Run 014.
- [x] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0` - paper-scale full pass completed and verified in Run 014.
- [x] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0` - paper-scale full pass completed and verified in Run 014.
- [x] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0` - paper-scale full pass completed and verified in Run 014.
- Paper-scale identity: seed 1234, global batch 1,024, 712 updates, dynamic
  FP16; five condition-parallel Secure A100 workers completed in Run 014.
- Analysis: `analyses/001-2026-08-30-run008-vs-run010-all-site-ol1/`
  remains pilot precedent. Analysis 008 now provides the numbered paper-scale
  A7/A7-OL1 frontier and interleaved count-reconciled table. Status Report
  Number 1 presents both; no finding has been promoted for that comparison.

### Pythia-70M

All cells are candidates, not designed, not run, not analyzed, and not cited.
Before the first design, pin the exact architecture/tokenizer revisions and
calibrate microbatch, accumulation, precision health, checkpoint size, and ETC.

#### `A0`

- [ ] Singleton.

#### `A1-H`

- [ ] Singleton.

#### `A1-H-L1`

- [ ] `lambda=0.05`.
- [ ] `lambda=0.1`.
- [ ] `lambda=0.5`.
- [ ] `lambda=1.0`.

#### `A1-H-OL1`

- [ ] `lambda=0.05`, `step_budget=1.0`.
- [ ] `lambda=0.1`, `step_budget=1.0`.
- [ ] `lambda=0.5`, `step_budget=1.0`.
- [ ] `lambda=1.0`, `step_budget=1.0`.

#### `A4`

- [ ] `kappa=0`.
- [ ] `kappa=0.01`.
- [ ] `kappa=0.05`.
- [ ] `kappa=0.1`.
- [ ] `kappa=0.5`.

#### `A4-OL1`

- [ ] `kappa=0`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0`.

#### `A7`

- [ ] `kappa=0`.
- [ ] `kappa=0.01`.
- [ ] `kappa=0.05`.
- [ ] `kappa=0.1`.
- [ ] `kappa=0.5`.

#### `A7-OL1`

- [ ] `kappa=0`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0`.

### Pythia-410M

All cells are candidates, not designed, not run, not analyzed, and not cited.
Before the first design, pin the exact architecture/tokenizer revisions and
calibrate microbatch, accumulation, activation checkpointing, precision health,
checkpoint size, transfer time, and ETC. Activation checkpointing is currently
off in the reference contract; enabling it is an explicit execution change.

#### `A0`

- [ ] Singleton.

#### `A1-H`

- [ ] Singleton.

#### `A1-H-L1`

- [ ] `lambda=0.05`.
- [ ] `lambda=0.1`.
- [ ] `lambda=0.5`.
- [ ] `lambda=1.0`.

#### `A1-H-OL1`

- [ ] `lambda=0.05`, `step_budget=1.0`.
- [ ] `lambda=0.1`, `step_budget=1.0`.
- [ ] `lambda=0.5`, `step_budget=1.0`.
- [ ] `lambda=1.0`, `step_budget=1.0`.

#### `A4`

- [ ] `kappa=0`.
- [ ] `kappa=0.01`.
- [ ] `kappa=0.05`.
- [ ] `kappa=0.1`.
- [ ] `kappa=0.5`.

#### `A4-OL1`

- [ ] `kappa=0`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0`.

#### `A7`

- [ ] `kappa=0`.
- [ ] `kappa=0.01`.
- [ ] `kappa=0.05`.
- [ ] `kappa=0.1`.
- [ ] `kappa=0.5`.

#### `A7-OL1`

- [ ] `kappa=0`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.01`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.05`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.1`, `lambda=1.0`, `step_budget=1.0`.
- [ ] `kappa=0.5`, `lambda=1.0`, `step_budget=1.0`.

## Paired intervention analysis control

Every pair requires the same model size, initialization, realized data order,
global batch, schedule, precision contract, validation cache, diagnostic code,
and checkpoint boundary unless stated otherwise. Raw pilot/full-pass pairs are
not matched comparisons.

| ID | Paired comparison | Primary question | Matching rule | Current status |
| --- | --- | --- | --- | --- |
| P01 | `A0 -> A1-H` | What changes when stock GeLU at `h` is replaced by ReLU? | Same size and all training identities; singleton pair. | 14M evidence available in Run 004; formal paper analysis pending. |
| P02 | `A1-H -> A1-H-L1(lambda)` | What is the effect of naive L1 pressure at `h`? | Compare each lambda with the shared ReLU control. | 14M evidence available in Run 004; paper selection pending. |
| P03 | `A1-H-L1(lambda) -> A1-H-OL1(lambda)` | What changes when naive pressure is replaced by orthogonal pressure? | Match lambda exactly; reuse the same A1-H control for context. | 14M full-pass Analysis 003 complete from Runs 004/009. |
| P04 | `A1-H -> A4(kappa=0)` | What changes when ReLU-equivalent one-sided gates expand from `h` to `a,m,h,z`? | Use A4 at zero threshold. | Matched 14M full-pass endpoints available from Runs 004/011; Analysis 004 places them on the common frontier. |
| P05 | `A4(0) -> A4(kappa>0)` | What is the threshold-strength effect at fixed A4 topology? | Within-run kappa contrast; zero row is mandatory. | 14M full-pass evidence complete in Run 011 and Analyses 004/007. |
| P06 | `A4(kappa) -> A4-OL1(kappa)` | Does all-active-site OL1 change A4 at fixed threshold? | Match kappa; `lambda=1`, `step_budget=1`. | Five corrected Run 011/015 pairs are verified in the Run 015 observation; a numbered frontier analysis is pending. Run 012/Analysis 007 do not answer the four-site question; F001 remains discarded. |
| P07 | `A4(kappa) -> A7(kappa)` | What changes when symmetric post-RoPE `q,k,v` gates are added? | Match kappa. At `kappa=0`, symmetric gates are identity, providing a null/equivalence check. | Five 14M full-pass pairs are complete in Analysis 008; tentative Finding F002 approved. |
| P08 | `A7(kappa) -> A7-OL1(kappa)` | Does all-seven-site OL1 change A7 at fixed threshold? | Match kappa; `lambda=1`, `step_budget=1`. | Five 14M full-pass pairs are count-reconciled in Analysis 008 and reported descriptively in Status Report Number 1; no finding promoted. |
| P09 | `A4-OL1(kappa) -> A7-OL1(kappa)` | Does attention-site expansion matter when OL1 is present? | Match kappa, lambda, and trust budget. | Corrected Run 015 and Run 014 are complete; a matched numbered analysis is pending. Historical Run 012 is A4-Z + OL1@h and is ineligible. |
| P10 | `(A7-OL1 - A7) - (A4-OL1 - A4)` | Does the OL1 effect interact with topology expansion? | Four-condition difference-in-differences at each matched kappa. | All four 14M legs in Runs 011/013/014/015 are complete; the difference-in-differences analysis is pending. |
| P11 | within `A1-H-L1` and within `A1-H-OL1` | What are the lambda dose-response and saturation patterns? | Compare the four lambdas to one shared A1-H control; do not treat lambda as equally spaced. | 14M full-pass Analysis 003 complete. |
| P12 | within A4/A7, with and without OL1 | What are the kappa dose-response and quality-logical-opportunity frontiers? | Always include the family's own `kappa=0` row. | A4, corrected A4-OL1, A7, and A7-OL1 are complete; a numbered analysis incorporating Run 015 is pending. Analysis 007 remains historical A4-Z + OL1@h. |
| P13 | same step/parameters across 14M, 70M, 410M | Which intervention effects persist with scale? | Match the complete condition identity; use size-specific recipe LR and report architecture separately. | Not started. |
| P14 | all selected steps within one size | Which rows are nondominated in validation loss versus `R_model`? | Same evaluation workload and metric implementation; report logical opportunity, not speedup. | Run 015's corrected A4-OL1 endpoints are verified; a new numbered frontier analysis is pending. Analysis 008's Run 012 points must remain labeled A4-Z + OL1@h. |
| P15 | `A1-H` versus L1/OL1 at `h` | Does targeted `h` pressure produce sparsity spillover at untargeted attention sites? | Sitewise count-first exact/near-zero and RMS changes at matched lambda/control. | Run 004 observations and full-pass Analysis 003 complete; finding selection remains open. |
| P16 | OL1 across A1-H, A4, and A7 | How do conflict, projection, trust saturation, and correction ratios change with pressure topology? | Use training-time OL1 boundary metrics; checkpoint-only reconstruction is invalid. | Full-pass A1-H, corrected four-site A4, and A7-OL1 metrics exist; cross-topology analysis is pending. |
| P17 | measured `R_model` versus analytic `R_model_max` across scales | How much of topology-conditioned reach is realized, and how does the ceiling scale? | Preserve integer counts and architecture/workload identity; `U_arch` needs its numerator caveat. | 14M partial evidence; 70M/410M not started. |
| P18 | intervention effect by model size | Does scale modify nonlinearity, pressure, or threshold effects? | Analyze paired deltas within each size before comparing deltas across sizes. | Not started. |

Interpretation cautions:

- `A1-H -> A4(kappa>0)` is a composite change in topology and threshold
  magnitude. Use P04 and P05 to separate the zero-threshold topology expansion
  from within-A4 threshold strength.
- `A4 -> A7` is cleanest at matched kappa because A4's four gate sites remain
  unchanged and A7 adds symmetric `q_post,k_post,v` gates. At `kappa=0` those
  added symmetric gates are identity in values and gradients.
- L1 and OL1 must be compared at matched lambda; the trust budget can make the
  realized OL1 correction nonlinear in lambda.
- Validation loss, exact zero, near-zero mass, logical opportunity, and runtime
  speed are separate outcomes.

## Open decisions before selecting the paper subset

- Whether Run 014's threshold-dependent A7-OL1 result, now included in Analysis
  008, should receive a tentative finding; analysis alone does not promote it.
- Whether corrected Run 015 results, after a new numbered analysis, warrant a
  new A4-OL1 finding or manuscript revision. Run 012 is settled as A4-Z + OL1@h.
- Whether all four A1-H lambda rows are needed at 70M/410M or a preregistered
  subset should be selected from 14M evidence.
- Whether one seed (`1234`) is sufficient for the paper claim or selected
  effects require additional matched seeds. Additional seeds are new conditions,
  not silent repetitions.
- Whether 70M/410M should use the currently pinned shared Pythia tokenizer/cache
  identity; confirm before their first design.
- Whether dynamic FP16 remains healthy for OL1 at 70M/410M; a BF16 change must
  be treated as a disclosed execution/numerical change.
- Whether all Run 004 checkpoint steps are worth retaining at larger scales;
  final checkpoints remain required unless explicitly waived after the post-hoc
  checklist.
- Which paired analyses become manuscript figures/tables and which remain
  exploratory observations.

## Evidence sources

- Ladder definition: `manuscript/artifacts/sparsification-ladder.md` and
  `manuscript/artifacts/sparsification-ladder.tex`.
- Full-pass L1/control source: `runs/004-2026-08-29-pythia14m-full-pass-l1n/`.
- A4 threshold pilot: `runs/006-2026-08-29-pythia14m-a4z-threshold-local/`.
- A4-OL1 pilot: `runs/007-2026-08-29-pythia14m-a4z-threshold-ol1-local/`.
- A7 threshold pilot: `runs/008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local/`.
- Full-pass A1-H OL1 source: `runs/009-2026-08-30-pythia14m-full-pass-ol1/`.
- A7-OL1 pilot: `runs/010-2026-08-30-pythia14m-a7-z-post-mixed-threshold-ol1-local/`.
- Full-pass A4 source: `runs/011-2026-08-30-pythia14m-full-pass-a4z/`.
- Historical A4-Z + OL1@h source:
  `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/`.
- Corrected full-pass A4-OL1 source:
  `runs/015-2026-08-31-pythia14m-corrected-a4-ol1/`.
- Full-pass A7 source:
  `runs/013-2026-08-30-pythia14m-full-pass-a7/`.
- Full-pass A7-OL1 source:
  `runs/014-2026-08-31-pythia14m-full-pass-a7-ol1/`.
- Historical A4 versus A4-Z + OL1@h synthesis (not four-site evidence):
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/` and
  `research/findings/F001-a4-ol1-improves-moderate-threshold-frontier.md`.
- Full-pass frontier with A7:
  `analyses/008-2026-08-31-full-pass-frontier-with-a7/` and
  `research/findings/F002-a7-extends-a4-logical-opportunity.md`.
- Status Report Number 1:
  `manuscript/reports/01-2026-08-30-status-update/status-update.tex` and
  `manuscript/reports/01-2026-08-30-status-update/status-update.pdf`.
- Existing paired pilots: `analyses/001-2026-08-30-run008-vs-run010-all-site-ol1/`
  and `analyses/002-2026-08-30-run006-vs-run007-partial-a4z-ol1/`.
- Operational contracts: `research/DATA.md`, `research/METHODS.md`,
  `research/METRICS.md`, and `research/COMPUTE.md`.
