# Status Report Number 1 notes

## Post-hoc measurement note for `a` and `z`

## Observation

Tables 6, 8, and 9 report `a` and `z` as `n.m.` because the terminal
activation diagnostics for Runs 004 and 009 did not request those sites. The
source diagnostics captured `m`, `h`, `q_post`, `k_post`, `v`, and
`attention_output`. The last quantity is the output of `W_o` and cannot be used
as a substitute for `z`, which is the concatenated attention context
immediately before `W_o`.

This gap does not require retraining. Activation statistics can be reconstructed
from a final model checkpoint by replaying the pinned validation set. Historical
gradient interactions cannot be reconstructed this way.

## Verified local material

The local inventory was checked on 30 August 2026.

- All ten final `step_000712` model checkpoints are present: six Run 004
  conditions and four Run 009 conditions.
- Every `model.safetensors` file is 56,279,344 bytes. All ten SHA-256 values
  match the corresponding `diagnostics/checkpoint_inventory.json` entry.
- The validation cache is present at
  `data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin`.
- The validation cache is 2,774,672 bytes with SHA-256
  `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`.
- Coverage is 500 documents, 338 complete 2,048-token blocks, 692,224 input
  tokens, and a recorded 1,444-token excluded tail.

The source checkpoints are:

| Run | Condition | Attempt |
| --- | --- | --- |
| 004 | `gelu-control` | `001-20260829-221007-bb5288c8` |
| 004 | `relu-control` | `002-20260829-231305-195b22c6` |
| 004 | `relu-l1n-0p05` | `003-20260829-221010-062e6eae` |
| 004 | `relu-l1n-0p1` | `004-20260829-221037-231415c7` |
| 004 | `relu-l1n-0p5` | `005-20260829-221010-866aa09f` |
| 004 | `relu-l1n-1` | `006-20260829-221054-e7b8a083` |
| 009 | `relu-ol1-0p05` | `001-20260830-120005-d3003910` |
| 009 | `relu-ol1-0p1` | `002-20260830-123426-c5b8515f` |
| 009 | `relu-ol1-0p5` | `003-20260830-115346-94d4aa8a` |
| 009 | `relu-ol1-1` | `004-20260830-112819-e750c51a` |

For each row, the final model directory is:

```text
runs/<run-folder>/artifacts/attempts/<attempt>/checkpoints/step_000712/
```

## Measurement contract

Measure the seven canonical sites in one consistent pass:

```text
a, m, h, z, q_post, k_post, v
```

The report may display `q_post` and `k_post` as `q` and `k`. Do not capture
`q_pre` or `k_pre` for these tables. Do not replace `z` with
`attention_output`.

For every condition:

- load the final checkpoint and its local `config.json` so the trained
  activation topology is restored exactly;
- evaluate without clipping or any other new intervention;
- use the complete pinned validation cache in deterministic block order;
- preserve the Run 004 diagnostic precision path: FP32 parameters, FP16 CUDA
  autocast, and the recipe attention context;
- capture all six layers at every requested site;
- accumulate integer totals, finite/nonfinite counts, exact-zero counts, and
  near-zero hits at `0`, `1e-3`, and `1e-2`;
- pool counts across batches and layers before dividing;
- retain activation RMS and the underlying sums so later scale comparisons do
  not require another pass.

The target exact-zero statistic for site `s` is:

```text
sum_layers(exact_zero_count_s) / sum_layers(total_s)
```

Do not average per-batch or per-layer percentages.

## Recommended implementation

This recommendation was written when Analysis 004 was still the next available
number, but that number was later assigned to another completed synthesis. The
diagnostic remains deferred; if approved later, it must use the next available
analysis number recorded in `research/INDEX.md`. Runs 004 and 009 are
append-only records; do not add or replace files inside their artifact
directories.

Reuse the existing tested components:

- `src/sparsity_research/pythia.py::load_checkpoint_pythia` to load the local
  checkpoint and reapply its topology;
- `src/sparsity_research/capture.py::ActivationCapture` for the canonical
  `a`, `m`, `h`, `z`, `q_post`, `k_post`, and `v` ports;
- `src/sparsity_research/metrics.py::ActivationAccumulator` for count-first
  statistics and pooled site rows;
- `src/sparsity_research/evaluation.py::evaluate_complete_blocks` for complete
  validation coverage;
- `runs/004-2026-08-29-pythia14m-full-pass-l1n/diagnostics.py` for the original
  FP16 diagnostic execution context;
- `runs/006-2026-08-29-pythia14m-a4z-threshold-local/diagnostics.py` as the
  existing reference that captures the canonical pre-`W_o` site `z`.

The analysis should write, at minimum:

- one machine-readable JSON artifact containing source paths and hashes,
  coverage, per-layer rows, and pooled rows for every condition;
- one concise Markdown table with validation loss, `R_model`, and the seven
  exact-zero percentages in report order;
- an observation Markdown file stating the question, method, coverage, result,
  caveats, and source script;
- a verification record for checkpoint identities, validation-cache identity,
  environment, and output hashes.

No figure is required merely to fill the report table. If a figure is created,
save it as PDF and add the corresponding observation file and observation index
entry.

## Acceptance checks

Before using the new values in the report, require all of the following:

1. All ten final checkpoint model hashes match their stored inventories.
2. The validation cache hash matches the value above.
3. Each condition covers exactly 338 sequences and 692,224 input tokens and
   reports the 1,444-token tail.
4. The artifact contains exactly 42 per-layer rows and seven pooled rows per
   condition: seven sites across six layers.
5. Every row has zero nonfinite values and internally consistent integer counts.
6. Recomputed `h`, `m`, `q_post`, `k_post`, and `v` counts match the existing
   terminal diagnostics. Stop and investigate any mismatch before accepting
   `a` or `z`.
7. The reloaded-checkpoint validation loss matches the stored endpoint within a
   declared numerical tolerance, with the observed difference recorded.
8. The analysis states that exact-zero mass is an activation statistic and that
   `R_model` remains a separate logical-product opportunity metric.

This is a forward-only diagnostic over ten Pythia-14M checkpoints. Benchmark one
checkpoint locally before scheduling the full set, but no optimizer state,
backward pass, or training replay is required.

## Analysis 005 manuscript integration

On 30 August 2026, the user approved reporting Analysis 005 as a strong
descriptive result in `status-update.tex`. The selected uniform TEAL-style
post-hoc table now sits in Section 4.1 beside the GeLU/ReLU controls and:

- explicitly identifies `a`, `m`, `h`, and `z` as the thresholded sites and
  `q_post`, `k_post`, and `v` as unthresholded;
- selected low-degradation control endpoints from
  `observations/O001-r-model-vs-final-validation-loss.md`;
- paired loss deltas, which are intentionally omitted from the trained A4 and
  A4-OL1 endpoint table.

The complete 20-point control sweep and all integer-count evidence remain owned
by `analyses/005-2026-08-30-run004-controls-teal-posthoc/`. The report does not
promote a centralized finding, infer runtime speedup, or treat the trained and
post-hoc interventions as method-isolated equivalents.

## Future `a`/`z` table update

After the analysis passes verification, replace `n.m.` for `a` and `z` in
Tables 6, 8, and 9. Link the table caption or adjacent sentence to the new
analysis and observation. Preserve the original Run 004 and Run 009 diagnostic
files as the record of what was measured during those runs.

## Analysis 007 manuscript integration

On 30 August 2026, the user approved updating the status report with Analysis
007 as a very strong descriptive result. The report now:

- marks the five-condition paper-scale Run 012 A4-OL1 cohort complete;
- replaces the Analysis 005 combined frontier figure with Analysis 007's
  augmented frontier PDF and makes it the report's final presented element;
- reports A4 and A4-OL1 as paired absolute endpoints in the same layout as the
  Analysis 003 naive-L1/OL1 table, without delta columns;
- states that every matched `kappa <= 0.1` A4-OL1 endpoint improves both loss
  and measured `R_model` relative to A4, while `kappa=0.5` reverses in loss;
- drops the older Analysis 003 endpoint plot, the A7/A7-OL1 pilot subsection,
  and the progress/evidence-scope sections;
- preserves the one-seed, joint-intervention, logical-opportunity, and
  no-runtime-speedup caveats.

The report links Analysis 007 Observation O001 and leaves the analysis-owned
PDF and machine-readable evidence in its numbered analysis folder. The matched
moderate-threshold conclusion was subsequently consolidated as tentative
Finding F001, and the report now names that evidence status explicitly.

## Analysis 008 manuscript integration

On 31 August 2026, the user approved adding the five-condition Run 013 A7
full-pass result to the status report. The report now:

- marks A7 complete in the experiment-scope and analysis-status tables;
- reports the five A7 endpoints, including count-pooled exact-zero mass at the
  seven gated sites and ungated post-`W_o` `attention_output`;
- combines the A4/A4-OL1 narrative into the subsection that owns their table;
- gives the trained/post-hoc frontier its own final subsection and uses
  Analysis 008's A7-augmented PDF;
- records one effective cost and one per-condition scientific-process
  wall-clock range for each RunPod run.

The matched A4/A7 comparison was subsequently consolidated as tentative Finding
F002. That finding does not cover A7/A7-OL1. `R_model` remains logical-product
opportunity rather than measured runtime speedup.

## Analysis 008 A7-OL1 manuscript integration

On 31 August 2026, the user approved extending the status report through the
completed five-condition Run 014 A7-OL1 full pass. The report now:

- marks A7-OL1 complete in both experiment-scope registers;
- replaces the A7-only table with ten interleaved A7/A7-OL1 rows at matched
  `kappa`, reporting validation loss, count-reconciled `R_model`, and all eight
  measured exact-zero site masses;
- embeds Analysis 008's corrected single-panel frontier without its former A7
  detail inset and uses plain `A4-OL1` and `A7-OL1` legend labels;
- adds Run 014's `$14.1285` settled cost and 74--115 minute scientific-process
  range, bringing effective recorded spend to `$89.99`.

The Run 014 extension remains descriptive one-seed, one-scale evidence. No
A7/A7-OL1 finding or runtime-speedup claim is promoted; Analysis 008 retains
the machine-readable and count-pooled numerical provenance.
