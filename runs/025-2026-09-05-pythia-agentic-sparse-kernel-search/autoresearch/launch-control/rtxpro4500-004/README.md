# RTX PRO 4500 attempt 004: fixed-R_model pairs

This attempt independently replicates the baseline-to-searched-kernel contrast
at a fixed checkpoint and therefore a fixed canonical `R_model`. It changes no
model weights, data, topology, gate, or logical-opportunity definition.

## Scope

- hardware: one secure-cloud NVIDIA RTX PRO 4500 Blackwell in `EUR-IS-1`;
- persistent inputs/runtime: the verified Run 025 network volume;
- conditions: A4-OL1 and A7-OL1 at kappa 0.5 for 14M, 70M, and 410M;
- implementations: P0 versus K013 (14M), K009 versus K016 (70M), and flagged
  z-only K004 versus layer-restricted z-only K010 (410M);
- repetitions: three separate eager-only processes per implementation and
  condition, 36 processes total;
- timing: 16 fixed input clusters, five paired passes, 80 paired samples;
- validation: all 338 complete MiniPile blocks (692,224 input tokens), with the
  1,444-token tail reported;
- ordering: baseline then optimized on repeats 1 and 3, optimized then baseline
  on repeat 2.

The matched comparison is the ratio of each condition's optimized fresh-process
median to its baseline fresh-process median. Candidate correctness and complete
validation are gates; failures remain part of the evidence rather than being
silently retried.

## Cost guard

Phase 15 observed per-process means of 41.6 s (14M), 42.9 s (70M), and 58.8 s
(410M), predicting about 29 minutes for this matrix before startup overhead.
The live secure-cloud quote on 2026-09-06 was $0.72/GPU-hour with high
availability. A two-hour automatic deletion deadline caps GPU exposure at
$1.44; the expected cost is roughly $0.45 including provisioning and closeout.

Only a small LF-preserving code bundle is transferred. Checkpoints, MiniPile
token caches, compiled extensions, and the Python environment remain on the
attached verified network volume.

## Execution and results

Phase 17 completed all 36 prespecified processes. Every candidate passed the
complete 338-block validation gate. The median optimized-over-baseline ratios
across three fresh process pairs are:

| Size | Condition | Fixed `R_model` | Implementations | Median ratio | Wins |
| --- | --- | ---: | --- | ---: | ---: |
| 14M | A4-OL1, `kappa=0.5` | 12.713% | P0 -> K013 | 0.9370x | 0/3 |
| 14M | A7-OL1, `kappa=0.5` | 27.483% | P0 -> K013 | 0.9477x | 0/3 |
| 70M | A4-OL1, `kappa=0.5` | 35.596% | K009 -> K016 | 1.0132x | 3/3 |
| 70M | A7-OL1, `kappa=0.5` | 40.602% | K009 -> K016 | 1.0075x | 2/3 |
| 410M | A4-OL1, `kappa=0.5` | 71.591% | K004 -> K010 | 1.0063x | 3/3 |
| 410M | A7-OL1, `kappa=0.5` | 80.616% | K004 -> K010 | 1.0065x | 3/3 |

Because K013 did not beat P0 at 14M, Phase 18 was added adaptively and is
labeled as such rather than presented as preregistered confirmation. It ran 12
additional processes comparing the direct P0-to-K001 optimization on the same
two checkpoints. All passed complete validation. Median ratios are 1.0202x
(2/3 wins) for A4-OL1 and 1.0102x (3/3 wins) for A7-OL1. Across the resulting
primary six-condition comparison, all three architecture medians exceed one
and 16/18 process pairs favor the optimized implementation.

## Closeout

The retrieved archive contains exactly 50 process/control directories: 36
Phase-17 processes, 12 Phase-18 processes, and two controller directories. It
is 731,849 bytes with SHA-256
`0753655f73ccd2bd8586265c1558641949fc4b6b112684570b891f3b00a815d8`.
Both local archive verifiers return zero.

The first closeout command reached the expected post-deletion 404 but
PowerShell promoted it to a terminating error before writing the summary. That
infrastructure-only event is preserved in
[`closeout-attempt-001-failure.json`](closeout-attempt-001-failure.json) and
reconciled in [`closeout-complete.json`](closeout-complete.json). A separate
RunPod MCP inventory confirmed zero Pods and zero endpoints. The local
two-hour deletion guard was stopped after Pod absence was verified.

Direct Pod lifetime was approximately 48.9 minutes at $0.72/GPU-hour, implying
about $0.59 GPU cost before billing lag. Because RunPod exposes reliable
account-window billing rather than per-attempt attribution here, the
authoritative conservative totals are stored in
[`billing-closeout.json`](billing-closeout.json).
