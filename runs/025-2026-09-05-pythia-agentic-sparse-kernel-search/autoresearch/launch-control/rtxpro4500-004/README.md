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
