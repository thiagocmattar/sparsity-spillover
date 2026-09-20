# 001: Completing the 14M T2/Ph manuscript grid

**Question.** How does the verified kappa=0.5 endpoint complete the
manuscript's T2/Ph quality, sparsity and latency curve?

**Method and coverage.** Join Run044's verified checkpoint to the previous
40-point Figure 1, preserving every historical row exactly. Recover the four
existing Run041 T2 rows also missing from the complete-results table, and
append all five to the retained 74-row full dataset. Gates are one-sided at
h,z, with OL1 at h only, lambda=1 and trust budget=1. The new model matches
initialization, seeds, data order, 712 updates and 1.493B input tokens.
Validation covers all 338 complete blocks from 500 documents; 1,444 tail
tokens are excluded. Model-wide sparsity pools integer product counts and
includes the dense LM-head denominator.

**Result.** Run044 has ordinary final-checkpoint loss 5.5362546204,
339,700,589,476 zero-operand products out of 6,363,055,915,008 model products,
and therefore 5.3386390755% model-wide sparsity. Exact-zero activation
fractions at h/z are 99.8408735%/99.8409444%; these are different estimands
from model-wide product opportunity. The T2 analytic ceiling is 5.3471468%.

K050 takes 0.5063232172 ms, with a paired geometric-mean speedup of
1.4189309536 against native CUDA graphs of the same checkpoint
(0.7184376854 ms). RTX5090 BF16, B1/T2048, full 50,304 logits, three fresh
processes and 1,344 timing pairs match the inherited protocol. All processes
qualify over the full validation set with zero observed logit or loss error.
BF16 qualification loss 5.5384558948 is retained separately from training
validation loss. This is a same-checkpoint speedup, not a fixed-base-model
speedup.

Figure 1 now contains 41 trained endpoints and the same 20 Base/ReLU clipping
records. The complete data contain 79 endpoints: 45 at 14M, 22 at 70M, 12 at 410M.
The difference at 14M is the four historical naive-L1 ablations excluded from
Figure 1 by the author's earlier selection. All previous 40 main-figure rows,
74 full-data rows and all clipping coordinates are unchanged.

**Legend and caption.** The ten-recipe layout, colors, line styles, bounds,
clipping-only-in-panel-(a) policy and ceiling guides are retained. T2/Ph now
connects kappa=0,.01,.05,.1,.5. The manuscript Figure 1 caption changes only
the cohort count and that grid, with source comments pointing here.
The complete 14M table adds T2/Ph alongside T4/P0 and T7/P0 for the same five
kappas. Its caption names the one-sided h,z gates, h-only OL1 and lambda=1.

**Caveats.** One seed, one model size, one data pass for this addition.
Thresholding and pressure are a joint intervention. Logical-product
opportunity is not a runtime speedup. Runs029/033/041/044 use separate
physical GPU/host sessions, so small cross-session latency differences are
descriptive. No T2/Ph post-hoc clipping was measured or inferred. The old
70M/410M cohorts, pressure contrasts and kernel ablations are unchanged.

**Source scripts and evidence.** 01_collect.py verifies the raw Run044
training and three timing processes, 02_plot.py retains the previous
Figure 1 rendering body, 03_table.py generates the compact table, and
04_verify.py independently checks all 79 final losses, product counts
and checkpoint identities, membership preservation and manuscript copies.
Exact source hashes, timing-session identities and coordinates are in
the figure-data.json and full-trained-results.json files under data/.
Run044's [source observation](../../../runs/044-2026-09-20-pythia14m-hz-h-only-ol1-kappa05/observations/001-final-results.md)
retains the training and latency qualification record.
