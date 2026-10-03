# Gate sign and tile diagnosis

**State:** completed audit of retained evidence, requested on 3 October 2026.

The user's hypothesis is that unilateral gates after LayerNorm made `a/m`
less useful than `h/z`, and that gates and pressure organized by tiles could
produce more useful sparsity for hardware. Existing evidence supports investigating
both questions. It establishes a quality cost from jointly adding the studied
`a/m` gates and a shortage of empty tiles in one trained checkpoint. It does
not establish that bilateral gates solve either problem.

The [diagnostic observation](observations/001-gates-and-tiles.md) distinguishes
measured results, mechanistic hypotheses and candidate next comparisons.
[Machine-readable evidence](data/diagnosis.json) preserves integer counts,
checkpoint keys, units, quality differences and source hashes.

## Question and manuscript relationship

This audit connects the submitted paper's targeted versus broad thresholding
comparison and its training–inference co-design discussion to the proposed
next steps. It uses the operational gates in `research/METHODS.md` and the
actual implementation in `src/sparsity_research/{sites,pythia}.py`.
It creates no new trained intervention, result-bearing TeX or promoted finding.

## Source cohort

- Analysis031's retained quality records: T2/Ph (`h,z`) versus T4/Ph
  (`a,m,h,z`), at all five existing thresholds, for 14M and 70M.
  These keep pressure at `h`, avoiding the separate change in pressure scope.
- Run031's complete baseline 14M signed `m` histogram, pooled over six layers.
- Run039's BF16 operand and tile counters for 14M T7/Pall at kappa 0.5.
- Run037's conditional site execution effects, retained through Analysis031.
- Run056's qualified h/z tile versus short-row factorial at that same trained
  recipe, in a separate session.
- Root training configuration declarations and gate/hook implementation.

All evaluated source cohorts use the 338 complete 2,048-token validation blocks
from 500 MiniPile documents, 692,224 input tokens, with 1,444 tail tokens excluded.
Signed baseline distributions use FP16 autocast; execution diagnostics use BF16.
Numbers across these distinct checkpoints and precision paths are not pooled.

## Method and verification

`01_collect.py` reads existing artifacts only and writes within this analysis.
It verifies the retained baseline histogram hash, independently pools six `m`
layer histograms, checks signed partitions and integer tile fractions, reconciles
the quality collector's five historical source references, verifies all 60 sources
of Run056's reduction, and reconstructs its tile-only contrast. Three historical
source hashes match. Analysis024's operation table JSON has changed hash, but
all six measured rows are identical; their differences and process-extrema spans
are independently reconstructed from the unchanged Run037 reduction. The old
standalone manuscript table path is absent. Neither discrepancy is silently
treated as a successful hash check; both are recorded in the audit JSON.
The ten quality pairs preserve source checkpoint keys. For 14M, it also checks
initialization, realized data schedule, update count, training tokens, full
coverage and the percentage conversion from logical-product integers.
The compact 70M quality records omit those training identity fields, so this
audit imports their reported results without independently rechecking that
matching from raw training manifests. No cross-session latency comparison is
computed from the old 70M port timings in the quality export.

Reproduce from the repository root:

```text
.venv/Scripts/python.exe analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/01_collect.py
.venv/Scripts/python.exe analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/01_collect.py --check
```

Both commands passed locally. The saved audit verifies 104 current source
hashes, ten quality pairs, six pooled m layers, integer tile ratios and the
15 retained Run056 timing processes. Separate arithmetic checks confirmed
the positive loss differences and zero empty m tiles. These checks audit
existing measurements; they do not qualify a new intervention or kernel.

## Interpretation limits and next steps

The root-config scan is bounded to `runs/*/config.yaml`; it is not an exhaustive
audit of every executed attempt or archived candidate. The studied paper recipes
use unilateral `a/m`; their bilateral gates are at Q/K/V. A direct trained
unilateral-versus-bilateral comparison at `a/m` is absent from this evidence.
Post-gate histograms cannot recover removed negative values in trained models.

Candidate diagnostic and training comparisons are specified as questions in the
observation, not as approved runs. Their precise source cohort, intervention,
budget and execution envelope still require the repository's design and launch
confirmations. No new model pass, benchmark, training or cloud action occurs.

The compact current-status section of `research/INDEX.md` ends at Run053, while
completed Run054–056 and Analysis037–038 are present. This audit consulted those
newer records directly; the central index was not revised.
