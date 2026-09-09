# Matched-intervention manuscript draft

Rewritten on 2026-09-04 at the user's request, using current completed evidence
and adversarial-review-v3. Read [main.pdf](main.pdf) for the full paper and
[rewrite-review.md](rewrite-review.md) for a concise review of the argument
and decisions. The active sources are main.tex and sections/.

## Scientific identity

The contribution is a matched study of gate placement and activation-pressure
optimization, centered on 14M and extended through selected 70M recipes.
R_model supplies a common operation-weighted evaluation axis. The complete
410M fixed-budget results and completed negative 70M runtime study delimit
the claims. The closest model-wide prior work, especially Q-Sparse and Spark,
is acknowledged directly.

## Build and evidence

Run `powershell -NoProfile -File manuscript/draft/01_build.ps1` from the repo
root. The builder regenerates Analysis 013 from existing artifacts, copies
its five PDFs and twelve generated table fragments, records input hashes,
and compiles the paper using the local official ICLR 2027 style.

Owning analysis:
[Analysis 013](../../analyses/013-2026-09-04-matched-intervention-manuscript/README.md).
It contains 59 main/appendix trained-checkpoint rows, 60 clipping rows, six
runtime sentinels, the separate three-rate screen, and per-figure observations.
This rewrite performs no training or checkpoint forward pass.

- [claim-ledger.md](claim-ledger.md): claim, evidence, and interpretation limits.
- [supplementary-results.md](supplementary-results.md): complete evidence inventory.
- [submission-readiness.md](submission-readiness.md): verification and remaining gaps.
- [adversarial-review-v3.md](adversarial-review-v3.md): the historical review
  used as editorial guidance; its runtime status predates completed Run 023.
- `revisions/before-evidence-rewrite/`: recoverable copy of the prior active
  TeX/PDF and associated draft notes.

The older narrative-review files and framing reviews are historical working
notes. The older top-level manuscript/introduction.tex and methodology.tex
remain historical/proposal sources. The active rewrite uses the operational
definitions and the evidence above. Previous unused figure/table copies may
remain locally; main.tex and the build-input hash manifest identify this bundle.

This draft remains under the repository's existing ignore rule. Analysis 013
and the manuscript evidence crosswalk are versioned separately. No source run,
training configuration, or finding registry is changed by this rewrite.
