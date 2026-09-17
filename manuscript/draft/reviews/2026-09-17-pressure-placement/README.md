# Targeted pressure-placement integration

This revision implements the author's updated `feedback-review-task.md` using
retained measurements. It starts from the restored draft after the rejected
review rewrite; the separate `review-alternative/` archive is unchanged.
The detailed [task audit](../../../../analyses/021-2026-09-10-training-results-figures/pressure-scope/TASK-AUDIT.md)
records all 30 requested items and the remaining evidence boundaries.

## Changes

- Threshold and pressure scope are independent, explicit design variables.
- Training coverage is 64 endpoints (40 at 14M, 12 at each larger size).
- Six main figures, two appendix PDFs and four tables incorporate the ten
  verified h-only endpoints. OL1 diagnostics use all 14,240 logged steps;
  activation summaries retain separate sites and layer detail.
- Five historical four-site h-only timings, ablations and counters extend
  the kernel cohort to 35. The main comparison uses absolute full K050
  latency and validation loss; original 30-point diagnostics retain their
  stated historical coverage.
- The approved reported-loss convention is preserved, with ordinary versus
  logical-pass losses disclosed and uniform-pass alternatives retained.
  Their maximum discrepancy is 0.000105806356 nats.

## Evidence boundaries

Five new seven-site h-only timings remain pending in separately prepared
Run 033. No runtime, training or cloud job was launched by this revision.
H-only scaling remains untested here; no 70M outcome is anticipated. The
missing T7/P4 condition and equal-tensor normalization limit attribution
to Q/K/V pressure specifically. All trained contrasts remain single-seed.

## Verification

The focused evidence, OL1 and cross-size tests pass (12 tests). The final
32-page PDF was built with pdflatex/BibTeX; references resolve and no overfull
boxes occur. Four underfull vertical boxes and one underfull horizontal box
remain. Figures and manuscript pages were rendered and visually inspected;
the final appendix text shares a page with its diagnostic figure.

All 27 figure copies, 19 measurement copies and four new table copies match
their recorded sources. ICLR style files and bibliography are unchanged.
The original supplementary `figure-data.json` is a historical snapshot:
its scientific contents match the current analysis, but its source metadata
differs. Its manifest and protocol now pin the exact source commit rather
than incorrectly claiming a current-file byte match. No numerical values
were changed to resolve that pre-existing provenance mismatch.

See [verification.json](verification.json) for output hashes and check details.
The user's task file, notes and unrelated run changes are excluded from the
commit. No push is requested.
