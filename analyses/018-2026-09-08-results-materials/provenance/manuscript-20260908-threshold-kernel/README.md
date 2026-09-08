# Fixed thresholds and specialized kernels - manuscript snapshot

This analysis-owned snapshot retains the user-authorized argument revision
while `manuscript/draft/` remains local-only. It supersedes the reading copy
in `../manuscript-20260908-polished/` without changing that earlier record.
[O015](../../observations/O015-threshold-kernel-argument.md) records the scope.

[main.pdf](main.pdf) is the verified 27-page reading copy. The methodology
motivates fixed symmetric thresholds from Q-Sparse and Spark; the kernel
subsection develops specialization, agent-assisted search, measured gains and
attention-skipping limits. Only `methodology.tex`, `kernel-autoresearch.tex`
and `results-appendix.tex` change among the twelve TeX sources. Introduction,
bibliography, eleven figure assets and seven numerical tables are unchanged.
Ten figures are embedded; overview v2 remains an alternative.

The [argument and review record](reviews/2026-09-08-kernel-argument/revision-log.md)
links primary-source and retained-evidence audits, independent reviews and the
final verification. `SNAPSHOT.json` maps each copied file to its live source,
byte count and SHA-256; `.gitattributes` preserves those bytes.

## Rebuild

All TeX build inputs are local to this snapshot. Run from this directory:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`CLEAN-BUILD.json` records an isolated rebuild and rendered-page comparison.
PDF metadata can differ while the pages match. The wrapper is a reading copy;
ICLR submission-template fitting is outside this revision.

## Evidence

The supplementary README, protocol and copy manifest are included. Fourteen
large raw measurement copies are not duplicated; their repository-relative
source paths and SHA-256 hashes in `supplementary-data/SOURCES.json` reproduce
the live supplementary directory. Thirty protocol-source hashes are verified.
The older argument map and plans are retained as historical context; the new
revision log states the current additions and evidence limits.

No experiment, benchmark, numerical table generation or figure regeneration
was performed. The fixed-threshold design is motivated by operation choice,
not a measured speed comparison against Q-Sparse or Spark. Kernel gains retain
their declared model, shape, hardware and numerical-qualification scope.
