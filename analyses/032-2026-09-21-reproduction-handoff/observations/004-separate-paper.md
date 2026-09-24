# O004 - Supplement distributed separately from the paper

## Request and change

The user requested omission of the manuscript because the paper is submitted
separately and its writing is still evolving, and requested the exact archive
name `suplementary.zip`.

Removed the manuscript export and the obsolete manifest-owned `handoff/main.pdf`.
Updated reviewer instructions, paper references and checks to require its absence.
The package retains the six supporting figure assets, code and measured evidence;
figure/table numbers are explicitly described as belonging to the evidence snapshot.
The archive and its internal top-level directory are named `suplementary`.
The former root `handoff.zip` was renamed and rebuilt, so it cannot be mistaken
for the current submission archive. The working export directory remains `/handoff/`.
The manuscript sources and paper PDF were not edited.

## Verification

- Isolated package suite: 66 passed in 5.63s.
- Full numerical reconstruction and the six figure-asset identity checks pass.
- ZIP CRCs, every extracted SHA-256, and result reconstruction after extraction pass.
- No manuscript PDF or TeX is present; the only six PDFs are supporting figures.
- No Markdown file in the release links to the removed manuscript.
- `/suplementary.zip`: 2,565,623 bytes (2.565623 MB),
  140 files, below the 100,000,000-byte limit.
- Archive SHA-256: `187c980862b0bfbeb9657979ce237bd4e6a9b1e2b801ad849e5e752f77252793`.

## Scope and provenance

This changes packaging and documentation only. Results, kernels and scientific
claims are unchanged. O003 describes the previous archive that included the paper.
The export, isolated verification and ZIP sources are `01_build.py`, `02_verify.py`
and `03_zip.py`; their latest checks are in `verification.json` and
`zip-verification.json`. No new training, GPU work or upload was performed.
