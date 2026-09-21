# Lean paper reproduction handoff

Question: can the central manuscript methods/results be exposed as a small,
independent repository without weights, datasets and historical infrastructure?

Method: audit the current TeX, map result-bearing sections to retained data,
extract the shared mathematical primitives and final kernel dependency closure,
and provide portable condition, training, evaluation, clipping and timing CLIs.
The first broad archive-export approach was abandoned after the user's explicit
lean-scope clarification; its partial output was removed.

Coverage: 84 endpoint records (45/27/12), 14M site/layer counts and OL1 traces,
Base training curves, retained post-hoc sweeps, conditional 14M operation effects,
70M controls and separate-session references. Run048 is outside the cutoff.

Figures/captions: original publication PDFs are copied unchanged to handoff's
results/paper-figures. `scripts/reproduce.py` produces nine central PDF views from
the compact records. Its result/figure observations and caveats are in the
companion's results/README.md; no new measurement or finding is asserted.

Result: see verification.json for package counts/size, all checks and isolated
reproduction. The source bootstrap suite passed 242 tests. Final kernel assembly
and synthetic optimizer updates are CPU checks only.

Caveats: no exact initial-state tensors are distributed, no full training or
CUDA measurements were rerun, and historical raw archives are excluded. The
portable wrappers need full qualification on the target GPU. License remains
undecided. A minor manuscript presentation discrepancy is recorded in the release
status: the text's separate marker description outlives the later endpoint's
marker simplification; session metadata remains separate.

Source: `01_build.py`, `02_verify.py`, `package/`; evidence hashes in the exported
PROVENANCE.json and MANIFEST.json. No manuscript sources or scientific run inputs
were changed.
