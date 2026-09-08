# Approved manuscript integration, 8 September 2026

The two source files here are byte-identical snapshots of the user-requested
`manuscript/draft/kernel-autoresearch.tex` and `kernel-implementation.md`.
They preserve this run's result-bearing manuscript addition without changing
the existing local-only policy for the rest of the draft. They are not a
standalone manuscript or new benchmark inputs.

To integrate, place the two files in `manuscript/draft/` and add
`\input{kernel-autoresearch}` after the setup/comparison text in
`experimental-study.tex`. The existing wrapper must provide `\Smodel`,
`graphicx` and the usual mathematical commands. Relative links in these
snapshots are interpreted at their original `manuscript/draft/` location.
The figure is referenced in its run-owned location, never copied here.

Evidence and limitations are in
`../../observations/06-implementation-and-claim-audit.md` and
`../../results/implementation-audit-001.json`. The handoff verifier
`../../20_verify_audit_handoff.py` records source/snapshot identities, test
receipts and checks on the compiled local reading copy in
`../../results/audit-publication-001.json`. The wrapper PDF is not a claim of
ICLR submission-template fit. No frozen kernel or benchmark data was changed.
