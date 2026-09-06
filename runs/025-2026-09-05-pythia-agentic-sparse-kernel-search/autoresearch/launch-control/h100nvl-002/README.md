# H100 NVL attempt 002: frozen-source transfer repair

Status: **complete and closed. Pod `9djdgi71c2yt9a` ran the full 92-process
matrix, produced a phase exit of zero at 05:05 UTC, and was deleted after all
1,088 inventoried files and 93 artifact directories verified locally. Zero
pods and zero endpoints remained.**

Attempt 001 transferred a Git archive produced under Windows
`core.autocrlf=true`. `git archive` consequently exported the Markdown files
listed in the K010, K013, and K016 frozen source manifests with CRLF bytes.
Their immutable source checks correctly rejected 74 winner/component processes
before GPU evaluation. The 18 P0 processes did execute and are retained.

This infrastructure retry changes no scientific definition. It reruns the same
92-process Phase 16 matrix so P0 and winner measurements remain co-located on
one H100 NVL. It deliberately reuses the Phase 16 controller and its internal
`h100nvl-001` labels byte-for-byte; the enclosing retrieval and launch-control
identity is `h100nvl-002` and must be used to distinguish the attempts.

The only transfer repair is creation of the code archive with
`git -c core.autocrlf=false archive`. `verify_code_bundle.py` then checks every
source byte count and SHA-256 declared by all three frozen manifests from the
archive itself. Transfer is prohibited unless that verifier passes.
