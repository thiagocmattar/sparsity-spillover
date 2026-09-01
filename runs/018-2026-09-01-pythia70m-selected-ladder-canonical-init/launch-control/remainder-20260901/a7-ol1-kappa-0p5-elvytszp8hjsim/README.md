# A7-OL1 kappa 0.5 retrieval

- Pod: `elvytszp8hjsim`
- Attempt: `012-20260901-155715-b6ac963f`
- Remote archive SHA-256: `da863221e5b9f84ace96ed36ebb39149f58e7df77986263a5d50934a33edbccd`
- The Pod is retained until the archive digest, per-file retrieval inventory, and local condition verifier all pass.
- Local archive SHA-256 matched the remote digest exactly.
- Initial extraction under this long launch-control path omitted a deep checkpoint metadata entry because of the Windows path-length limit. Re-extraction under short root `tmp/r18r/p012` verified all 30 inventoried files.
- The verified attempt was copied to the canonical artifact directory and `04_verify.py --condition a7-ol1-kappa-0p5` passed locally.
- Pod deletion returned HTTP 204; the subsequent Pod inventory confirmed it absent, and its exact local deletion guard was stopped.
