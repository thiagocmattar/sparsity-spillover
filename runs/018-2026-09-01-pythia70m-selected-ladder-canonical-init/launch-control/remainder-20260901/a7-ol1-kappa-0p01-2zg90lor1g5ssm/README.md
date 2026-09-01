# A7-OL1 kappa 0.01 retrieval

- Pod: `2zg90lor1g5ssm`
- Attempt: `009-20260901-155845-6d49e30d`
- Remote archive SHA-256: `274a0a90b983eb060a14499250a2260b5ba539a2abbe72a1e21451576d0a7ff9`
- The Pod is retained until the archive digest, per-file retrieval inventory, and local condition verifier all pass.
- Local archive SHA-256 matched the remote digest exactly.
- Initial extraction under this long launch-control path omitted a deep checkpoint metadata entry because of the Windows path-length limit. Re-extraction under short root `tmp/r18r/p009` verified all 29 inventoried files.
- The verified attempt was copied to the canonical artifact directory and `04_verify.py --condition a7-ol1-kappa-0p01` passed locally.
- Pod deletion returned HTTP 204; the subsequent Pod inventory confirmed it absent, and its exact local deletion guard was stopped.
