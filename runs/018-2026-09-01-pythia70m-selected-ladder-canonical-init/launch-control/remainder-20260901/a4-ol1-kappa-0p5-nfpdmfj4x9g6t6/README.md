# A4-OL1 kappa 0.5 retrieval

- Pod: `nfpdmfj4x9g6t6`
- Attempt: `007-20260901-155848-8b7ee7e3`
- Remote archive SHA-256: `45f838c4b275db008ac30733c88df6fd607446fce3e926fd5e4a2a4a384a75ce`
- The Pod is retained until the archive digest, per-file retrieval inventory, and local condition verifier all pass.
- Local archive SHA-256 matched the remote digest exactly.
- Initial extraction under this long launch-control path omitted a deep checkpoint metadata entry because of the Windows path-length limit. Re-extraction under short root `tmp/r18r/p007` verified all 29 inventoried files.
- The verified attempt was copied to the canonical artifact directory and `04_verify.py --condition a4-ol1-kappa-0p5` passed locally.
- Pod deletion returned HTTP 204; the subsequent Pod inventory confirmed it absent, and its exact local deletion guard was stopped.
