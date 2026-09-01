# A4-OL1 kappa 0.01 retrieval

- Pod: `1loj0oi8jrrr39`
- Attempt: `004-20260901-160446-a489068c`
- Remote archive SHA-256: `86fc69d935a0c859b9d445297ae173d54e0bf5db747ae7607b7e5637970b8999`
- The Pod is retained until the archive digest, per-file retrieval inventory, and local condition verifier all pass.
- Local archive SHA-256 matched the remote digest exactly.
- Initial extraction under this long launch-control path omitted a deep checkpoint metadata entry because of the Windows path-length limit. Re-extraction under short root `tmp/r18r/p004` verified all 29 inventoried files.
- The verified attempt was copied to the canonical artifact directory and `04_verify.py --condition a4-ol1-kappa-0p01` passed locally.
- Pod deletion returned HTTP 204; the subsequent Pod inventory confirmed it absent, and its exact local deletion guard was stopped.
