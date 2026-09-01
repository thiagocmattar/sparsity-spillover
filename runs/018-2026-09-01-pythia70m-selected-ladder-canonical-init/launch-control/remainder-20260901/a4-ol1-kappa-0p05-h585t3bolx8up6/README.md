# A4-OL1 kappa 0.05 retrieval

- Pod: `h585t3bolx8up6`
- Attempt: `005-20260901-161549-02c4d451`
- Remote archive SHA-256: `8129aa6845ea307312baf8a740297d57bc9a719b93cdc4c06d14ca373ff4fb68`
- The Pod is retained until the archive digest, per-file retrieval inventory, and local condition verifier all pass.
- Local archive SHA-256 matched the remote digest exactly; the archive is 846,899,200 bytes.
- Short-root extraction at `tmp/r18r/p005` verified all 29 inventoried files.
- The verified attempt was copied to the canonical artifact directory and `04_verify.py --condition a4-ol1-kappa-0p05` passed locally.
- Pod deletion returned HTTP 204; the subsequent Pod inventory confirmed zero Pods, the retained 100 GB Standard volume `9luykg5yc3` remained unchanged, and the exact local deletion guard was stopped.
