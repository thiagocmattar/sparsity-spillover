# Attempt 003 outcome

The inference-context correction passed A0, A1-H, both A4 sentinels, and all
A7 kappa=0 validation and timing stages. The attempt then stopped before
serializing A7 kappa=0 because the newly measured V-only causal count exceeded
the archived canonical P/V-union count. The invariant remained enabled; no
mixed-context coverage result was accepted. The worker recorded exit code 1
and `status.txt=failed:pythia70m_phase_benchmark` at
2026-09-04T17:42:20Z.

Forensics showed that the lower-bound pass used SDPA Flash while Run 018's
canonical logical-product protocol used eager attention. Small attention-output
differences changed later-layer q/v exact-zero positions. The complete
canonical counter nevertheless reproduced all archived H200 integers exactly
on this H100 NVL, ruling out a GPU-SKU explanation.
