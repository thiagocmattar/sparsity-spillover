# Secure RTX 4090 preflight attempt r1

Approved execution Step 3 ran on Secure Pod `qq6u7wif86vdyk` on 2026-08-29.
The shared cache was copied to disposable container storage in 14 seconds. The
normal Run 004 loader verified both cache hashes and reproduced schedule shape
`712 x 32 x 32` and schedule SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

The exact MB32/GAS32, sequence-length-2048 preflight then failed CUDA memory
fit for both sampled conditions. ReLU control had 20.00 GiB allocated and the
operation requested another 12.28 GiB with only 3.03 GiB free. ReLU lambda 1
also requested another 12.28 GiB and failed. No complete optimizer boundary,
scientific attempt, checkpoint, or ETC timing was produced.

The JSON and log files are byte-for-byte copies from persistent storage on the
Secure Pod. `SHA256SUMS` records the verified transfer identities.
