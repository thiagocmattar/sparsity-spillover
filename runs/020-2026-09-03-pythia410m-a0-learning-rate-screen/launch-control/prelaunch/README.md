# Run 020 prelaunch controls

These scripts implement the approved two-worker Run 020 payload, preflight,
detached training, per-attempt verification, and retrieval packaging. They do
not create, stop, or delete a Pod and are not launch authorization.

The input payload contains only the public MiniPile token caches and Run 019's
hash-pinned random model/RNG initialization artifacts. It contains no
credentials. `verify_worker.sh` hashes every input, checks the pinned runtime,
requires an approved GPU with at least 75 GiB visible memory and flash
attention, and performs two strict CPU loads plus parameter/RNG identity checks.

The controller supplies the committed source bundle SHA-256, input payload
SHA-256, expected Git commit, condition ID, and generated attempt ID. Training
must be started under `nohup` or an equivalent detached supervisor; all state
is written beneath `/workspace/run020-control` and the run's `artifacts/`
directory. `finalize_worker.sh` refuses to package an unverified attempt.

The controller retrieves and verifies both terminal attempts before running
the training-only selection locally. It then deletes any Pod that cannot be
used for the selected TEAL evaluation. If a new arm wins, its signed-off
selection record is copied to that still-running worker and only that worker
runs `06_teal_posthoc.py`; this prevents a third Pod and checkpoint re-upload.

Immediately after each Pod is created, the controller starts
`guard_runpod_deadline.ps1` as an independent local process with the exact Pod
ID, the SKU-specific UTC deadline from `config.yaml`, the verified RunPod CLI
2.12.0 binary, and a run-local log path. The guard sleeps in bounded intervals
and retries exact-ID deletion after the deadline. Normal completion still
retrieves and verifies artifacts and then deletes the Pod early; the guard is
only the automatic billing backstop.
