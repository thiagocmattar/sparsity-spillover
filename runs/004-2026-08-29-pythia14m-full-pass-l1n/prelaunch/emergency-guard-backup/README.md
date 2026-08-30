# Lambda-1 guard-risk safety copy

The A100 PCIe worker was created with a three-hour termination guard ending at
approximately `2026-08-30T00:02:05Z`. Corrected live wall-clock timing showed
that final diagnostics and retrieval could cross that deadline. This directory
therefore receives immutable recovery checkpoints before the guard while the
scientific process remains untouched. These files are a safety copy, not the
final authoritative artifact retrieval.

Both optimizer-bearing recovery checkpoints were copied and hash-verified
against their immutable remote files:

- `step_000256`: five files, approximately 163 MB;
- `step_000512`: five files, approximately 163 MB.

For each checkpoint, SHA-256 matched for `model.safetensors`,
`training_state.pt`, `config.json`, `generation_config.json`, and
`checkpoint_metadata.json`.
