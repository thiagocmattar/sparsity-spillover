# Run 018 H200 timing preflight

This was an approved, non-evidence timing preflight. It did not launch any of
the 12 scientific conditions.

Pod `cnekba7p07zplh` was created on one Secure `NVIDIA H200` in `US-NC-1` at
`$4.59/hour`. An independent deletion guard was armed for
`2026-09-01T13:18:28.335Z`, 1.5 hours after Pod creation.

Attempt 1 stopped before model load because the transfer inventory omitted the
ignored cache `metadata.json` files. The binary cache itself had the declared
byte count and SHA-256. This was an infrastructure failure with zero scientific
boundaries. Its records are under `attempt-001-missing-cache-metadata/`.

The two local metadata files were transferred and hash-verified without changing
the source, config, initialization, cache binaries, or preflight code. Attempt 2
then passed under the same guard. Its exact result is
`attempt-002-passed/remote-preflight.json` (SHA-256
`8df80426c3a8e1c0646a87429f879b2c1d37eedbb6a51fd0ae6e7df4e45570f9`).

The five measured A0 boundaries had a 4.542-second median and about 461,762
input tokens/second. The five A7-OL1 boundaries had a 7.798-second median and
about 268,929 input tokens/second. All ten boundaries were finite, none skipped,
and peak reserved VRAM was 10.225 GiB. The preflight also completed all 338
validation blocks, activation diagnostics, logical diagnostics including
`R_model`, and checkpoint serialization. Its validation loss and `R_model` are
timing-probe outputs after only five updates, not scientific results.

The measured condition projections are 55.32 minutes for each control, including
the conservative TEAL proxy, and 92.77 minutes for each OL1 condition. A4 uses
the slower measured A7 boundary as a conservative proxy. Both configured
parallel waves therefore project to 92.77 minutes after their workers start.

The first complete transfer/setup reached preflight in 535.665 seconds. Including
the manifest omission and retry, the passed attempt began 939.665 seconds after
Pod creation. Applying the approved formula to that worse observed overhead and
rounding upward gives a 1.65-hour guard for A0/A1-H and a 2.6-hour guard for each
A4/A7 OL1 worker. The additive approval-pending launch envelope is in
`../../prelaunch/measured-scientific-launch-plan.json`.

The retrieved result hash matched the remote inventory before deletion. The Pod
was deleted, the independent guard process was stopped, zero Pods remain, and
the pre-existing 100 GB volume `9luykg5yc3` is unchanged. The observed balance
decrease was `$1.3664052186`; billing remains provisional until RunPod's posted
intervals settle. Continuing spend returned to `$0.01/hour` for that retained
volume.
