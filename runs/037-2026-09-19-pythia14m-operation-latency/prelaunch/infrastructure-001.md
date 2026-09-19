# Launch and infrastructure record

The user explicitly approved launch and the diagnostic inventory on19 September
2026: "Approved. launch it". The USD2/90 aggregate GPU-minute envelope applies.

The live precreation catalog showed Secure RTX5090 stock in EU-RO-1/EUR-NO-1
at USD0.99/hour. Pod `ay9uq9m2nmdxmb`, `run037-operation-latency-001`, was created
at13:28:50.832UTC in EUR-NO-1 with the approved pinned image, one GPU,20GB
container and25GB persistent Pod volume. No network volume was attached or
created. Absolute compute stop deadline:14:58:50.832UTC; retrieval reserve
starts14:48:50.832UTC. Local guard PID40484 reported ARMED.

The93,603,840-byte input tar transferred at approximately0.267MB/s (about351s).
Remote SHA256 verification printed `run037-input-001.tar: OK`; its digest is
`2393815ba4afc7e22cfbee5aa52f1f7c886197f6afc99da94f32dae9b440ec86`.
The remote stop guard reported ARMED at13:36:05.090771UTC and consumed/unlinked
its private credential file. No credentials are retained in this record.

The initial launcher timed out waiting for SSH EOF after starting that guard:
the detached command's enclosing shell retained an output descriptor. It had
not started setup. The completed upload and running guard were verified before
`start_setup_002.py` launched a worker with all descriptors redirected. Setup
PID158 runs independently of the terminal and writes `runtime/setup-001.log`
and `runtime/setup-001.exit`. No kernel, checkpoint, validation input or numerical
bound changed. The failed launcher and corrected infrastructure helper remain
as distinct records.
