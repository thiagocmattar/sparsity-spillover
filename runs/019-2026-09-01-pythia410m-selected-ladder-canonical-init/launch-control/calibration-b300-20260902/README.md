# Run 019 fastest-candidate calibration launch record

This directory records the approved non-evidence compatibility work used to
select the fastest feasible GPU for the scientific Run 019 cohort. It does not
contain a scientific condition.

Live values captured before provisioning on 2026-09-02:

- candidate: `NVIDIA B300 SXM6 AC`, 288 GB;
- Community price: $6.94/GPU-hour;
- Secure price: $7.89/GPU-hour;
- catalog stock: Low;
- account balance: $416.2929129723;
- initial resource audit: zero Pods and zero serverless endpoints; the existing
  100 GB volume `9luykg5yc3` remained unattached;
- compatibility Pod guard: two hours, at most $13.88 Community or $15.78 Secure.

The first B300/B200 Pods were capacity or readiness probes only and were deleted
without staging scientific inputs. The Community H200 Pod in Croatia was also
deleted after slow local transfer. None reached a calibration boundary. The
pre-existing network volume remained unattached throughout.

Secure H200 Pod `wf9jk1mg5v206u` in `US-CA-2` was created at $4.59/GPU-hour
with a four-hour platform termination deadline and an independent local guard.
The exact pinned MiniPile caches were rebuilt from the pinned Hugging Face
revisions and accepted only after matching the four approved cache hashes. The
canonical initialization was transported in bounded pieces, reconstructed to
1,621,370,392 bytes, and accepted only after its original SHA-256 matched.
Remote setup then passed the seven-file hash gate and independently loaded the
initialization twice, reproducing parameter hash
`76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`.

Attempt 01 stopped before CUDA work because the original validator admitted
only the A40/A100-era calibration candidate list. Its exit code, timestamps,
log, execution context, and post-failure GPU inventory are retained under
`wf9jk1mg5v206u/attempt01/`. The later user-approved fastest-GPU expansion adds
`NVIDIA H200` to the exact non-evidence calibration candidates. Attempt 02 uses
a distinct artifact slug and may start only from the committed, tested repair.
