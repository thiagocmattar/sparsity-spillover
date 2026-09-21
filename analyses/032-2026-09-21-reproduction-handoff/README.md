# Lean paper reproduction handoff

User request: read manuscript/draft and create a gitignored handoff for a future
public repository. The user clarified that it must be lean: central methods,
configs, final kernels and paper-linked human/agent reproduction guides, without
copying the experiment or analysis history. Project license is left undecided.

Build: `.venv/Scripts/python.exe analyses/032-2026-09-21-reproduction-handoff/01_build.py`.
The exporter copies only an explicit dependency closure and compact evidence;
`package/` supplies the portable entry points and documentation. Output is
`/handoff/`, ignored by the source repository. No data preparation, full training,
GPU qualification, cloud allocation or manuscript edits are authorized/performed
by this packaging task.

The release is a refactored companion, not a claim of remeasured results. Core
scientific primitives and CUDA arithmetic retain source hashes; relocation,
generalized pressure-site checking and the reduced checkpoint cadence are
documented. Exact historical initialization tensors remain external by design.

See `verification.json` for isolated-copy checks and
`observations/001-handoff.md` for scope and limitations. Generated verification
scratch space is outside the handoff and is not committed. Existing Run048 work
is unrelated and remains untouched.
