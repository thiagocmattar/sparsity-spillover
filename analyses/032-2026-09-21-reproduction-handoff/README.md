# Paper reproduction handoff and submission supplement

User request: read manuscript/draft and create a gitignored handoff for a future
public repository. The user clarified that it must be lean: central methods,
configs, final kernels and paper-linked human/agent reproduction guides, without
copying the experiment or analysis history. Project license is left undecided.

Build: `.venv/Scripts/python.exe analyses/032-2026-09-21-reproduction-handoff/01_build.py`.
The exporter selects the final components and compact paper evidence;
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
scratch space is outside the handoff and is not committed. Original experiment run records remain unchanged.

After the user's cleanup review, `release_kernels.py` extracts only the final
component bodies and `package/kernels/install.py` assembles them directly.
Intermediate installers, candidate manifests, unused CUDA versions, original
run-numbered configs, historical reference documents and TeX are excluded.
The manuscript PDF is excluded because it is submitted separately. Only the five
paper topologies are exposed.
`release_results.py` replaces private run IDs with scientific condition names
and descriptive session labels while retaining measured values and identity
hashes. The detailed author-archive mapping lives here in `source-map.json`,
outside the handoff. See `observations/002-clean-public-layout.md`.


## Current submission supplement (2026-09-24)

The current user request is a ZIP supplement supportive of the revised paper,
with a maximum size of 100 MB. The manuscript is excluded at the user's request
so its writing can evolve independently. The exporter includes six figure
assets, all 95 conditions (45/11/27/12 at 14M/31M/70M/410M),
the 31M kernels/configuration, 36 cross-scale execution coordinates, 31M full
qualification/raw timing records, four Base trajectories, and the retained
same-checkpoint execution controls. The rejected absolute/relative companion
figure is excluded. Prior observations below describe earlier snapshots.

`release_supplement.py` extends the historical reductions; the first 84 endpoint
values remain unchanged. Reviewer guidance links the paper claims to code and
measurements, with three ready-to-read CSV tables. Kernel controls retain their
scientific distinctions: PyTorch Base is an across-recipe reference, native h,z
is a same-checkpoint substitution, and custom skips-off has its own denominator.

Build, verify in isolation, and package from the repository root:

```powershell
.venv/Scripts/python.exe analyses/032-2026-09-21-reproduction-handoff/01_build.py
.venv/Scripts/python.exe analyses/032-2026-09-21-reproduction-handoff/02_verify.py
.venv/Scripts/python.exe analyses/032-2026-09-21-reproduction-handoff/03_zip.py
```

The ZIP builder requires a passing verification of the current manifest, writes
`/suplementary.zip`, enforces the conservative decimal limit of 100,000,000 bytes,
checks CRCs and every extracted file hash, and runs result reconstruction from
the extracted ZIP. It includes only manifest-owned files plus the manifest.
Both `/handoff/` and `/suplementary.zip` are ignored; tracked export sources and
verification records reproduce the delivery. Rebuilding after source edits
requires repeating verification. `source_commit` identifies the base checkout;
per-file hashes identify the exported working-tree snapshot.

See `observations/004-separate-paper.md`, `verification.json` and
`zip-verification.json` for the tested snapshot and its remaining limits.
