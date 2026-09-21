# Verification and limits

The companion packages the supplied manuscript and its measured endpoints.
Original author files and experiment records were not rewritten.

Verified locally:

- Package file hashes and absence of datasets, model weights, binaries and
  credential-like strings in the export.
- All 84 condition identifiers, gate/pressure assignments, 712-update schedule
  identities and pooled endpoint fractions; 15 analytic architecture/topology
  ceilings; the recorded 17.4% conditional 70M h,z effect.
- Paper-topology mathematical/hook/aggregation tests, actual synthetic CPU
  updates for Base/L1/OL1-HZ/OL1-seven-site, checkpoint round trips, final kernel
  assembly and extension build-path resolution on CPU.
- Selected CUDA token streams match the measured sources after identifier and
  comment cleanup, with only the selected vocabulary tile retained. The included
  manuscript PDF matches the supplied file byte-for-byte.
- Offline table/figure reconstruction in a separate copy, outside the source
  repository layout. No access to historical run/analysis folders is required.
- The original repository's full 242-test bootstrap suite remained green.

Not performed for this release: full pretraining, CUDA compilation/execution,
fresh latency qualification, or a clean network installation of all pinned
packages. The final assembly imports its components directly; historical
candidate chains and the superseded 70M port are omitted. Comparison measurements still
allow reading the paper's port-versus-final result. CPU assembly and preservation
of the selected CUDA arithmetic do not prove GPU equivalence.
The supplied portable harness must pass the included full-validation tests on
the target device before its new timings can be reported.

Exact historical initialization bytes are not bundled. No public model/initial
state download URL has been provided. Historical RNG/backend differences are
known, so a fresh random draw has an explicit CLI flag and distinct provenance.
Even matching initial weights does not guarantee bitwise training trajectories
across accelerators and runtime builds.

`results/` contains compact published-measurement reductions, diagnostic traces,
integer counts and source hashes. It does not contain every raw training event,
every timing sample, or every rejected kernel candidate from the author archive.
Fresh runs do write their full boundary records, qualification rows and raw
timings. Canonical published loss is ordinary final validation loss; some older
diagnostic sources also retain logical-pass loss and should not be silently mixed.

Paper notes: the source appendix still describes the later 70M endpoint as
visually marked separately, while the latest author figure uses ordinary recipe
markers. This is a presentation discrepancy; the release preserves its separate
session identity and never pools or rescales those timings. The narrow T2 gate is HZ=(h,z).

Project license remains undecided at the author's request. Author identities,
final citation metadata and publication URL are intentionally not invented.
