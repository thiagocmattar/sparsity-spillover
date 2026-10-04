# Tile gates aligned with the retained GPU implementations

Status: source audit complete. This analysis performed no GPU job, experiment
or benchmark. The subsequently approved gate is implemented separately in
[Run060](../../runs/060-2026-10-04-pythia14m-temporal-corner-tile/README.md), pending launch.

The user requests a tile structure whose zeros can prevent weight loads and
whose definition can carry across Pythia14M/31M/70M and other model families.
The next training comparison remains 14M T2/Ph at kappa=0/.1, measuring quality,
activation zeros and empty tiles before any inference-speed experiment.

The original audit recommends fixed contiguous groups of16 input features per
token (1x16 gate), with independent hardware execution row groups. Its source
observation remains unchanged. The user subsequently selects a stronger causal
restriction: decide a shared mask at the first token and enforce it on all
following tokens in the same fixed block. The approved8x16 gate examines only
the first feature of the first token: corner <= kappa rejects the entire tile.
An admitted tile retains its original values. See the backlog for this refinement.

Both use K16 support compatible with the inspected consumers. Neither is a
measured optimal gate or proof of transferable speedup.

See [the observation](observations/001-tile-geometry-and-weight-loads.md) for
source locations, causal semantics, portability and evidence limits; the
[source identities](data/source-identity.json) preserve inspected file hashes.
The [backlog](../../research/BACKLOG.md) retains the approved training recipe,
inclusive single-corner trigger, unchanged Ph, backward rule and evidence limits.
