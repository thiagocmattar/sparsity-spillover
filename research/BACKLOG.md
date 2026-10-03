# Research backlog

The human selects the research direction. This list records requested work;
it does not authorize implementation, launch or cloud spending.

## Deferred tile pressure and gates

Requested and explicitly deferred on 3 October 2026. Investigate pressure and
gating organized by activation tiles, with the aim of creating blocks that
hardware can skip during training or inference. Resume after the 14M comparison
of unilateral and bilateral gates at a/m.

The any-zero trigger and a group-magnitude trigger remain different candidate
methods. Tile axes, partition, kernel alignment, pressure definition, backward
rule, prefill versus decoding workload and training execution are undecided.
Retain row/tile occupancy in the preceding gate experiment where practical.

Evidence and limitations: [gate and tile diagnosis](../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/README.md).

## Active planning for the 14M gate sign comparison

The latest request narrows the study to T4 with two pressure recipes, RunPod,
and two or at most three thresholds per intervention. The second pressure
label was spoken as P-o and still needs its exact site set clarified.

The [concrete proposed design](../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/PROPOSED-14M-T4-DESIGN.md)
uses two thresholds, holds h/z gates fixed, and varies the sign rule at a/m.
Design confirmation and the later launch confirmation remain pending.
