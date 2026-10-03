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

## Approved design: 14M gate sign comparison

The latest request narrows the study to T4 with two pressure recipes, RunPod,
and two or at most three thresholds per intervention. The proposed P-o = Pall
interpretation (a,m,h,z pressure) was included in the design approved by the
user's “Toca o barco” on 3 October 2026.

The [concrete proposed design](../analyses/039-2026-10-03-gate-sign-and-tile-diagnosis/PROPOSED-14M-T4-DESIGN.md)
uses two thresholds, holds h/z gates fixed, and varies the sign rule at a/m.
Implementation belongs to [Run 057](../runs/057-2026-10-03-pythia14m-t4-gate-sign/README.md).
The user's subsequent explicit RunPod/AFK authorization covers launch,
monitoring, retrieval and analysis through figures/results within the recorded
responsible envelope.
