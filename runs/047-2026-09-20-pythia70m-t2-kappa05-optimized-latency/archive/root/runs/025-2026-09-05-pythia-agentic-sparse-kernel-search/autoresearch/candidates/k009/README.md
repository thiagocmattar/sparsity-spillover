# K009: frozen 70M final-two-layer K001 policy

K009 freezes the unique passing/speed-winning policy from the complete K008
12-mask development search: use the unchanged K001 fused compaction kernel in
transformer layers 4 and 5, and native linears in layers 0 through 3.  Site
selection still follows the declared intervention topology.  No runtime
density threshold, checkpoint identity, or held-out result affects dispatch.

The frozen candidate must pass all 16 development inputs for all six 70M
endpoints before it can be evaluated on untouched interior kappas or complete
validation.
