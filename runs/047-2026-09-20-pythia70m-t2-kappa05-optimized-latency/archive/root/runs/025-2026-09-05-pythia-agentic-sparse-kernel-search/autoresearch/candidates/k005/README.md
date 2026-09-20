# K005: width-specialized fused Sakana descendant

K005 changes only the output decomposition of the qualified K001 fused exact
compaction kernel.  The exact-zero test, signed nonzero retention, ascending-K
compaction order, FP32 FMA, BF16 rounding, bias, model gates, and dense LM head
remain unchanged.

K001 assigns eight outputs per lane (256 outputs per row warp).  That leaves
half a warp's output lanes idle for Pythia-14M h/z (`N=128`) and scans each
input row repeatedly for wider projections.  K005 uses four outputs per lane
and eight row warps when `N=128`; otherwise it uses sixteen outputs per lane
(512 outputs per row warp).  This is an architecture-shape specialization, not
additional pruning or a changed R_model.

The candidate must independently pass CUDA primitive gates and full-model
development gates.  A measured speedup over K001 on the same checkpoint is the
fixed-R_model optimization test; a slowdown is retained as a failed search
step.
