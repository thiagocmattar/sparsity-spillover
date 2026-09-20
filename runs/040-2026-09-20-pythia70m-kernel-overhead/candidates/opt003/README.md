# opt003: N512 h/z output tiles

Keep opt001's native a/m and attention replacements. Expand each h/z output
tile from128 to512 columns, using all eight existing warps for matrix
work. Each eight-row activation group is consequently inspected1
time(s), instead of four, across the512 output columns. This tests redundant
inspection and output scheduling; reduced inspection requests are not a claim
about physical DRAM traffic or measured speedup.

The16-row MMA instructions still contain eight real rows. Preserve the exact
short-row test, gates, K16 support masks, accumulation order, branch/residual
rounding, and total potential-MMA/scalar counter definitions. More columns
per block can increase register pressure or reduce parallelism; those are
measured risks, not assumed improvements. One policy applies to both checkpoints.

The T7/Ph initial trace measured about0.20ms in h/z. Qualify operators and both
checkpoints on the fixed16 training-development blocks. No optimization
validation evaluation or timing is used to choose this candidate.
