# Compact kernel implementation and validation appendix

User-approved editorial revision of Appendix E: execution mechanisms,
numerical validation/timing, controlled 14M execution, and 70M execution
control. No experiment, result, tolerance, or table entry changes.

The execution-rules and 14M control tables are retained byte-for-byte.
The 70M control's table body is unchanged; its caption replaces the reference
to the removed implementation-comparison table with "main-text latency sweep",
preserving the separate-session qualification. Automatic numbering changes
the retained former Table 13 to Table 12.

The hardware/software, validation coverage, numerical acceptance criteria,
timing protocol, and exclusions are unchanged. Fixed PyTorch Base references
remain 0.65544 ms (14M) and 1.61105 ms (70M). The 14M exact-logit/activation
agreement and 70M tolerance-only qualification remain explicit.

The 14M joint effect remains 82.7 microseconds / 12.9% (12.8--13.0% across
processes). The conditional h/z effects, their non-additivity, the 0.5614 ms
unmodified-kernel sensitivity (12.5%, 2.6 microsecond offset), and secondary
same-checkpoint PyTorch comparison (0.7018 ms, 20.4%, 57.9%) remain.
The 70M controls retain 1.128, 2.704, and 1.365 ms with 58.3% and 17.4%
interpretations and the inefficient-dense-reference/fusion/layout qualifications.

The initial-versus-optimized implementation table is no longer included.
Its source file and underlying evidence remain retained, and the main text
still reports transfer overhead. The development chronology is removed as
requested. Evidence remains in Analysis028 observation 001 and retained-controls,
Run037 observation 001, and Run048 observation 001; the original appendix
is preserved in Git history. No global experiment claims are changed.

A temporary full build has 18 pages. Appendix E occupies pages 17--18;
both execution-control tables and interpretations fit together on the final
page. Visual review confirms legibility and no clipped content. References
resolve, with no overfull boxes; one existing underfull-line warning remains
outside this appendix. Main.pdf was not replaced.
