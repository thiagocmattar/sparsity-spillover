# Run048 observations

- [001: causal h/z latency control](001-causal-hz-latency.md): at fixed 14M
  T2/Ph, kappa=0.1, h/z exploitation saves 82.749 microseconds (12.899%) within
  the specialized stack. This is 57.867% of its same-checkpoint native latency
  gap. h dominates; z has a small conditional benefit and the effects do not add.
