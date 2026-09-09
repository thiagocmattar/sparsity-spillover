# O002: architectural reach before training

Question: how much declared work can a site set reach before checkpoint values exist?
Method: symbolic d_f=4d specialization checked against the integer operation
inventory in src/sparsity_research/ceilings.py for all three configurations.
Coverage: T=2048, V=50304, (L,d)=(6,128),(6,512),(24,1024), plus analytic trends.
Legend/caption: h/mh/A4/A7 ratios use Td[L(12d+T+1)+V]; sites combine by union,
including V=0 => PV=0 => zero attention-projection input.
Result: h-only/A4/A7 percentages are 4.278/12.833/29.952 at 14M,
12.354/37.063/49.424 at 70M, 24.925/74.776/87.245 at 410M.
Caveats: full uncached workload only; natural zeros can lie outside selected
reach; no quality or runtime guarantee follows from these counts.
Source scripts: ../geometry.py and ../test_geometry.py. No new measurement.
