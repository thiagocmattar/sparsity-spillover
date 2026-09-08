# Operation-level interpretation

## Question

Does higher raw model-wide sparsity reflect learned zero rates, a larger reachable workload, or both?

## Method

Select A4-OL1 and A7-OL1 at kappa=0/.5 for each scale. Decompose R_model into each operation's zero-product count divided by the full model denominator. Also compute zero products divided by that operation's own products. These satisfy R_model = sum_o workload_share_o * zero_rate_o exactly.

## Coverage

Twelve trained endpoints, all layers and all 338 validation blocks per endpoint. Topology ceilings are analytic per-sequence integers scaled by 338 for measured denominator comparisons. Dense LM-head products receive no zero credit.

## Figure and caption

[Publication PDF](../figures/04-operation-accounting.pdf)

**Learned zero rates and architectural weighting.** Upper panels stack the six operation contributions to model-wide sparsity; stacks sum to each observed percentage. Short black lines mark the selected-site all-zero reach ceilings. Lower panels show each operation's own zero-product percentage, count-pooled over layers and validation inputs. A4/A7 labels here denote their OL1 recipes, with kappa shown below. Tiny nonzero values below .01% are marked <.01. The head remains dense in the declared accounting.

## Result

At high threshold, A4 nearly saturates its reachable linear work at all sizes. A7 adds high QK/PV zero-product rates: 97.2/98.7% at 14M, 81.8/86.9% at 70M, and 83.5/94.0% at 410M. Its raw model-wide opportunity also rises because the dense head's relative weight shrinks. Thus raw growth alone is not evidence of a uniform increase in learned sparsity.

## Caveats

A4 can receive natural zero credit outside its selected reach; its analytic ceiling is not a universal bound on observed R_model. Per-operation rates cannot be replaced by unweighted averages of site sparsities. QK credits union of Q/K zeros; PV is causally position-weighted and includes valid probability zeros. These counts are logical opportunities, not saved time.

## Source script and evidence

`plots.py:operations`; `evidence.py`; `01_build.py`; `tables/operation-counts.md`, `tables/architecture-counts.md`, `tables/normalization-audit.md`. Raw logical_products.json diagnostics from Runs 014/015/018/019.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).
