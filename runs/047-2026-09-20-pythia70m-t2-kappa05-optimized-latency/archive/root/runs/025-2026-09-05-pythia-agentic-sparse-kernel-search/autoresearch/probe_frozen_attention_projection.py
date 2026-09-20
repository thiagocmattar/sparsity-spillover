"""Complete-validation attention-projection contribution probe.

This covers only the QKV/output projection linears at sites ``a`` and ``z``.
The QK-score and probability-value matmuls remain dense SDPA.
"""

from probe_frozen_component import run


if __name__ == "__main__":
    run("attention_projection", __file__)
