"""Complete-validation FFN-only contribution probe for frozen policies."""

from probe_frozen_component import run


if __name__ == "__main__":
    run("ffn", __file__)
