import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run025_portfolio_probe", HERE / "probe_kernel_portfolio.py"
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def model():
    mlp = SimpleNamespace(
        dense_h_to_4h=torch.nn.Linear(128, 512),
        dense_4h_to_h=torch.nn.Linear(512, 128),
    )
    attention = SimpleNamespace(
        query_key_value=torch.nn.Linear(128, 384),
        dense=torch.nn.Linear(128, 128),
    )
    return SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)])
    )


def test_modes_are_mutually_exclusive_and_sites_are_respected():
    instance = model()
    portfolio = P.PortfolioAdapter(instance, {"h", "z"})
    for mode in ("k001", "native", "k006"):
        portfolio.set_mode(mode)
        selected = portfolio.adapters.get(mode)
        for parent, name, original, _wrapped, site in portfolio.adapters["k001"].entries:
            actual = getattr(parent, name)
            if mode == "native" or site not in {"h", "z"}:
                assert actual is original
            else:
                matching = next(
                    row for row in selected.entries if row[0] is parent and row[1] == name
                )
                assert actual is matching[3]


def test_source_inventory_contains_both_inherited_kernels_and_dispatch():
    paths = [row["path"] for row in P.source_records()]
    for suffix in (
        "candidates/k001/kernel.cu",
        "candidates/k005/kernel.cu",
        "candidates/k006/candidate.py",
        "probe_kernel_portfolio.py",
    ):
        assert any(path.endswith(suffix) for path in paths)
