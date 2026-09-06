import importlib.util
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
AUTORESEARCH = HERE.parents[1]


def test_phase4h_covers_the_complete_bounded_k007_space_once():
    spec = importlib.util.spec_from_file_location(
        "run025_k007_phase4h_test",
        AUTORESEARCH / "candidates/k007/candidate.py",
    )
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = (HERE / "phase4h-70m-k007-a4-high-geometry.sh").read_text()
    requested = re.findall(r"^run_check (m\S+)$", source, flags=re.MULTILINE)
    assert len(requested) == len(set(requested))
    assert set(requested) == set(candidate.CONFIGS)
    assert source.count("--condition 70m/a4-0p5") == 1
    assert source.count("--inputs 16 --passes 3 --seconds 600") == 1
