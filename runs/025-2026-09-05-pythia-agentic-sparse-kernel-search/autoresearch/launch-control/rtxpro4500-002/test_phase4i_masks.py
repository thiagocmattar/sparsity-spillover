import importlib.util
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
AUTORESEARCH = HERE.parents[1]


def test_phase4i_covers_complete_bounded_k008_space_once():
    spec = importlib.util.spec_from_file_location(
        "run025_k008_phase4i_test",
        AUTORESEARCH / "candidates/k008/candidate.py",
    )
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    source = (HERE / "phase4i-70m-k008-a7-high-mask-search.sh").read_text()
    requested = re.findall(r"^run_check (\S+)$", source, flags=re.MULTILINE)
    assert len(requested) == len(set(requested))
    assert set(requested) == set(candidate.MASKS)
    assert source.count("--condition 70m/a7-0p5") == 1
    assert source.count("--inputs 7 --passes 2 --seconds 600") == 1
