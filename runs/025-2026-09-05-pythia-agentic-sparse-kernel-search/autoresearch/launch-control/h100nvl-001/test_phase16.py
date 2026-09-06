from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
PHASE = (HERE / "phase16-h100-frozen-transfer.sh").read_text(encoding="utf-8")


def _array(name: str) -> list[str]:
    match = re.search(rf"{name}=\(\n(.*?)\n\)", PHASE, flags=re.DOTALL)
    assert match
    return re.findall(r'^\s+"([^"]+)"$', match.group(1), flags=re.MULTILINE)


def test_exact_endpoint_sentinel_matrix():
    assert "h100sxm" not in PHASE
    rows = _array("sentinels")
    assert len(rows) == 18
    expected_suffixes = {"a0", "a1h", "a4-0", "a4-0p5", "a7-0", "a7-0p5"}
    for size in ("14m", "70m", "410m"):
        actual = {
            row.split()[0].split("/", 1)[1]
            for row in rows
            if row.startswith(size + "/")
        }
        assert actual == expected_suffixes
    assert not any(re.search(r"0p0?1|0p05|0p1(?:\s|$)", row) for row in rows)


def test_frozen_policy_and_site_mapping():
    rows = [row.split() for row in _array("sentinels")]
    assert {row[5] for row in rows if row[0].startswith("14m/")} == {"k013"}
    assert {row[5] for row in rows if row[0].startswith("70m/")} == {"k016"}
    assert {row[5] for row in rows if row[0].startswith("410m/")} == {"k010"}
    assert all(row[2] == "all" for row in rows if row[0].endswith("/a0"))
    assert all(row[2] == "h" for row in rows if row[0].endswith("/a1h"))
    assert all(row[6] == "active" for row in rows if row[0].startswith("410m/"))


def test_process_and_measurement_scope():
    assert "for run025_repeat in 1 2 3" in PHASE
    assert "--inputs 16 --passes 5" in PHASE
    assert "--full-validation" in PHASE
    assert "eager graph" in PHASE
    components = _array("components")
    component_processes = len(components) * 2 - 2
    assert 18 + 18 * 3 + component_processes == 92


def test_components_do_not_mislabel_qk_or_pv():
    assert "projection linears" in PHASE
    assert "K010 is" in PHASE and "z-only projection policy" in PHASE
    assert "probe_frozen_attention_projection.py" in PHASE
