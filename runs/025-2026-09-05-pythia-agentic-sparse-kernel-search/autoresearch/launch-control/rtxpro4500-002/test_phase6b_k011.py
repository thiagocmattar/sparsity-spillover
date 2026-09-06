from pathlib import Path


SCRIPT = Path(__file__).with_name("phase6b-14m-k011-mask-search.sh")


def test_phase6b_has_fixed_twelve_masks_and_three_development_conditions():
    source = SCRIPT.read_text(encoding="utf-8")
    masks = (
        "prefix1 prefix2 prefix3 prefix4 prefix5 "
        "suffix1 suffix2 suffix3 suffix4 suffix5 even odd"
    )
    assert f"for run025_mask in {masks}; do" in source
    assert source.count("run_probe 14m/") == 3
    assert "run_probe 14m/a7-0 a7-0" in source
    assert "run_probe 14m/a4-0p5 a4-0p5" in source
    assert "run_probe 14m/a7-0p5 a7-0p5" in source
    assert "--inputs 16 --passes 3" in source
    assert "0p01" not in source and "0p05" not in source
