"""Release boundaries: resolved science, portable source closure and actual updates."""

import collections
import json
import sys
import numpy as np
import pytest
import torch
from training.config import ROOT, identifier, paper_rows, resolve, register_hz
from sparsity_research.data import build_training_schedule
from sparsity_research.sites import resolve_topology_and_gates


def test_grid_gate_pressure_and_schedule_contracts():
    register_hz()
    rows = paper_rows()
    assert collections.Counter(r["model"] for r in rows) == {
        "14M": 45,
        "31M": 11,
        "70M": 27,
        "410M": 12,
    }
    assert len({identifier(r) for r in rows}) == 95
    for r in rows:
        assert r["condition"] == identifier(r)
        c = resolve(identifier(r))
        m = c["model"]
        t = c["training"]
        topology, gates = resolve_topology_and_gates(
            m["topology_id"], None, m["site_gates"]
        )
        if r["scope"] == "hz":
            assert set(topology.active_sites) == {"h", "z"} and c["pressure"][
                "sites"
            ] == ["h"]
        if r["pressure"] == "all":
            assert set(c["pressure"]["sites"]) == set(topology.active_sites)
        if r["pressure"] == "h":
            assert c["pressure"]["sites"] == ["h"]
        assert t["micro_batch_size"] * t["gradient_accumulation_steps"] == 1024
        assert c["expected_initial_parameter_sha256"] and c["expected_schedule_sha256"]
    for size in ("14M", "31M", "70M", "410M"):
        c = resolve(next(identifier(r) for r in rows if r["model"] == size))
        t = c["training"]
        _, sha, meta = build_training_schedule(
            np,
            token_count=1491711416,
            block_size=2048,
            max_steps=712,
            micro_batch_size=t["micro_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            seed=1234,
        )
        assert sha == c["expected_schedule_sha256"]
        assert meta["wrapped_blocks"] == 714


@pytest.mark.parametrize(
    "condition", ["14M-T0-P0", "14M-T1-L1-0.1", "14M-T2-Ph-0.1", "14M-T7-Pall-0.1"]
)
def test_real_portable_boundary_and_serialization(tmp_path, monkeypatch, condition):
    from training.train import main
    from transformers import AutoModelForCausalLM
    from sparsity_research.pythia import load_checkpoint_pythia

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--condition",
            condition,
            "--smoke",
            "--output",
            str(tmp_path / "run"),
        ],
    )
    main()
    rows = [
        json.loads(x) for x in (tmp_path / "run/events.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2 and all(not r["optimizer_step_skipped"] for r in rows)
    m = load_checkpoint_pythia(
        AutoModelForCausalLM, tmp_path / "run/final", torch=torch
    )
    assert m.config.topology_id == resolve(condition)["model"]["topology_id"]
    if resolve(condition)["pressure"]["method"] == "orthogonal_l1":
        assert all(r["ol1_correction_applied"] for r in rows)
        assert rows[-1]["pressure_capture_tensor_count"] == 2 * len(
            resolve(condition)["pressure"]["sites"]
        )


def test_final_kernel_source_closure_and_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    import support

    for index, path in enumerate(sorted((ROOT / "kernels").rglob("*.py"))):
        support.module(f"closure_check_{index}", path)
    assert not list((ROOT / "kernels").rglob("candidate.py"))
    assert not list(ROOT.rglob("*.tex"))
    assert (ROOT / "main.pdf").read_bytes().startswith(b"%PDF-")


def test_clipping_order_statistics_and_equality():
    from training.clipping import empirical_thresholds

    values = np.array([0, 0, 1, 3], dtype=np.float32)
    result = empirical_thresholds(values, (0, 0.5, 0.75), np=np)
    assert [r["threshold"] for r in result["targets"]] == [0, 0, 1]
    assert result["targets"][-1]["calibration_zero_count"] == 3


@pytest.mark.parametrize(
    "size,width,heads,operation",
    [
        ("14M", 128, 4, None),
        ("70M", 512, 8, None),
        ("31M", 256, 8, None),
        ("14M", 128, 4, "full"),
        ("14M", 128, 4, "without-h"),
        ("14M", 128, 4, "without-pv"),
        ("14M", 128, 4, "off"),
    ],
)
def test_final_kernel_assembly_without_cuda_execution(
    monkeypatch, size, width, heads, operation
):
    from transformers import GPTNeoXConfig, AutoModelForCausalLM
    from sparsity_research.pythia import apply_activation_topology

    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    import install as final_kernel

    c = GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=width,
        intermediate_size=4 * width,
        num_hidden_layers=6,
        num_attention_heads=heads,
        max_position_embeddings=2048,
    )
    recipe = resolve(f"{size}-T2-Ph-0.1")
    c.topology_id = "HZ"
    c.site_gates = recipe["model"]["site_gates"]
    c.use_cache = False
    model = (
        apply_activation_topology(AutoModelForCausalLM.from_config(c), torch=torch)
        .to(dtype=torch.bfloat16)
        .eval()
    )
    original = [
        (
            layer.mlp.dense_4h_to_h,
            layer.attention.dense,
            layer.attention.query_key_value,
            layer.mlp.dense_h_to_4h,
        )
        for layer in model.gpt_neox.layers
    ]
    with torch.inference_mode():
        metadata = final_kernel.install(
            model, "optimized", None if operation or size == "31M" else "native-hz", operation
        )
    assert metadata
    for layer, (down, output, qkv, up) in zip(model.gpt_neox.layers, original):
        assert layer._sparsity_joint.w2 is down and layer._sparsity_joint.wo is output
        assert layer.attention.query_key_value is qkv and layer.mlp.dense_h_to_4h is up
        assert layer._sparsity_norm_pair.first is layer.input_layernorm
        assert layer._sparsity_norm_pair.second is layer.post_attention_layernorm
        assert layer.attention._sparsity_rope.thresholds == [0.0, 0.0, 0.0]
        assert layer._sparsity_joint.th == layer._sparsity_joint.tz == 0.1
        assert isinstance(layer.mlp.dense_4h_to_h, torch.nn.Identity)
    if size in {"31M", "70M"}:
        assert model.embed_out._sparsity_head.weight is model.embed_out.weight
        assert model.embed_out._sparsity_head.tile == 0
        assert all(
            layer.attention._sparsity_attention.native
            for layer in model.gpt_neox.layers
        )
        assert all(
            not hasattr(layer.attention.query_key_value, "_sparsity_projection")
            for layer in model.gpt_neox.layers
        )
    if operation:
        for layer in model.gpt_neox.layers:
            mask = metadata["operation_mask"]
            assert layer._sparsity_joint.skip_h == mask["h"]
            assert layer._sparsity_joint.skip_z == mask["z"]
            assert layer.attention._sparsity_attention.skip_qk == mask["qk"]
            assert layer.attention._sparsity_attention.skip_pv == mask["pv"]
            assert (
                layer.attention.query_key_value._sparsity_projection.skip == mask["a"]
            )
            assert layer.mlp.dense_h_to_4h._sparsity_projection.skip == mask["m"]
    elif size != "31M":
        assert all(layer._sparsity_joint.native for layer in model.gpt_neox.layers)


def test_final_extension_build_inputs_are_self_contained(monkeypatch):
    """Exercise build-path resolution without compiling or launching CUDA."""
    from pathlib import Path
    from types import SimpleNamespace
    import torch.utils.cpp_extension

    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    import support

    requests = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(
        torch.utils.cpp_extension,
        "load",
        lambda **kw: requests.append(kw) or SimpleNamespace(),
    )
    paths = [
        *list((ROOT / "kernels/model_14m").rglob("*.py")),
        *list((ROOT / "kernels/model_70m").rglob("*.py")),
        *list((ROOT / "kernels/model_31m").rglob("*.py")),
        ROOT / "kernels/rotary/implementation.py",
        ROOT / "kernels/ablation/implementation.py",
    ]
    for i, path in enumerate(paths):
        m = support.module(f"build_inputs_{i}", path)
        # Downloaded headers are covered by the fetcher's hashes, not this CPU check.
        if hasattr(m, "read_json"):
            monkeypatch.setattr(m, "read_json", lambda path: {"files": []})
        if "ablation" in path.parts:
            m.extension("joint", True, False)
            m.extension("attention", False, True)
        else:
            m.extension()
    assert len(requests) == 15
    for request in requests:
        for source in request["sources"]:
            path = Path(source).resolve()
            assert path.is_relative_to(ROOT / "kernels") and path.is_file()
        for include in request.get("extra_include_paths", []):
            assert Path(include).resolve().is_relative_to(ROOT / "kernels")
    switches = [
        flag
        for request in requests
        for flag in request["extra_cuda_cflags"]
        if flag.startswith("-DSPARSE_SKIP_")
    ]
    assert set(switches) == {
        "-DSPARSE_SKIP_H=1",
        "-DSPARSE_SKIP_Z=0",
        "-DSPARSE_SKIP_QK=0",
        "-DSPARSE_SKIP_PV=1",
    }


@pytest.mark.parametrize("condition,width,heads,hz_mode", [
    ("31M-T7-Ph-0.1", 256, 8, None),
    *(('14M-T2-Ph-0.1', 128, 4, m) for m in ('A', 'B', 'C', 'D')),
])
def test_seven_site_gates_and_hz_only_switches(monkeypatch, condition, width, heads, hz_mode):
    from transformers import GPTNeoXConfig, AutoModelForCausalLM
    from sparsity_research.pythia import apply_activation_topology
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    import install
    recipe = resolve(condition)["model"]
    c = GPTNeoXConfig(vocab_size=50304, hidden_size=width, intermediate_size=4*width,
                     num_hidden_layers=6, num_attention_heads=heads, max_position_embeddings=2048)
    c.topology_id, c.site_gates = recipe["topology_id"], recipe["site_gates"]
    model = apply_activation_topology(AutoModelForCausalLM.from_config(c), torch=torch).bfloat16().eval()
    with torch.inference_mode():
        install.install(model, hz_mode=hz_mode)
    for layer in model.gpt_neox.layers:
        pair = layer._sparsity_norm_pair
        if hz_mode:
            assert (layer._sparsity_joint.skip_h, layer._sparsity_joint.skip_z) == {
                'A': (True, True), 'B': (False, True), 'C': (True, False), 'D': (False, False)}[hz_mode]
            assert layer.attention._sparsity_attention.skip
            assert not hasattr(layer.attention._sparsity_attention, 'skip_qk')
            assert layer.attention.query_key_value._sparsity_projection.skip
            assert layer.mlp.dense_h_to_4h._sparsity_projection.skip
            assert not pair.ga and not pair.gm
        else:
            threshold = float(torch.tensor(.1, dtype=torch.bfloat16))
            assert pair.ga and pair.gm and pair.ta == pair.tm == threshold
            # RoPE converts thresholds inside CUDA; normalization preconverts BF16 scalars.
            assert layer.attention._sparsity_rope.thresholds == [.1] * 3
            assert layer.attention._sparsity_attention.native
        assert layer._sparsity_joint.gh and layer._sparsity_joint.gz
        assert layer._sparsity_joint.th == layer._sparsity_joint.tz == .1


def test_timing_pool_and_rejection_of_mixed_or_incomplete_processes(tmp_path):
    from scripts.aggregate_timings import aggregate

    folders = []
    for replicate in (1, 2, 3):
        folder = tmp_path / str(replicate)
        folder.mkdir()
        folders.append(folder)
        status = dict(
            replicate=replicate,
            weight_sha256="test-checkpoint",
            backend="optimized",
            control=None,
            operation_mode=None,
            gpu="test-device",
            runtime={"torch": "test"},
            qualified=True,
            status="qualified",
        )
        (folder / "result.json").write_text(json.dumps(status))
        samples = [
            dict(
                mode=mode,
                repeat=r,
                input_index=i,
                host_ms=(2.0 if mode == "native" else 1.0) * replicate,
            )
            for mode in ("native", "candidate")
            for r in range(7)
            for i in range(64)
        ]
        (folder / "timing.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in samples)
        )
    summary = aggregate(folders)
    assert summary["same_checkpoint_speedup"] == pytest.approx(2.0)
    assert summary["latency_ms"]["candidate"] == pytest.approx(6.0 ** (1 / 3))
    assert summary["process_ranges_ms"]["candidate"] == pytest.approx([1.0, 3.0])
    assert summary["samples_per_implementation"] == 1344
    status["weight_sha256"] = "another-checkpoint"
    (folders[-1] / "result.json").write_text(json.dumps(status))
    with pytest.raises(ValueError, match="Mixed weight_sha256"):
        aggregate(folders)
    status["weight_sha256"] = "test-checkpoint"
    (folders[-1] / "result.json").write_text(json.dumps(status))
    (folders[-1] / "timing.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples[:-1])
    )
    with pytest.raises(ValueError, match="Incomplete/duplicate"):
        aggregate(folders)
