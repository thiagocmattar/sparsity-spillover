"""Select paper evidence and give records scientific names instead of archive IDs."""

import json
import re


def condition(row):
    pressure = {"none": "P0", "h": "Ph", "all": "Pall", "L1": "L1"}[row["pressure"]]
    dose = (
        row.get("local_pressure_weight")
        if row["scope"] == "1" and row["pressure"] != "none"
        else row.get("kappa")
    )
    return (
        f"{row['model']}-T{'2' if row['scope'] == 'hz' else row['scope']}-{pressure}"
        + (f"-{dose:g}" if dose is not None else "")
    )


def clean_results(name, data, sources):
    endpoints = json.loads(sources["endpoints"].read_text())["trained_points"]
    attempts = {r["source_attempt"]: condition(r) for r in endpoints}
    sessions = {
        "Run029": "14m-main",
        "Run033": "14m-seven-site-h-pressure",
        "Run036": "posthoc-clipping",
        "Run041": "14m-two-site",
        "Run044": "14m-two-site-high-threshold",
        "Run042": "70m-component-controls",
        "Run045": "70m-main",
        "Run046": "70m-two-site-training",
        "Run047": "70m-additional-endpoint",
        "Run037": "14m-operation-ablation",
        "Run035": "70m-reference-port",
        "Run039": "14m-mechanism",
    }
    fields = {
        "endpoints": [
            "trained_points",
            "clipping_points",
            "ceilings",
            "coverage",
            "training_steps",
            "training_tokens",
            "cross_size_protocol",
            "loss_convention",
            "paired_pressure",
            "matched_summary",
            "later_session",
            "timing_note",
        ],
        "14m-figure": ["series", "trained_points", "clipping_points", "ceilings"],
        "pressure-placement": ["coverage", "rows", "contrasts"],
        "clipping": [
            "posthoc_trained",
            "posthoc_points",
            "coverage",
            "loss_convention",
        ],
        "kernel-structure": [
            "base_references",
            "points",
            "coverage",
            "definition",
            "limits",
        ],
        "ol1-geometry": ["geometry", "t1_comparison"],
        "base-training": [
            "coverage",
            "task_loss_definition",
            "gradient_definition",
            "smoothing",
            "runs",
        ],
        "70m-controls": [
            "rows",
            "device_uuid",
            "native_base_ms",
            "sparse_ms",
            "estimand",
        ],
        "70m-sessions": ["references", "later"],
        "operation-latency": [
            "scope",
            "coverage",
            "timing",
            "controls_ms",
            "control_process_ranges_ms",
            "attention_overhead_microseconds",
            "verification",
            "rows",
        ],
    }
    data = {k: data[k] for k in fields[name]}
    if name == "70m-controls":
        data["rows"] = [
            r
            for r in data["rows"]
            if r["implementation"] in {"selected-native-hz", "opt073"}
        ]
    omitted = {
        "run",
        "attempt_id",
        "events_source",
        "manifest_source",
        "code_identity",
        "diagnostic_source",
        "source_records",
        "sources_sha256",
        "source_sha256",
        "verified_primary_sources",
        "verified_historical_code",
        "source",
        "script",
        "script_sha256",
        "observations_source",
        "observation",
        "observations_path",
        "historical_h_only_source",
        "historical_h_only_audit",
        "declared_pressure_sites",
        "parent_manifest",
        "parent_manifests",
        "output",
        "output_sha256",
        "figure",
        "file",
        "path",
    }

    def clean(value):
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            result = {}
            for key, v in value.items():
                if (
                    key in omitted
                    or re.fullmatch(r"run\d+_id", key)
                    or key.endswith("_source")
                    or key.startswith("runs/")
                    or key.startswith("analyses/")
                ):
                    continue
                if key == "source_attempt":
                    if v in attempts:
                        result["condition"] = attempts[v]
                    continue
                if isinstance(v, str) and (
                    "runs/" in v or "analyses/" in v or "artifacts/attempts/" in v
                ):
                    continue
                result[clean(key)] = clean(v)
            if (
                all(k in value for k in ("model", "scope", "pressure"))
                and value.get("scope") in ("0", "1", "hz", "4", "7")
                and "source_attempt" in value
            ):
                result["condition"] = attempts.get(
                    value["source_attempt"], condition(value)
                )
            return result
        if isinstance(value, str):
            if value in attempts:
                return attempts[value]
            for old, new in sessions.items():
                value = value.replace(old, new)
            value = re.sub(r"\b(?:K050|k050)\b", "specialized-14m", value)
            value = re.sub(r"\bopt073\b", "specialized-70m", value)
            value = value.replace("selected-native-hz", "native-hz")
            if name in {"70m-controls", "70m-sessions"}:
                value = {
                    "c00": "70M-T0-P0",
                    "c21": "70M-T7-Ph-0.5",
                    "c25": "70M-T2-Ph-0.1",
                    "c26": "70M-T2-Ph-0.5",
                }.get(value, value)
            return value
        return value

    result = clean(data)
    if name == "endpoints":
        result["timing_note"] = (
            "Session identities are distinct. Do not pool or rescale cross-session latency."
        )
    return result
