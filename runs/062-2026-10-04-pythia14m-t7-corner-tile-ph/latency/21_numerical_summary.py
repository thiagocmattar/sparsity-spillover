"""Summarize the sealed full-coverage own-eager numerical qualification."""
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def record(path):
    return {"path": path.relative_to(RUN).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    attempts = json.loads((RUN / "artifacts/scientific/summary-001.json").read_text())
    assert len(attempts) == 6
    processes, sources = [], []
    for attempt in attempts:
        path = RUN / "artifacts/attempts" / attempt["attempt"] / "quality.json"
        quality = json.loads(path.read_text())
        groups = []
        assert set(quality) == {"scalar", "tile"}
        for family, q in quality.items():
            assert q["pass"] and q["blocks"] == 338 and q["documents"] == 500
            assert q["input_tokens"] == 692224 and q["prediction_tokens"] == 691886
            assert q["excluded_tail_tokens"] == 1444
            assert set(q["gates"]) == {"native_graph", "no_skip_graph", "sparse_graph"}
            rows = [row for graph in q["gates"].values() for row in graph]
            assert all(len(graph) == 338 for graph in q["gates"].values())
            assert all(row["pass"] and row["finite"] for row in rows)
            assert all(value == 0 for value in q["loss_delta"].values())
            groups.append({"family": family, "blocks": q["blocks"],
                           "prediction_tokens": q["prediction_tokens"],
                           "comparisons": len(rows),
                           "max_abs": max(row["max_abs"] for row in rows),
                           "max_relative_l2": max(row["relative_l2"] for row in rows),
                           "all_pass": True})
        processes.append({"attempt": attempt["attempt"], "qualified": True, "groups": groups})
        sources.append(record(path))
    output = {"status": "verified", "scope": "All six complete processes, two checkpoints and three graphs each; own-eager BF16 logits",
              "processes": processes, "source_files": sources, "source_script": record(Path(__file__).resolve())}
    destination = RUN / "results/numerical-verification.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Verified", sum(g["comparisons"] for p in processes for g in p["groups"]), "model/graph/block comparisons")


if __name__ == "__main__":
    main()
