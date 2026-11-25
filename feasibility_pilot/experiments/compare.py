"""Compare deterministic evidence from two independent executions of the same contract."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.config import read_config
from experiments.storage import budgets, digest, load_json, save_json


def inventory(root):
    paths = []
    for name in ("branches", "samples", "windows", "validation", "summary"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    return {p.relative_to(root) for p in paths if p.name != "clean-reproduction.json"}


def expected_inventory(config):
    paths = {
        Path("validation") / name
        for name in (
            "checks.json",
            "queue.json",
            "placebo.json",
            "result.json",
            "workload-integrity.json",
        )
    }
    paths.update(
        Path("summary") / name
        for name in (
            "branch-diagnostics.csv",
            "window-diagnostics.csv",
            "db-work.csv",
            "latency-by-route.csv",
            "verdict.json",
            "independent-audit.json",
        )
    )
    paths.update(
        Path("samples/C1") / f"{root:04d}.npz" for root in range(config.placebo_roots)
    )
    for family, count in budgets(config).items():
        paths.add(Path("windows") / f"{family}.csv")
        for root in range(count):
            paths.add(Path("samples") / family / f"{root:04d}.npz")
            paths.add(Path("branches") / family / f"{root:04d}.json")
    return paths


def _check_manifest(root, manifest, config_path):
    config = read_config(config_path)
    expected = budgets(config)
    input_digest = digest(config_path)
    if digest(root / "pilot.toml") != input_digest or manifest.get("config") != asdict(
        config
    ):
        raise ValueError(
            f"{root}: manifest/config does not match the approved workload"
        )
    if (
        manifest.get("status") != "report complete"
        or manifest.get("expected") != expected
        or manifest.get("completed") != expected
    ):
        raise ValueError(f"{root}: incomplete manifest root counts")
    files = manifest.get("input_digests", {})
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if manifest.get("signature") != signature or files.get("config") != input_digest:
        raise ValueError(f"{root}: inconsistent manifest signature")
    required = expected_inventory(config)
    observed = inventory(root)
    if observed != required:
        raise ValueError(
            f"{root}: incomplete or unexpected evidence: {sorted(observed ^ required)[:5]}"
        )
    if any((root / path).stat().st_size == 0 for path in required):
        raise ValueError(f"{root}: empty evidence file")
    validation = load_json(root / "validation/result.json")
    if (
        not validation["passed"]
        or validation["signature"] != signature
        or sorted(row["root_id"] for row in validation["placebo"])
        != list(range(config.placebo_roots))
    ):
        raise ValueError(f"{root}: invalid persisted prerequisite evidence")
    if not load_json(root / "summary/independent-audit.json")["verified"]:
        raise ValueError(f"{root}: unaudited conclusion")
    return required


def compare(original, reproduced, config_path=None):
    config_path = (
        config_path or Path(__file__).resolve().parents[1] / "configs/pilot.toml"
    )
    first = load_json(original / "manifest.json")
    second = load_json(reproduced / "manifest.json")
    if original.resolve() == reproduced.resolve():
        raise ValueError("Expected distinct execution directories")
    expected = _check_manifest(original, first, config_path)
    _check_manifest(reproduced, second, config_path)
    for key in (
        "signature",
        "input_digests",
        "analysis_source_files",
        "seeds",
        "python",
        "dependencies",
        "ordering",
        "time_representation",
    ):
        if key not in first or first[key] != second.get(key):
            raise ValueError(f"Run manifests differ on {key}")
    ledger = hashlib.sha256()
    for path in sorted(expected):
        a = digest(original / path)
        if a != digest(reproduced / path):
            raise ValueError(f"Deterministic outputs differ: {path}")
        ledger.update(f"{path} {a}\n".encode())
    return {
        "passed": True,
        "signature": first["signature"],
        "files_compared": len(expected),
        "by_area": dict(Counter(path.parts[0] for path in expected)),
        "inventory_sha256": ledger.hexdigest(),
        "reproduction_elapsed_s": second["timings_s"],
        "completed_roots": second["completed"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("reproduced", type=Path)
    args = parser.parse_args()
    result = compare(args.original, args.reproduced)
    save_json(args.original / "summary/clean-reproduction.json", result)
    print("Verified", result["files_compared"], "deterministic outputs")
