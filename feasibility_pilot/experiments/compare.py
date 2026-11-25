"""Compare deterministic evidence from two independent executions of the same contract."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from experiments.publications import require_output
from experiments.config import read_config
from experiments.parquet import EVIDENCE_FORMAT, read_records, iter_samples
from experiments.runner import arms
from experiments.storage import budgets, digest, load_json, save_json


def inventory(root):
    paths = []
    for name in ("evidence", "validation", "summary"):
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
        Path("evidence") / family / f"{table}.parquet"
        for family in ("C1", *budgets(config))
        for table in ("branches", "windows", "samples")
    )
    return paths


def _check_manifest(root, manifest, config_path):
    if manifest.get("evidence_format", EVIDENCE_FORMAT) != EVIDENCE_FORMAT:
        raise ValueError(f"{root}: unsupported pilot evidence format")
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
    for family, count in {"C1": config.placebo_roots, **expected}.items():
        records = read_records(
            root,
            family,
            {a.name for a in arms(config, family)},
            count,
            signature,
            config.horizon // config.window,
        )
        for _ in iter_samples(root, family, records):
            pass
    return required


def inventory_receipt(root, config_path=None):
    """Bind a comparison to every file in the canonical pilot inventory."""
    config_path = (
        config_path or Path(__file__).resolve().parents[1] / "configs/pilot.toml"
    )
    manifest = load_json(root / "manifest.json")
    expected = _check_manifest(root, manifest, config_path)
    ledger = hashlib.sha256()
    for path in sorted(expected):
        ledger.update(f"{path} {digest(root / path)}\n".encode())
    return {
        "format": EVIDENCE_FORMAT,
        "passed": True,
        "signature": manifest["signature"],
        "files_compared": len(expected),
        "by_area": dict(Counter(path.parts[0] for path in expected)),
        "inventory_sha256": ledger.hexdigest(),
        "completed_roots": manifest["completed"],
    }


def compare(original, reproduced, config_path=None):
    if original.resolve() == reproduced.resolve():
        raise ValueError("Expected distinct execution directories")
    baseline = inventory_receipt(original, config_path)
    candidate = inventory_receipt(reproduced, config_path)
    if baseline != candidate:
        raise ValueError("Canonical evidence or retained validation/summary differs")
    first = load_json(original / "manifest.json")
    second = load_json(reproduced / "manifest.json")
    for key in (
        "signature",
        "input_digests",
        "seeds",
        "python",
        "ordering",
        "time_representation",
    ):
        if key not in first or first[key] != second.get(key):
            raise ValueError(f"Run manifests differ on {key}")
    # The original execution predates Parquet. After migration the analysis sources
    # and storage runtime must also match between new executions.
    if "pyarrow" in first["dependencies"] and (
        first.get("analysis_source_files") != second.get("analysis_source_files")
        or first["dependencies"]["pyarrow"] != second["dependencies"].get("pyarrow")
    ):
        raise ValueError("Run manifests differ on analysis sources or Parquet version")
    # Storage-only changes introduce PyArrow. The pinned numerical runtime must match.
    old, new = first["dependencies"].copy(), second["dependencies"].copy()
    old.pop("pyarrow", None)
    new.pop("pyarrow", None)
    if old != new:
        raise ValueError("Run manifests differ on numerical dependencies")
    return {**baseline, "reproduction_elapsed_s": second["timings_s"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("reproduced", type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Optional path for a new Parquet comparison receipt",
    )
    args = parser.parse_args()
    if args.receipt:
        require_output(args.receipt)
    result = compare(args.original, args.reproduced)
    if args.receipt:
        save_json(args.receipt, result)
    print("Verified", result["files_compared"], "deterministic outputs")
