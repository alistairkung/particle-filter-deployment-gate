"""Compare deterministic evidence from two independent executions of the same contract."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.publications import PILOT, require_output
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


def _inventory_digest(hashes):
    ledger = hashlib.sha256()
    for path, checksum in sorted(hashes.items()):
        ledger.update(f"{path} {checksum}\n".encode())
    return ledger.hexdigest()


def _with_source_signature(root, hashes, signature):
    """Expected bytes when only the implementation signature changes."""
    expected = hashes.copy()
    validation = Path("validation/result.json")
    result = load_json(root / validation)
    result["signature"] = signature
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected[validation] = hashlib.sha256((encoded + "\n").encode()).hexdigest()
    for path in hashes:
        if path.name == "branches.parquet":
            table = pq.ParquetFile(root / path).read()
            field = table.schema.field("signature")
            column = pa.repeat(pa.scalar(signature, type=field.type), len(table))
            index = table.schema.get_field_index(field.name)
            table = table.set_column(index, field, column)
            stream = pa.BufferOutputStream()
            pq.write_table(table, stream, compression="zstd", write_page_checksum=True)
            expected[path] = hashlib.sha256(stream.getvalue()).hexdigest()
    return _inventory_digest(expected)


def inventory_receipt(root, config_path=None, *, reproduction_signature=None):
    """Bind a comparison to every file in the canonical pilot inventory."""
    config_path = (
        config_path or Path(__file__).resolve().parents[1] / "configs/pilot.toml"
    )
    manifest = load_json(root / "manifest.json")
    expected = _check_manifest(root, manifest, config_path)
    hashes = {path: digest(root / path) for path in expected}
    receipt = {
        "format": EVIDENCE_FORMAT,
        "passed": True,
        "signature": manifest["signature"],
        "files_compared": len(expected),
        "by_area": dict(Counter(path.parts[0] for path in expected)),
        "inventory_sha256": _inventory_digest(hashes),
        "completed_roots": manifest["completed"],
    }
    if reproduction_signature is not None:
        receipt.update(
            reproduction_signature=reproduction_signature,
            reproduction_inventory_sha256=_with_source_signature(
                root, hashes, reproduction_signature
            ),
        )
    return receipt


def frozen_reproduction_receipt():
    """Bind the expected frozen outputs to the actual source of a fresh pilot."""
    from experiments.__main__ import identity

    config = PILOT / "configs/pilot.toml"
    signature, _ = identity(config)
    return inventory_receipt(
        PILOT / "results/frozen", config, reproduction_signature=signature
    )


def compare(original, reproduced, config_path=None):
    if original.resolve() == reproduced.resolve():
        raise ValueError("Expected distinct execution directories")
    if original.resolve() == PILOT / "results/frozen":
        baseline = frozen_reproduction_receipt()
        signature = baseline["reproduction_signature"]
        inventory_hash = baseline["reproduction_inventory_sha256"]
    else:
        baseline = inventory_receipt(original, config_path)
        signature = baseline["signature"]
        inventory_hash = baseline["inventory_sha256"]
    candidate = inventory_receipt(reproduced, config_path)
    if (
        candidate["signature"] != signature
        or candidate["inventory_sha256"] != inventory_hash
    ):
        raise ValueError(
            "Source or deterministic outputs differ from expected reproduction"
        )
    first = load_json(original / "manifest.json")
    second = load_json(reproduced / "manifest.json")
    for key in ("seeds", "python", "ordering", "time_representation"):
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
