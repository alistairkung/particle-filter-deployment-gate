"""Verify the committed migration receipts against immutable sources and current canonical evidence."""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from .source import (
    FROZEN,
    PILOT,
    READY,
    RECEIPTS,
    REVISIONS,
    ROOT,
    TELEMETRY,
    dump,
    git_objects,
    inventory,
    tree,
)


def replaced(path):
    if path.is_relative_to(FROZEN):
        return path.relative_to(FROZEN).parts[0] in {"branches", "windows", "samples"}
    return path.is_relative_to(READY) and (
        path.suffix == ".gz" or path.name == "inventory.json"
    )


def provenance(trees, read, require_cleanup=False):
    kept, removed = {}, []
    for study, files in trees.items():
        for path, entry in files.items():
            relative = path.relative_to(study)
            if replaced(path):
                removed.append(path)
                if require_cleanup and (ROOT / path).exists():
                    raise ValueError(f"Legacy data present: {path}")
            elif relative.parts[0] in {
                "results",
                "plots",
                "configs",
            } or relative.name in {
                "FEASIBILITY_REPORT.md",
                "SPEC.md",
                "IMPLEMENTATION_PLAN.md",
                "requirements.lock",
                "SCIENTIFIC_SPEC.md",
                "IMPLEMENTATION_SPEC.md",
                "SPEC_REVIEW.md",
            }:
                data = (ROOT / path).read_bytes()
                if data != read(entry):
                    raise ValueError(
                        f"Historical report, configuration or evidence changed: {path}"
                    )
                kept[str(path)] = hashlib.sha256(data).hexdigest()
    # The historical audit implementations and FIFO dynamics must be the ones used in the study.
    for relative in ["audit.py", "audit_models.py", "audit_statistics.py"]:
        path = Path(TELEMETRY, relative)
        if (ROOT / path).read_bytes() != read(trees[TELEMETRY][path]):
            raise ValueError(f"Independent scientific audit changed: {path}")
    # The physical scheduler and named RNG primitives moved. Compare their
    # complete definitions against the immutable source, ignoring import paths.
    for old, new, names in (
        ("model.py", "model.py", ("SerialDB", "Write", "Model")),
        ("inputs.py", "workload.py", ("rng", "arrivals")),
    ):
        path = Path(PILOT, "simulator", old)
        original = read(trees[PILOT][path])
        current = (ROOT / "simulator" / new).read_bytes()
        if _definitions(current, names) != _definitions(original, names):
            raise ValueError(f"Historical simulator definition changed: {new}")
    return dict(byte_identical=kept, replaced_files=len(removed))


def _definitions(source, names):
    definitions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    return [ast.dump(definitions[name]) for name in names]


def canonical_files():
    expected_pilot = {
        ROOT / FROZEN / "evidence" / family / (name + ".parquet")
        for family in ("C1", "C23", "C4", "C5")
        for name in ("branches", "windows", "samples")
    }
    if set((ROOT / FROZEN).rglob("*.parquet")) != expected_pilot:
        raise ValueError("Unexpected or missing pilot Parquet files")
    files = sorted(
        expected_pilot
        | set((ROOT / READY).rglob("*.parquet"))
        | {ROOT / READY / "development/selection.json"}
    )
    result = {}
    for path in files:
        with path.open("rb") as stream:
            meta = {
                "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                "bytes": path.stat().st_size,
            }
        if path.suffix == ".parquet":
            meta["rows"] = pq.read_metadata(path).num_rows
            meta["schema"] = str(pq.read_schema(path).remove_metadata())
        result[str(path.relative_to(ROOT))] = meta
    return result


def result_sizes(trees, canonical, require_cleanup):
    sizes = {}
    for study, base in ((PILOT, FROZEN), (TELEMETRY, READY)):
        original = {
            p: size for p, (_, size) in trees[study].items() if p.is_relative_to(base)
        }
        active = {p: n for p, n in original.items() if not replaced(p)}
        active.update(
            {
                Path(p): v["bytes"]
                for p, v in canonical.items()
                if Path(p).is_relative_to(base)
            }
        )
        if require_cleanup:
            staged = subprocess.check_output(
                [
                    "git",
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                    "--",
                    str(base),
                ],
                cwd=ROOT,
            )
            paths = {
                Path(p.decode())
                for p in staged.split(b"\0")
                if p and (ROOT / p.decode()).is_file()
            }
            if set(active) != paths:
                raise ValueError(
                    f"{base}: unexpected active evidence inventory: {sorted(set(active) ^ paths)}"
                )
        sizes[str(base)] = {
            "original_files": len(original),
            "original_bytes": sum(original.values()),
            "current_files": len(active),
            "current_bytes": sum(active.values()),
        }
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=6, help="parallel protected-root replays"
    )
    parser.add_argument(
        "--write-receipts",
        action="store_true",
        help="replace saved receipts after every check passes",
    )
    parser.add_argument(
        "--require-cleanup",
        action="store_true",
        help="also require replaced originals to be absent",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    sys.path.insert(0, str(ROOT / PILOT))
    from . import pilot, telemetry

    trees = {study: tree(study) for study in REVISIONS}
    with TemporaryDirectory() as scratch, git_objects() as read:
        temp = Path(scratch)
        inventory(temp)
        historical = provenance(trees, read, args.require_cleanup)
        dump(temp / "pilot-equivalence.json", pilot.verify(trees[PILOT], read))
        dump(
            temp / "telemetry-equivalence.json",
            telemetry.verify(trees[TELEMETRY], read, args.workers),
        )
        files = canonical_files()
        dump(
            temp / "migration-verification.json",
            dict(
                status="pass",
                original_commits=REVISIONS,
                tolerance=dict(
                    stored="exact integers, booleans, strings, nulls and binary64 values; exact array order",
                    derived_absolute=pilot.ATOL,
                    derived_relative=pilot.RTOL,
                ),
                historical_evidence=historical,
                canonical_files=files,
                result_directories=result_sizes(trees, files, args.require_cleanup),
            ),
        )
        for path in sorted(temp.iterdir()):
            saved = RECEIPTS / path.name
            if args.write_receipts:
                saved.write_bytes(path.read_bytes())
            elif path.name == "pilot-equivalence.json":
                pilot.derived(
                    json.loads(path.read_bytes()),
                    json.loads(saved.read_bytes()),
                    "pilot audit receipt",
                )
            elif saved.read_bytes() != path.read_bytes():
                raise ValueError(f"Recomputed receipt differs: {saved}")
    print(
        "Verified immutable source inventories, complete typed evidence, GO, READY and retained reports."
    )


if __name__ == "__main__":
    main()
