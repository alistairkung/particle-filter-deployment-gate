"""Compare deterministic evidence from two independent executions of the same contract."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

from experiments.storage import digest, load_json, save_json


def inventory(root):
    paths = []
    for name in ("branches", "samples", "windows", "validation", "summary"):
        paths.extend(p for p in (root / name).rglob("*") if p.is_file())
    return {p.relative_to(root) for p in paths if p.name != "clean-reproduction.json"}


def compare(original, reproduced):
    first = load_json(original / "manifest.json")
    second = load_json(reproduced / "manifest.json")
    if (
        original.resolve() == reproduced.resolve()
        or first["signature"] != second["signature"]
    ):
        raise ValueError(
            "Expected distinct execution directories with identical frozen inputs"
        )
    if first["status"] != "report complete" or second["status"] != "report complete":
        raise ValueError("Both runs must be complete")
    expected = inventory(original)
    observed = inventory(reproduced)
    if expected != observed:
        raise ValueError(
            f"Output inventories differ: {sorted(expected.symmetric_difference(observed))[:5]}"
        )
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
