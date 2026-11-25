"""Convert completed pilot worker records into canonical family partitions."""

import argparse
from pathlib import Path

import numpy as np

from experiments.publications import require_output
from experiments.config import read_config
from experiments.parquet import publish_family
from experiments.runner import arms
from experiments.storage import budgets, digest, load_json, root_paths


def _check_worker_inventory(source, family, count, families):
    for area, suffix in (("samples", ".npz"), ("branches", ".json")):
        directory = source / area
        if directory.exists():
            extra = {p.name for p in directory.iterdir()} - families
            if extra:
                raise ValueError(
                    f"Unexpected worker inventory in {directory}: {sorted(extra)}"
                )
        partition = directory / family
        expected = (
            set()
            if family == "C1" and area == "branches"
            else {f"{root:04d}{suffix}" for root in range(count)}
        )
        actual = {p.name for p in partition.iterdir()} if partition.exists() else set()
        if actual != expected:
            raise ValueError(
                f"Incorrect worker inventory in {partition}: {sorted(actual ^ expected)}"
            )


def consolidate(source, destination, config, family, signature, placebo=None):
    """Check original completion markers and cohort digests before publishing."""
    count = config.placebo_roots if family == "C1" else budgets(config)[family]
    _check_worker_inventory(source, family, count, {"C1", *budgets(config)})
    expected_arms = {a.name for a in arms(config, family)}
    if family == "C1":
        if len(placebo) != count:
            raise ValueError("Incorrect placebo root list")
        for root, row in enumerate(placebo):
            if (
                set(row) != {"root_id", "passed", "arms", "sample_digest"}
                or type(row["root_id"]) is not int
                or row["root_id"] != root
                or row["passed"] is not True
            ):
                raise ValueError(
                    f"Incomplete or unexpected placebo evidence for root {root}"
                )

    def records():
        for root in range(count):
            branch_path, sample_path = root_paths(source, family, root)
            if family == "C1":
                row = placebo[root]
                record = dict(
                    root=root,
                    family=family,
                    signature=signature,
                    branches=row["arms"],
                    sample_digest=row["sample_digest"],
                )
            else:
                record = load_json(branch_path)
            if (
                record["signature"] != signature
                or set(record["branches"]) != expected_arms
                or any(
                    len(b["windows"]) != config.horizon // config.window
                    for b in record["branches"].values()
                )
            ):
                raise ValueError(
                    f"Invalid branch vector or provenance in {family}/{root}"
                )
            if digest(sample_path) != record["sample_digest"]:
                raise ValueError(f"Sample digest mismatch at {sample_path}")
            with np.load(sample_path, allow_pickle=False) as samples:
                yield record, samples

    publish_family(destination, family, records())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original pilot result directory")
    parser.add_argument(
        "destination", type=Path, help="Adds evidence/ alongside existing files"
    )
    args = parser.parse_args()
    require_output(args.destination)
    config = read_config(args.source / "pilot.toml")
    signature = load_json(args.source / "manifest.json")["signature"]
    placebo = load_json(args.source / "validation/placebo.json")
    for family in ("C1", *budgets(config)):
        consolidate(args.source, args.destination, config, family, signature, placebo)


if __name__ == "__main__":
    main()
