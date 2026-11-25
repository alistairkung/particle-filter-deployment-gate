"""Convert completed pilot worker records into canonical family partitions."""

import argparse
from pathlib import Path

import numpy as np

from experiments.config import read_config
from experiments.parquet import publish_family
from experiments.runner import arms
from experiments.storage import budgets, digest, load_json, root_paths


def consolidate(source, destination, config, family, signature, placebo=None):
    """Check original completion markers and cohort digests before publishing."""
    count = config.placebo_roots if family == "C1" else budgets(config)[family]
    expected_arms = {a.name for a in arms(config, family)}
    if family == "C1" and [row["root_id"] for row in placebo] != list(range(count)):
        raise ValueError("Incorrect placebo root list")

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
    config = read_config(args.source / "pilot.toml")
    signature = load_json(args.source / "manifest.json")["signature"]
    placebo = load_json(args.source / "validation/placebo.json")
    for family in ("C1", *budgets(config)):
        consolidate(args.source, args.destination, config, family, signature, placebo)


if __name__ == "__main__":
    main()
