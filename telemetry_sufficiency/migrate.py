"""One-time lossless conversion of the audited sufficiency evidence at 624541b."""

import argparse
import gzip
import json
from pathlib import Path
import subprocess

from . import custody as c, parquet_evidence as p

BASE = "624541b"
STUDY = "telemetry_sufficiency/results/v1"


def source(name):
    return subprocess.check_output(
        ["git", "show", f"{BASE}:{STUDY}/{name}"], cwd=Path(__file__).resolve().parent
    )


def migrate(run):
    from .archive import protect_destination

    protect_destination(run)
    inventory = json.loads(source("inventory.json"))
    if inventory != c.inventory():
        raise ValueError("Archived inventory differs from the frozen seed scheme")

    def legacy(name):
        return [json.loads(line) for line in gzip.decompress(source(name)).splitlines()]

    files = [p.write(run, "inventory", inventory)]
    for partition in c.PARTITIONS:
        directory = run / partition
        roots = [r for r in inventory if r["partition"] == partition]
        public = p.checked(legacy(f"{partition}/observations.jsonl.gz"), roots)
        private = p.checked(legacy(f"{partition}/evaluator.jsonl.gz"), roots)
        files.extend(
            p.write_partition(
                directory,
                roots,
                (
                    dict(truth, history=obs["history"])
                    for truth, obs in zip(private, public, strict=True)
                ),
            )
        )
        examples = []
        for root in roots:
            if root["index"] == 0:
                examples.extend(legacy(f"{partition}/examples/{root['key']}.jsonl.gz"))
        files.extend(p.write_examples(directory, examples))
        if partition == "development":
            (fit,) = legacy("development/fit.jsonl.gz")
            files.extend(p.write_fit(directory, fit["fit"]))
        else:
            for name in ("predictions", "evaluator-predictions"):
                files.append(
                    p.write_predictions(
                        directory, name, legacy(f"protected/{name}.jsonl.gz")
                    )
                )
            files.append(
                p.write_replays(directory, legacy("protected/replay-audit.jsonl.gz"))
            )
    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    migrate(args.output)
