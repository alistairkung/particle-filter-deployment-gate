"""Independently reproduce the pinned pilot in a new output directory."""

import subprocess
import sys
import tempfile
from pathlib import Path

from . import PILOT
from .custody import file_hash, read, save


def reproduce(receipt):
    from experiments.compare import compare

    frozen = PILOT / "results/frozen"
    original = read(frozen / "summary/clean-reproduction.json")
    with tempfile.TemporaryDirectory(prefix="aims5701-pilot-") as temporary:
        output = Path(temporary)
        commands = [
            ["pytest", "-q", "tests"],
            [
                "experiments",
                "validate",
                "--config",
                "configs/pilot.toml",
                "--out",
                str(output),
            ],
            [
                "experiments",
                "run",
                "--config",
                "configs/pilot.toml",
                "--out",
                str(output),
            ],
            [
                "experiments",
                "report",
                "--from",
                str(output),
                "--plots",
                str(output / "plots"),
                "--output",
                str(output / "FEASIBILITY_REPORT.md"),
            ],
        ]
        for command in commands:
            subprocess.run([sys.executable, "-m", *command], cwd=PILOT, check=True)
        result = compare(frozen, output)
        for field in ("inventory_sha256", "files_compared", "signature"):
            if result[field] != original[field]:
                raise ValueError(f"Frozen comparison receipt differs on {field}")
        verdict = read(output / "summary/verdict.json")["verdict"]
        if verdict != "GO":
            raise ValueError("Frozen pilot verdict changed")
        subprocess.run(
            ["git", "diff", "--exit-code", "bb20ebe", "--", "."], cwd=PILOT, check=True
        )
        save(
            receipt,
            dict(
                result,
                verdict=verdict,
                pilot_revision="bb20ebe",
                frozen_manifest=file_hash(frozen / "manifest.json"),
                original_receipt=file_hash(frozen / "summary/clean-reproduction.json"),
                commands=commands,
            ),
        )


if __name__ == "__main__":
    reproduce(Path(sys.argv[1]))
