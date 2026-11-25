"""Independently reproduce the pinned pilot in a new output directory."""

import os
import subprocess
import sys
import tempfile
import shutil
import time
from pathlib import Path

from . import PILOT
from .custody import file_hash, read, save


def reproduce(receipt):
    from .workflow import attempt

    receipt = Path(receipt).resolve()
    if receipt.is_relative_to(PILOT):
        raise ValueError("Study output path inside frozen pilot")
    if receipt.name != "pilot-reproduction.json":
        raise ValueError("Use the run directory's pilot-reproduction.json")
    from .archive import historical

    if historical(receipt.parent):
        raise ValueError("Use a fresh run directory for pilot reproduction")
    started = time.time_ns()
    archive = receipt.parent / "failures" / str(started)
    with attempt(receipt.parent, "reproduce", started=started) as manifest:
        with tempfile.TemporaryDirectory(prefix="aims5701-pilot-") as temporary:
            output = Path(temporary)
            try:
                _compare_fresh_pilot(receipt, output)
            except BaseException:
                if any(output.iterdir()):
                    archive.mkdir(parents=True, exist_ok=True)
                    shutil.move(output, archive / "pilot-output")
                raise
        manifest["status"] = "reproduce complete"


def _compare_fresh_pilot(receipt, output):
    from experiments.compare import compare

    frozen = PILOT / "results/frozen"
    commands = reproduction_commands(output)
    environment = dict(
        os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="-p no:cacheprovider"
    )
    for command in commands:
        subprocess.run(
            [sys.executable, "-m", *command], cwd=PILOT, check=True, env=environment
        )
    result = compare(frozen, output)
    verdict = read(output / "summary/verdict.json")["verdict"]
    if verdict != "GO":
        raise ValueError("Frozen pilot verdict changed")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "bb20ebe",
            "--",
            "configs",
            "simulator",
            "requirements.lock",
        ],
        cwd=PILOT,
        check=True,
    )
    comparison = dict(
        result,
        verdict=verdict,
        pilot_revision="bb20ebe",
        frozen_manifest=file_hash(frozen / "manifest.json"),
        original_receipt=file_hash(frozen / "summary/clean-reproduction.json"),
        commands=commands,
    )
    from .verification import verify_pilot_receipt

    verify_pilot_receipt(comparison)
    save(receipt, comparison)


def reproduction_commands(output):
    return [
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


if __name__ == "__main__":
    reproduce(Path(sys.argv[1]))
