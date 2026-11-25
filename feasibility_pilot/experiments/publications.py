"""Paths reserved for the two independently verified scientific publications."""
from pathlib import Path

PILOT = Path(__file__).resolve().parents[1]
RETAINED = (
    PILOT / "results/frozen",
    PILOT / "results/review",
    PILOT / "FEASIBILITY_REPORT.md",
    PILOT / "plots",
    PILOT.parent / "telemetry_sufficiency/results/v1",
)


def retained(path):
    resolved = Path(path).resolve()
    return any(resolved.is_relative_to(publication) for publication in RETAINED)


def require_output(path):
    if retained(path):
        raise ValueError(f"Choose an output outside the retained publications: {path}")
