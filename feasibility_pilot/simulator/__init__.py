"""Resolve shared modules for historical commands run inside feasibility_pilot/."""

from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[2] / "simulator"))
