"""Outbox telemetry sufficiency study, using the frozen feasibility plant."""

import sys
from pathlib import Path

CONDITIONS = ("reference", "contention", "publisher")

PILOT = Path(__file__).resolve().parents[1] / "feasibility_pilot"
# Historical pilot scoring and the independent audit import experiments.*.
sys.path.append(str(PILOT))
