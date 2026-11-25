"""Private development labels, aligned to the interval that produced each reading."""

from dataclasses import dataclass, replace

from simulator.observations import Collector, INTERVAL, START, Reading
from simulator.replay import ObservedModel
from simulator.workload import admissions
from telemetry_sufficiency.plant import CONFIG, ONSET, raw_tape

from ..model.physical import Boundary, Closure, anonymous_boundary


@dataclass(frozen=True, slots=True)
class DevelopmentInterval:
    regime: str
    start: Boundary
    end: Boundary
    reading: Reading


def replay_development(condition: str, index: int) -> tuple[DevelopmentInterval, ...]:
    tape = raw_tape(condition, "development", index, "prefix")
    collector = Collector()
    model = ObservedModel(CONFIG, collector)
    model.schedule(admissions(tape, CONFIG.hold, CONFIG))
    previous = Boundary()
    intervals = []
    for end in range(START + INTERVAL, 1, INTERVAL):
        model.advance(end)
        boundary = anonymous_boundary(model)
        regime = (
            "A"
            if end <= ONSET
            else {"reference": "A", "contention": "B", "publisher": "C"}[condition]
        )
        intervals.append(
            DevelopmentInterval(regime, previous, boundary, collector.seal(end))
        )
        previous = boundary
        if condition == "publisher" and end == ONSET:
            model.c = replace(CONFIG, ack_ms=CONFIG.slow_ms)
    return tuple(intervals)


class PhaseDiagnostic:
    """Development-only reconstruction conditional on the revealed publisher phase."""

    def __init__(self, donors):
        groups = {phase: [] for phase in ("idle", "ack", "claim", "mark")}
        for regime, boundary in donors:
            groups[boundary.phase].append((regime, boundary))
        self.by_phase = {phase: Closure(records) for phase, records in groups.items()}

    def reconstruct(self, regime, work_ns, pending, phase, ack_residual_ns, rng):
        boundary, fallback = self.by_phase[phase].reconstruct(
            regime, work_ns, pending, rng
        )
        if fallback or boundary.phase != phase:
            return None
        return replace(boundary, ack_residual_ns=ack_residual_ns)
