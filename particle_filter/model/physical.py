"""Hypothetical FIFO workload and anonymous publisher obligations."""

from collections import defaultdict
from dataclasses import dataclass, replace

from telemetry_sufficiency import (
    PILOT,
)  # Registers the frozen pilot experiment imports.
from experiments.config import read_config
from experiments.observations import breaches, score_window
from simulator.config import MS, SECOND
from simulator.model import CLAIM, MARK, QUERY, WRITE, Model, Write
from simulator.workload import admissions, primitive_tape

CONFIG = read_config(PILOT / "configs/pilot.toml")


@dataclass(frozen=True, slots=True)
class Visit:
    kind: int
    remaining_ns: int


@dataclass(frozen=True, slots=True)
class Boundary:
    pending: int = 0
    visits: tuple[Visit, ...] = ()
    phase: str = "idle"
    ack_residual_ns: int = 0

    @property
    def work_ns(self):
        return sum(v.remaining_ns for v in self.visits)


def anonymous_boundary(model: Model) -> Boundary:
    """Capture admitted residual visits after events at the boundary, in FIFO order."""
    tick = model.env.now
    outstanding = []
    for kind, _, _, start, end in reversed(model.db.visits):
        if end <= tick:
            break
        outstanding.append(Visit(kind, end - max(tick, start)))
    residual = model.ack_due - tick if model.ack_due is not None else 0
    return Boundary(
        len(model.pending), tuple(reversed(outstanding)), model.phase, residual
    )


def _work_band(work_ns):
    if work_ns == 0:
        return 0
    if work_ns <= 25 * MS:
        return 1
    if work_ns <= 100 * MS:
        return 2
    return 3


@dataclass(frozen=True, slots=True)
class _Donor:
    boundary: Boundary
    floors: tuple[int, ...]
    flexible: tuple[int, ...]
    minimum_ns: int

    @classmethod
    def prepare(cls, boundary):
        floors = []
        flexible = []
        for i, visit in enumerate(boundary.visits):
            if visit.kind in (CLAIM, MARK):
                floors.append(visit.remaining_ns)
            else:
                floors.append(
                    CONFIG.insert_ms * MS if visit.kind == WRITE and i > 0 else 1
                )
                flexible.append(i)
        return cls(boundary, tuple(floors), tuple(flexible), sum(floors))

    def fits(self, work_ns):
        return (
            self.minimum_ns == work_ns
            or self.minimum_ns < work_ns
            and bool(self.flexible)
        )

    def allocate(self, work_ns, pending):
        if self.boundary.work_ns == work_ns:
            return replace(self.boundary, pending=pending)
        sizes = list(self.floors)
        excess = work_ns - self.minimum_ns
        weights = [
            self.boundary.visits[i].remaining_ns - sizes[i] for i in self.flexible
        ]
        if not any(weights):
            weights = [1] * len(weights)
        denominator = sum(weights)
        remainders = []
        for i, weight in zip(self.flexible, weights):
            whole, remainder = divmod(excess * weight, denominator)
            sizes[i] += whole
            remainders.append((-remainder, i))
        for _, i in sorted(remainders)[: work_ns - sum(sizes)]:
            sizes[i] += 1
        visits = tuple(
            Visit(v.kind, size) for v, size in zip(self.boundary.visits, sizes)
        )
        return replace(self.boundary, pending=pending, visits=visits)


def _draw_compatible(pool, work_ns, rng):
    if not pool:
        return None
    # Rejection samples uniformly from compatible donors. Bound its cost in sparse cells.
    for _ in range(8):
        donor = pool[int(rng.integers(len(pool)))]
        if donor.fits(work_ns):
            return donor
    compatible = [d for d in pool if d.fits(work_ns)]
    return compatible[int(rng.integers(len(compatible)))] if compatible else None


class Closure:
    """Development-calibrated conditional law of omitted publisher and FIFO details."""

    def __init__(self, donors):
        self.cells = defaultdict(list)
        self.positive_work = defaultdict(list)
        self.idle_by_band = defaultdict(list)
        for regime, boundary in donors:
            donor = _Donor.prepare(boundary)
            band = _work_band(boundary.work_ns)
            pending = boundary.pending > 0
            self.cells[regime, pending, band].append(donor)
            if band:
                self.positive_work[regime, pending].append(donor)
                if not pending:
                    self.idle_by_band[band].append(donor)

    def reconstruct(self, regime, work_ns, pending, rng, *, prior=None):
        """Return anonymous obligations compatible with the supplied state and fallback flag."""
        if regime not in ("A", "B", "C") or work_ns < 0 or pending < 0:
            raise ValueError("Invalid reconstruction regime, work or pending count")
        if pending == 0 and work_ns == 0:
            return Boundary(), False
        band = _work_band(work_ns)
        has_pending = pending > 0
        pools = [self.cells.get((regime, has_pending, band))]
        if band:
            pools.append(self.positive_work.get((regime, has_pending)))
            if not has_pending:
                pools.append(self.idle_by_band.get(band))
        for pool in pools:
            donor = _draw_compatible(pool, work_ns, rng)
            if donor is not None:
                return donor.allocate(work_ns, pending), False

        visits = (Visit(QUERY, work_ns),) if work_ns else ()
        if not has_pending:
            return Boundary(0, visits), True
        duration = (CONFIG.slow_ms if regime == "C" else CONFIG.ack_ms) * MS
        residual = (
            prior.ack_residual_ns
            if prior and prior.phase == "ack"
            else int(rng.integers(1, duration + 1))
        )
        return Boundary(pending, visits, "ack", residual), True


def _restore(boundary, regime, config):
    """Install anonymous inherited visits once, using an independent simulated clock."""
    if regime not in ("A", "B", "C"):
        raise ValueError("Unsupported regime")
    config = replace(config, ack_ms=config.slow_ms) if regime == "C" else config
    model = Model(config, 0)
    # Negative identities denote inherited rows and writes; future identifiers start at zero.
    for i in range(1, boundary.pending + 1):
        model.pending.append(-i)
        model.writes[-i] = Write(0, False, 0, commit=0)
    model.phase = boundary.phase
    for i, visit in enumerate(boundary.visits, start=1):
        identity = -boundary.pending - i
        if visit.kind == WRITE:
            model.writes[identity] = Write(0, False, visit.remaining_ns)

            def callback(_, key=identity):
                model._commit(key)
        elif visit.kind == QUERY:
            callback = None
        elif visit.kind == CLAIM:
            identity = model.pending[0]
            callback = model._await_sink
        elif visit.kind == MARK:
            identity = model.pending[0]
            callback = model._finish_mark
        else:
            raise ValueError("Unsupported inherited visit")
        model.db.visit(visit.kind, identity, visit.remaining_ns, callback)
    if boundary.phase == "ack":
        model.ack_due = boundary.ack_residual_ns
        model.env.timeout(model.ack_due).callbacks.append(model._request_mark)
    return model


@dataclass(frozen=True, slots=True)
class ReplayResult:
    end: Boundary
    marks: int
    eligible: tuple[int, ...]
    misses: tuple[int, ...]
    breach: bool


def replay_hypothesis(boundary, regime, tape, *, share, horizon_ns, config=CONFIG):
    """Replay a reconstructed cut or interval with fresh primitives and the frozen FIFO."""
    model = _restore(boundary, regime, config)
    model.schedule(admissions(tape, share, config))
    model.advance(horizon_ns)
    limits = [0] * (config.horizon // config.window)
    misses = limits.copy()
    scored = horizon_ns == config.horizon * SECOND
    marks = 0
    for write in model.writes.values():
        marks += write.mark is not None
        if (
            scored
            and (w := score_window(write.arrival, write.eligible, config)) is not None
        ):
            limits[w] += 1
            misses[w] += (
                write.commit is None
                or write.commit > write.arrival + config.deadline_ms * MS
            )
    breach = any(
        breaches(late, count, config.miss_fraction)
        for late, count in zip(misses, limits)
    )
    return ReplayResult(
        anonymous_boundary(model), marks, tuple(limits), tuple(misses), breach
    )


def _fresh_tape(regime, rng, seconds, config):
    return primitive_tape(
        replace(config, horizon=seconds),
        ("particle-filter", int(rng.bit_generator.random_raw())),
        "future",
        reference_rate=config.reference,
        extra_rate=config.extra if regime == "B" else 0,
    )


def propagate(boundary, regime, rng, *, config=CONFIG):
    tape = _fresh_tape(regime, rng, 5, config)
    result = replay_hypothesis(
        boundary, regime, tape, share=config.hold, horizon_ns=5 * SECOND, config=config
    )
    return result.end, result.marks


def matched_pair(boundary, regime, rng, *, config=CONFIG):
    """Reconstruct once upstream, then use common primitive inputs for both actions."""
    tape = _fresh_tape(regime, rng, config.horizon, config)

    def breach(share):
        return int(
            replay_hypothesis(
                boundary,
                regime,
                tape,
                share=share,
                horizon_ns=config.horizon * SECOND,
                config=config,
            ).breach
        )

    return breach(config.hold), breach(config.promote)
