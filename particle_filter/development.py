"""Reproduce interval labels from the frozen development population only."""

from dataclasses import dataclass
from pathlib import Path

from simulator.config import SECOND
from telemetry_sufficiency import CONDITIONS
from telemetry_sufficiency.custody import digest
from telemetry_sufficiency.parquet_evidence import rows

DEVELOPMENT = (
    Path(__file__).resolve().parents[1] / "telemetry_sufficiency/results/v1/development"
)


@dataclass(frozen=True)
class DevelopmentRoot:
    key: str
    condition: str
    index: int
    fold: int
    hold: int
    promote: int
    cut_work_s: float

    @property
    def panel(self):
        """First 12 ordinary development indices in each condition, before observing fits."""
        return self.index < 12


def development_inventory():
    records = {r["key"]: r for r in rows(DEVELOPMENT, "oof")}
    roots = []
    for condition in CONDITIONS:
        for index in range(400):
            key = digest(f"envelope-v1/{condition}/development/{index}".encode())[:32]
            record = records.pop(key)
            if record["condition"] != condition or record["fold"] not in range(5):
                raise ValueError("development fold or condition mismatch")
            roots.append(
                DevelopmentRoot(
                    key,
                    condition,
                    index,
                    record["fold"],
                    record["hold"],
                    record["promote"],
                    record["unfinished"],
                )
            )
    if records:
        raise ValueError("unexpected development roots")
    return roots


def replay_root(root):
    from .development_replay import replay_development

    intervals = replay_development(root.condition, root.index)
    if intervals[-1].end.work_ns / SECOND != root.cut_work_s:
        raise ArithmeticError(
            f"cut-work replay disagreement for development root {root.key}"
        )
    return intervals


def training_samples(replays):
    from .emissions import Context

    for intervals in replays:
        previous = None
        for interval in intervals:
            reading = interval.reading
            context = Context(
                interval.regime,
                interval.start.work_ns / SECOND,
                interval.end.work_ns / SECOND,
                previous,
                reading.interval_start_ns,
            )
            yield context, reading
            previous = reading


@dataclass
class FoldFit:
    likelihood: object
    closure: object
    cut_work_ns: dict
    thresholds: dict
    training_digest: str


def fit_fold(training, replays):
    from .emissions import fit_emission
    from .emission_diagnostics import training_thresholds
    from .physical import Closure

    samples = list(training_samples(replays[r.key] for r in training))
    donors = ((i.regime, i.end) for r in training for i in replays[r.key])
    work = {z: [] for z in "ABC"}
    for r in training:
        cut = replays[r.key][-1]
        work[cut.regime].append(cut.end.work_ns)
    return FoldFit(
        fit_emission(samples),
        Closure(donors),
        work,
        training_thresholds(samples),
        digest("\n".join(sorted(r.key for r in training)).encode()),
    )


def study_source_digest():
    from hashlib import sha256

    source = Path(__file__).parent
    combined = sha256()
    for path in sorted(source.glob("*.py")):
        combined.update(path.name.encode() + b"\0" + sha256(path.read_bytes()).digest())
    return combined.hexdigest()
