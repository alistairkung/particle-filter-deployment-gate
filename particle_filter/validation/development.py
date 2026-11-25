"""Reproduce interval labels from the frozen development population only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulator.config import SECOND
from telemetry_sufficiency import CONDITIONS
from telemetry_sufficiency.custody import digest
from telemetry_sufficiency.parquet_evidence import rows

DEVELOPMENT = (
    Path(__file__).resolve().parents[2] / "telemetry_sufficiency/results/v1/development"
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
    from ..emissions import Context

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
    backlog_median: float


def fit_fold(training, replays):
    from ..emissions import fit_emission
    from .emission_diagnostics import training_thresholds
    from ..physical import Closure

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
        float(np.median([replays[r.key][-1].reading.pending_count for r in training])),
    )


def study_source_digest():
    """Bind study code and variant parameters by content and portable relative path."""
    from hashlib import sha256

    source = Path(__file__).resolve().parents[1]
    paths = []
    for directory, subdirs, files in source.walk():
        subdirs[:] = [
            d
            for d in subdirs
            if d not in {"results", "tests", "__pycache__"} and not d.startswith(".")
        ]
        paths.extend(directory / f for f in files if f.endswith((".py", ".toml")))
    combined = sha256()
    for path in sorted(paths):
        name = path.relative_to(source).as_posix().encode()
        combined.update(name + b"\0" + sha256(path.read_bytes()).digest())
    return combined.hexdigest()


def development_seeds(roots, seed):
    """Assign independent seeds to inventory slots before forecasts or scoring."""
    random = np.random.default_rng([seed, 0])
    return {r.key: int(random.bit_generator.random_raw()) for r in roots}


def _frozen_physical_files(repo):
    return [
        *sorted((repo / "simulator").glob("*.py")),
        repo / "feasibility_pilot/configs/pilot.toml",
        repo / "feasibility_pilot/experiments/config.py",
        repo / "feasibility_pilot/experiments/observations.py",
        repo / "telemetry_sufficiency/__init__.py",
        repo / "telemetry_sufficiency/plant.py",
        repo / "telemetry_sufficiency/results/v1/development/oof.parquet",
        repo / "particle_filter/physical.py",
        repo / "particle_filter/validation/development_replay.py",
    ]


def _model_fingerprint(repo, files, packages):
    from hashlib import sha256
    from importlib.metadata import version
    import platform

    combined = sha256()
    combined.update(f"Python {platform.python_version()}\n".encode())
    for package in packages:
        combined.update(f"{package} {version(package)}\n".encode())
    for path in sorted(files):
        name = path.relative_to(repo).as_posix().encode()
        combined.update(name + b"\0" + sha256(path.read_bytes()).digest())
    return combined.hexdigest()


def physical_digest():
    """Fingerprint the development replay, FIFO transition and matched future law."""
    repo = Path(__file__).resolve().parents[2]
    return _model_fingerprint(
        repo, _frozen_physical_files(repo), ("numpy", "simpy", "pyarrow")
    )


def model_digest():
    """Bind the emitting, fitting and forecasting model used in each fold."""
    repo = Path(__file__).resolve().parents[2]
    study = repo / "particle_filter"
    modules = [
        "emissions",
        "outbox_support",
        "emission_families",
        "inference",
        "forecast",
        "config",
        "validation/development",
        "validation/emission_diagnostics",
    ]
    files = _frozen_physical_files(repo) + [study / (name + ".py") for name in modules]
    files.extend(study.glob("configs/*.toml"))
    return _model_fingerprint(repo, files, ("numpy", "scipy", "simpy", "pyarrow"))
