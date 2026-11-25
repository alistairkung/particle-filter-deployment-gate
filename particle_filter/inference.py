"""Time-causal bootstrap inference and matched action-risk forecasts."""

from dataclasses import dataclass
import math

import numpy as np

from simulator.config import SECOND
from simulator.observations import INTERVAL, START, Reading, parse_history
from simulator.workload import rng

REGIMES = ("A", "B", "C")


def transition_matrix(half_life):
    if half_life not in (60, 120, 240):
        raise ValueError(
            "Expected a reviewed regime half-life of 60, 120 or 240 seconds"
        )
    q = -math.expm1(-math.log(2) * 5 / half_life)
    return np.array(((1 - q, q / 2, q / 2), (q, 1 - q, 0), (q, 0, 1 - q)))


def normalise_weights(log_mass):
    """Return normalised log weights, retaining exact zeros as negative infinity."""
    log_mass = np.asarray(log_mass, dtype=float)
    if log_mass.ndim != 1 or not len(log_mass):
        raise ValueError("Expected a nonempty vector of log masses")
    largest = log_mass.max()
    if np.isneginf(largest):
        raise ValueError("Delivered evidence has zero posterior mass")
    if not np.isfinite(largest):
        raise ValueError("Posterior has a nonfinite normaliser")
    shifted = log_mass - largest
    return shifted - np.log(np.exp(shifted).sum())


def systematic_ancestors(weights, unit_offset):
    """Select one equally spaced point per slot using a unit-uniform offset."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not weights.size or not 0 <= unit_offset < 1:
        raise ValueError("Invalid systematic resampling inputs")
    if (weights < 0).any() or not math.isclose(weights.sum(), 1.0):
        raise ValueError("Expected nonnegative normalised weights")
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    points = (np.arange(len(weights)) + unit_offset) / len(weights)
    return np.searchsorted(cumulative, points, side="right")


@dataclass(frozen=True)
class Update:
    end_ns: int
    evidence_intervals: tuple[int, ...]
    ess: float
    max_weight: float
    resampled: bool
    unique_ancestors: int
    regime_counts: tuple[int, ...]
    fallbacks: int
    regime_probabilities: tuple[float, ...]
    mean_work_s: float


@dataclass(frozen=True)
class Posterior:
    cutoff: int
    regimes: tuple[str, ...]
    boundaries: tuple
    weights: tuple[float, ...]
    cut_reading: Reading | None
    steps: tuple[Update, ...]

    @property
    def regime_probabilities(self):
        return self.steps[-1].regime_probabilities

    @property
    def mean_work_s(self):
        return self.steps[-1].mean_work_s


def _advance_population(
    boundaries,
    previous_regimes,
    regimes,
    prior,
    start,
    closure,
    physics_rngs,
    closure_rngs,
):
    from .emissions import Context
    from .physical import propagate

    fresh = prior is not None and prior.interval_end_ns == start
    ends, contexts, marks = [], [], []
    fallbacks = 0
    for i, boundary in enumerate(boundaries):
        work_s = boundary.work_ns / SECOND
        if fresh:
            boundary, fallback = closure.reconstruct(
                REGIMES[previous_regimes[i]],
                boundary.work_ns,
                prior.pending_count,
                closure_rngs[i],
                prior=boundary,
            )
            fallbacks += fallback
        regime = REGIMES[regimes[i]]
        advanced, completed = propagate(boundary, regime, physics_rngs[i])
        context = Context(regime, work_s, advanced.work_ns / SECOND, prior, start)
        ends.append(advanced)
        contexts.append(context)
        marks.append(completed)
    return ends, tuple(contexts), tuple(marks), fallbacks


def infer(
    history,
    closure,
    likelihood,
    *,
    particles,
    seed,
    half_life=120,
    on_prediction=None,
    cutoff=0,
):
    """Condition only on readings delivered through cutoff, on five-second steps.

    on_prediction receives the end tick, proposed contexts, *pre-evidence* weights,
    simulated mark counts and proposed end workloads (seconds) for development checks.
    """
    if type(particles) is not int or particles <= 0:
        raise ValueError("A positive integer particle count is required")
    if type(cutoff) is not int or cutoff not in range(START + INTERVAL, 1, INTERVAL):
        raise ValueError("Expected a five-second posterior boundary")
    available = parse_history(history, cutoff)
    from .physical import Boundary

    transition = transition_matrix(half_life).cumsum(axis=1)
    transition[:, -1] = 1
    regimes = rng(seed, "pf", "initial").integers(3, size=particles)
    boundaries = [Boundary()] * particles
    ancestors = np.arange(particles)
    log_weights = np.full(particles, -math.log(particles))
    physics_rngs = [rng(seed, "pf", "propagate", i) for i in range(particles)]
    closure_rngs = [rng(seed, "pf", "reconstruct", i) for i in range(particles)]
    contexts_by_end = {}
    steps = []
    prior = None
    for end in range(START + INTERVAL, cutoff + 1, INTERVAL):
        start = end - INTERVAL
        previous_regimes = regimes
        if start != START:
            uniforms = rng(seed, "pf", "regime", end).random(particles)
            regimes = (uniforms[:, None] >= transition[regimes]).sum(axis=1)
        boundaries, contexts, marks, fallbacks = _advance_population(
            boundaries,
            previous_regimes,
            regimes,
            prior,
            start,
            closure,
            physics_rngs,
            closure_rngs,
        )
        contexts_by_end[end] = contexts
        end_work = np.array([b.work_ns / SECOND for b in boundaries])
        if on_prediction is not None:
            on_prediction(end, contexts, np.exp(log_weights), marks, end_work.copy())

        delivered = [r for r in available if start < r.available_at_ns <= end]
        for reading in delivered:
            ancestral = contexts_by_end.pop(reading.interval_end_ns)
            scores = np.asarray(likelihood.log_likelihoods(ancestral, reading))
            if scores.shape != (particles,):
                raise ValueError("Expected one log likelihood per particle trajectory")
            log_weights = normalise_weights(log_weights + scores)
            if prior is None or reading.interval_end_ns > prior.interval_end_ns:
                prior = reading

        weights = np.exp(log_weights)
        ess = float(1 / np.dot(weights, weights))
        probabilities = np.bincount(regimes, weights, minlength=3)
        resample = end < cutoff and ess < particles / 2
        if resample:
            offset = rng(seed, "pf", "resample", end).random()
            indices = systematic_ancestors(weights, offset)
            regimes = regimes[indices]
            boundaries = [boundaries[i] for i in indices]
            ancestors = ancestors[indices]
            # Unassimilated readings must follow the selected trajectories.
            contexts_by_end = {
                tick: tuple(rows[i] for i in indices)
                for tick, rows in contexts_by_end.items()
            }
            log_weights = np.full(particles, -math.log(particles))
        steps.append(
            Update(
                end,
                tuple(r.interval_end_ns for r in delivered),
                ess,
                float(weights.max()),
                resample,
                len(np.unique(ancestors)),
                tuple(np.bincount(regimes, minlength=3)),
                fallbacks,
                tuple(probabilities),
                float(np.dot(weights, end_work)),
            )
        )
    cut_reading = (
        prior if prior is not None and prior.interval_end_ns == cutoff else None
    )
    return Posterior(
        cutoff,
        tuple(REGIMES[z] for z in regimes),
        tuple(boundaries),
        tuple(weights),
        cut_reading,
        tuple(steps),
    )


@dataclass(frozen=True)
class Forecast:
    hold_only: int
    promote_only: int
    both: int
    neither: int
    fallbacks: int

    @property
    def pairs(self):
        return self.hold_only + self.promote_only + self.both + self.neither

    @property
    def hold(self):
        return (self.hold_only + self.both) / self.pairs

    @property
    def promote(self):
        return (self.promote_only + self.both) / self.pairs

    @property
    def delta(self):
        return (self.promote_only - self.hold_only) / self.pairs


def forecast(posterior, closure, *, pairs, seed, work_by_regime=None):
    """Forecast from a weighted cut. Ablation accepts development cut work in ns."""
    from .physical import matched_pair

    if posterior.cutoff != 0:
        raise ValueError("Action risk forecasts require the decision-cut posterior")
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("A positive integer forecast-pair budget is required")
    if work_by_regime is not None:
        for regime in REGIMES:
            work = np.asarray(work_by_regime[regime])
            if (
                work.ndim != 1
                or not len(work)
                or work.dtype.kind not in "iu"
                or (work < 0).any()
            ):
                raise ValueError(
                    "Ablation requires nonnegative development cut work in ns per regime"
                )
    selected = rng(seed, "pf-forecast", "joint-draw").choice(
        len(posterior.weights),
        size=pairs,
        replace=True,
        p=posterior.weights,
    )
    cells = np.zeros((2, 2), dtype=int)
    fallbacks = 0
    for k, index in enumerate(selected):
        regime = posterior.regimes[index]
        prior = posterior.boundaries[index]
        pending = (
            prior.pending
            if posterior.cut_reading is None
            else posterior.cut_reading.pending_count
        )
        work = prior.work_ns
        if work_by_regime is not None:
            draws = work_by_regime[regime]
            i = rng(seed, "pf-forecast", "ablate", k).integers(len(draws))
            work = int(draws[i])
        boundary, fallback = closure.reconstruct(
            regime,
            work,
            pending,
            rng(seed, "pf-forecast", "reconstruct", k),
            prior=prior,
        )
        fallbacks += fallback
        hold, promote = matched_pair(
            boundary, regime, rng(seed, "pf-forecast", "future", k)
        )
        cells[hold, promote] += 1
    return Forecast(
        int(cells[1, 0]),
        int(cells[0, 1]),
        int(cells[1, 1]),
        int(cells[0, 0]),
        fallbacks,
    )
