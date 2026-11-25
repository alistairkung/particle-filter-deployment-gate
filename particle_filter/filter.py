"""Time-causal bootstrap particle filtering."""

from dataclasses import dataclass
import math

import numpy as np

from simulator.config import SECOND
from simulator.observations import INTERVAL, START, Reading, parse_history
from simulator.workload import rng

from .model.regimes import REGIMES, transition_matrix


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
    # Count grid points below each CDF boundary. Keeping the offset separate
    # avoids rounding the final point of the half-open grid up to one.
    whole, fraction = np.divmod(cumulative * len(weights), 1)
    below = whole.astype(np.intp) + (fraction > unit_offset)
    return np.repeat(np.arange(len(weights)), np.diff(below, prepend=0))


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
    from .model.likelihood import Context
    from .model.physical import propagate

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
    from .model.physical import Boundary

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
        surviving_ancestors = ancestors if resample else ancestors[weights > 0]
        steps.append(
            Update(
                end,
                tuple(r.interval_end_ns for r in delivered),
                ess,
                float(weights.max()),
                resample,
                len(np.unique(surviving_ancestors)),
                tuple(np.bincount(regimes, minlength=3).tolist()),
                fallbacks,
                tuple(probabilities.tolist()),
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
