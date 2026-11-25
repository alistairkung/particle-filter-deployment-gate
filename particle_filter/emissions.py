"""Counts-first joint likelihood fitted on training intervals and scored once at delivery."""

from dataclasses import dataclass
import numpy as np
from scipy.special import log_expit

from simulator.observations import SECOND, START, Reading
from .outbox_support import (
    BAND_UPPER,
    MAX_MARKS,
    feasible_delay_bands,
    observation_support,
)
from .emission_families import (
    OrdinalBands,
    fit_marks,
    fit_pending,
    mark_logmass,
    positive_nb_logmass,
)

REGIMES = ("A", "B", "C")


@dataclass(frozen=True, slots=True)
class Context:
    regime: str
    work_start_s: float
    work_end_s: float
    prior: Reading | None
    start_ns: int


def _context_columns(contexts):
    regime = []
    features = []
    prior_count = []
    prior_age = []
    for c in contexts:
        if not (0 <= c.work_start_s < np.inf and 0 <= c.work_end_s < np.inf):
            raise ValueError("Expected finite nonnegative unfinished work")
        p = c.prior
        if p is not None and (
            p.available_at_ns > c.start_ns or p.interval_end_ns > c.start_ns
        ):
            raise ValueError("Prior reading unavailable at interval start")
        regime.append(REGIMES.index(c.regime))
        fresh = p is not None and p.interval_end_ns == c.start_ns
        prior_count.append(
            p.pending_count if fresh else (0 if c.start_ns == START else -1)
        )
        # Oldest possible row age at the start, including across a delivery gap.
        prior_age.append(
            p.oldest_pending_age_s + (c.start_ns - p.interval_end_ns) / SECOND
            if p is not None
            else (c.start_ns - START) / SECOND
        )
        features.append(
            (
                c.work_start_s,
                c.work_end_s,
                p.pending_count if p else 0,
                (c.start_ns - p.interval_end_ns) / SECOND if p else 0,
                int(p is None),
            )
        )
    base = np.array(features, dtype=float)
    base[:, :3] = np.log1p(base[:, :3])
    return np.array(regime), base, np.array(prior_count), np.array(prior_age)


@dataclass
class _Design:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features):
        scale = features.std(axis=0)
        return cls(features.mean(axis=0), np.where(scale > 1.0e-10, scale, 1.0))

    def matrix(self, regime, features):
        return np.column_stack((np.eye(3)[regime], (features - self.mean) / self.scale))


def _reading_columns(readings):
    count = np.array([r.pending_count for r in readings])
    marks = np.array([r.publication_throughput_per_s * 5 for r in readings])
    age = np.array([r.oldest_pending_age_s for r in readings])
    delay = np.array([r.completed_delay_median_s for r in readings], dtype=float)
    return marks, count, age, delay


@dataclass
class Emission:
    designs: tuple
    marks: np.ndarray
    hurdle: np.ndarray
    nb: np.ndarray
    age: OrdinalBands
    delay: OrdinalBands

    def _score(self, contexts, readings, smoothing):
        if any(
            c.start_ns != r.interval_start_ns
            for c, r in zip(contexts, readings, strict=True)
        ):
            raise ValueError("Misaligned proposed transition and measurement")
        reg, base, prev, prev_age = _context_columns(contexts)
        marks, pending, age, delay = _reading_columns(readings)
        valid, lower, exact, mask = observation_support(
            marks, pending, age, delay, reg, prev, prev_age
        )
        scores = np.full(len(contexts), -np.inf)
        if not valid.any():
            return scores
        reg, base, prev, prev_age, marks, pending, age, delay, lower, exact, mask = (
            v[valid]
            for v in (
                reg,
                base,
                prev,
                prev_age,
                marks,
                pending,
                age,
                delay,
                lower,
                exact,
                mask,
            )
        )
        x = self.designs[0].matrix(reg, base)
        mass = mark_logmass(
            marks, x @ self.marks[:-3], np.exp(self.marks[-3:][reg]), MAX_MARKS[reg]
        )
        pending_features = np.column_stack((base, marks))
        x = self.designs[1].matrix(reg, pending_features)
        excess = pending - lower
        pos = excess > 0
        mass += log_expit(np.where(pos, -1, 1) * (x @ self.hurdle))
        mass[pos] += positive_nb_logmass(
            excess[pos], x[pos] @ self.nb[:-1], self.nb[-1]
        )
        timing_features = np.column_stack((pending_features, pending))
        band = np.searchsorted(BAND_UPPER, age)
        stochastic = np.isnan(exact)
        x = self.designs[2].matrix(reg[stochastic], timing_features[stochastic])
        probs = self.age.probabilities(x, mask[stochastic], smoothing)
        mass[stochastic] += np.log(probs[np.arange(len(x)), band[stochastic]])
        defined = marks > 0
        x = self.designs[3].matrix(
            reg[defined], np.column_stack((timing_features[defined], band[defined]))
        )
        probs = self.delay.probabilities(
            x,
            feasible_delay_bands(
                reg[defined],
                marks[defined],
                prev[defined],
                prev_age[defined],
                age[defined],
            ),
            smoothing,
        )
        mass[defined] += np.log(
            probs[np.arange(len(x)), np.searchsorted(BAND_UPPER, delay[defined])]
        )
        scores[valid] = mass
        return scores

    def log_likelihoods(self, contexts, reading, *, smoothing=0.001):
        """Score the original transition for each particle once at delivery."""
        return self._score(contexts, [reading] * len(contexts), smoothing)

    def log_likelihood(self, context, reading, *, smoothing=0.001):
        return float(self.log_likelihoods([context], reading, smoothing=smoothing)[0])


def fit_emission(samples):
    """Fit on caller-supplied training intervals; retain only the conditional laws."""
    if not samples:
        raise ValueError("Empty development fit")
    contexts, readings = zip(*samples)
    reg, base, prev, prev_age = _context_columns(contexts)
    if set(reg) != {0, 1, 2}:
        raise ValueError("Training requires interval examples of all operating regimes")
    if any(c.start_ns != r.interval_start_ns for c, r in samples):
        raise ValueError("Misaligned development interval")
    marks, pending, age, delay = _reading_columns(readings)
    valid, lower, exact, mask = observation_support(
        marks, pending, age, delay, reg, prev, prev_age
    )
    if not valid.all():
        raise ValueError(
            f"Training contains infeasible observation at row {np.flatnonzero(~valid)[0]}"
        )
    designs = []

    def matrix(features, subset):
        design = _Design.fit(features[subset])
        designs.append(design)
        return design.matrix(reg[subset], features[subset])

    all_rows = np.ones(len(marks), dtype=bool)
    marks_fit = fit_marks(matrix(base, all_rows), reg, marks)
    pending_features = np.column_stack((base, marks))
    hurdle, nb = fit_pending(matrix(pending_features, all_rows), pending - lower)
    timing_features = np.column_stack((pending_features, pending))
    band = np.searchsorted(BAND_UPPER, age)
    stochastic = np.isnan(exact)
    age_fit = OrdinalBands.fit(
        matrix(timing_features, stochastic), band[stochastic], mask[stochastic]
    )
    defined = marks > 0
    dx = matrix(np.column_stack((timing_features, band)), defined)
    delay_fit = OrdinalBands.fit(
        dx,
        np.searchsorted(BAND_UPPER, delay[defined]),
        feasible_delay_bands(
            reg[defined], marks[defined], prev[defined], prev_age[defined], age[defined]
        ),
    )
    return Emission(tuple(designs), marks_fit, hurdle, nb, age_fit, delay_fit)
