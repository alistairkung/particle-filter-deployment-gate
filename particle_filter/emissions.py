"""Development-fitted conditional law of one delivered five-second outbox reading."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, log_expit, log_ndtr, logsumexp

from simulator.observations import SECOND, START, Reading


REGIMES = ("A", "B", "C")
MAX_MARKS = np.array([1251, 1251, 97])
# Band zero is the singleton {0}; subsequent intervals are open left, closed right.
BAND_UPPER = np.array([0.0, 0.025, 0.1, 0.5, 2.0, 5.0, 10.0, 20.0, np.inf])
RIDGE = 0.01


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


def _estimate(name, loss, initial, bounds=None):
    fit = minimize(
        loss,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 350, "ftol": 1.0e-10},
    )
    if not fit.success or not np.isfinite(fit.fun) or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"{name} fit failed: {fit.message}, objective {fit.fun}")
    return fit.x


def _log_difference(large, small):
    """Log(exp(large) - exp(small)); allow equality for exactly zero mass."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return large + np.log(-np.expm1(np.minimum(0.0, small - large)))


def _normal_interval(lo, hi, centre, scale):
    left = (lo - centre) / scale
    right = (hi - centre) / scale
    use_survival = left > 0
    a = np.where(use_survival, -left, right)
    b = np.where(use_survival, -right, left)
    return _log_difference(log_ndtr(a), log_ndtr(b))


def _mark_logmass(mark, centre, scale, cap):
    return _normal_interval(mark - 0.5, mark + 0.5, centre, scale) - _normal_interval(
        -0.5, cap + 0.5, centre, scale
    )


def _penalty(beta):
    return 0.5 * RIDGE * np.sum(beta[3:] ** 2)


def _fit_marks(x, regime, marks):
    intercept = [marks[regime == i].mean() for i in range(3)]
    spreads = [np.log(max(0.5, marks[regime == i].std())) for i in range(3)]
    initial = np.r_[intercept, np.zeros(x.shape[1] - 3), spreads]

    def loss(params):
        beta, spread = params[:-3], params[-3:]
        logp = _mark_logmass(marks, x @ beta, np.exp(spread[regime]), MAX_MARKS[regime])
        return -logp.mean() + _penalty(beta)

    return _estimate(
        "discretised normal marks",
        loss,
        initial,
        [(None, None)] * x.shape[1] + [(-3, 8)] * 3,
    )


def _nb_logs(mean_log, size_log):
    # Stable even when exp(mean_log) dwarfs exp(size_log) or vice versa.
    log_success = -np.logaddexp(0.0, mean_log - size_log)
    log_failure = -np.logaddexp(0.0, size_log - mean_log)
    size = np.exp(size_log)
    with np.errstate(divide="ignore"):
        log_positive = np.where(
            mean_log < -35.0 + min(0.0, size_log),
            mean_log,
            np.log(-np.expm1(size * log_success)),
        )
    return size, log_success, log_failure, log_positive


def _nb_positive(k, mean_log, size_log):
    size, logp, logq, log_positive = _nb_logs(mean_log, size_log)
    return (
        gammaln(k + size)
        - gammaln(size)
        - gammaln(k + 1)
        + size * logp
        + k * logq
        - log_positive
    )


def _fit_pending(x, excess):
    at_lower = excess == 0
    if not at_lower.any() or at_lower.all():
        raise ValueError(
            "Pending hurdle fit requires both hurdle and positive-excess observations"
        )

    def hurdle_loss(beta):
        return -log_expit(np.where(at_lower, 1, -1) * (x @ beta)).mean() + _penalty(
            beta
        )

    hurdle = _estimate("pending hurdle", hurdle_loss, np.zeros(x.shape[1]))
    positive = x[~at_lower]
    k = excess[~at_lower]

    def excess_loss(params):
        beta = params[:-1]
        return -_nb_positive(k, positive @ beta, params[-1]).mean() + _penalty(beta)

    nb = _estimate(
        "positive pending excess",
        excess_loss,
        np.zeros(x.shape[1] + 1),
        [(None, None)] * x.shape[1] + [(-8, 10)],
    )
    return hurdle, nb


def _age_support(count, marks, pending, previous_age):
    lower = np.where((count >= 0) & (marks < count), 5.0, 0.0)
    upper = np.where((count >= 0) & (marks >= count), 5.0, previous_age + 5.0)
    exact = np.where(
        pending == 0,
        0.0,
        np.where((count > 0) & (marks == 0), previous_age + 5.0, np.nan),
    )
    return lower, upper, exact


def _band_mask(lower, upper):
    left = np.r_[-np.inf, BAND_UPPER[:-1]]
    return (BAND_UPPER >= lower[:, None]) & (left < upper[:, None])


@dataclass
class _Ordinal:
    categories: np.ndarray
    cutpoints: np.ndarray
    beta: np.ndarray

    def probabilities(self, x, feasible, smoothing=0.001):
        # Ordered logistic band mass computed from a product of logistic tails,
        # avoiding cancellation when two neighbouring CDFs both round to one.
        count = len(self.categories)
        logp = np.full((len(x), len(BAND_UPPER)), -np.inf)
        if count == 1:
            logp[:, self.categories[0]] = 0.0
        elif count > 1:
            d = self.cutpoints[None, :] - (x @ self.beta)[:, None]
            logp[:, self.categories[0]] = log_expit(d[:, 0])
            logp[:, self.categories[-1]] = log_expit(-d[:, -1])
            logp[:, self.categories[1:-1]] = (
                log_expit(d[:, 1:])
                + log_expit(-d[:, :-1])
                + np.log(-np.expm1(-np.diff(self.cutpoints)))
            )
        logp[~feasible] = -np.inf
        norm = logsumexp(logp, axis=1)
        supported = np.isfinite(norm)
        fitted = np.zeros_like(logp)
        fitted[supported] = np.exp(logp[supported] - norm[supported, None])
        uniform = feasible / feasible.sum(axis=1)[:, None]
        return np.where(
            supported[:, None], (1 - smoothing) * fitted + smoothing * uniform, uniform
        )

    @classmethod
    def fit(cls, x, observed, mask):
        classes = np.unique(observed)
        # With an empty training component the fit cannot be accepted.
        if len(classes) == 0:
            raise ValueError("No stochastic timing observations to fit")
        if len(classes) == 1:
            return cls(classes, np.array([]), np.zeros(x.shape[1]))
        # First regime coefficient is fixed at zero to identify threshold location.
        ncuts = len(classes) - 1

        def model(params):
            cuts = np.r_[params[0], params[0] + np.cumsum(np.exp(params[1:ncuts]))]
            return cls(classes, cuts, np.r_[0.0, params[ncuts:]])

        def loss(params):
            law = model(params)
            probabilities = law.probabilities(x, mask, smoothing=0.0)
            return -np.log(
                probabilities[np.arange(len(x)), observed]
            ).mean() + _penalty(law.beta)

        initial = np.zeros(ncuts + x.shape[1] - 1)
        initial[0] = -2.0
        bounds = (
            [(None, None)]
            + [(-10, 5)] * (ncuts - 1)
            + [(None, None)] * (x.shape[1] - 1)
        )
        return model(_estimate("ordered timing bands", loss, initial, bounds))


def _reading_columns(readings):
    count = np.array([r.pending_count for r in readings])
    marks = np.array([r.publication_throughput_per_s * 5 for r in readings])
    age = np.array([r.oldest_pending_age_s for r in readings])
    delay = np.array([r.completed_delay_median_s for r in readings], dtype=float)
    return marks, count, age, delay


def _delay_minimum(regime, marks, previous_count):
    # At most the first mark can inherit an acknowledgement from before the
    # boundary. Each subsequent row incurs the full claim, acknowledgement and
    # mark cycle. Use the minimum possible median even if the inherited row
    # received a faster acknowledgement under the preceding regime.
    minimum = np.full(len(regime), 0.004)
    slow = regime == 2
    minimum[slow & (marks == 2)] = 0.028
    minimum[slow & ((marks >= 3) | (previous_count == 0))] = 0.052
    return minimum


def _delay_mask(regime, marks, previous_count, previous_age, oldest_age):
    minimum = np.maximum(
        _delay_minimum(regime, marks, previous_count), oldest_age - 5.0
    )
    return _band_mask(minimum, previous_age + 5.0)


def _physical_support(marks, pending, age, delay, regime, prev, prev_age):
    lower = np.maximum(0, prev - marks)
    low_age, high_age, exact = _age_support(prev, marks, pending, prev_age)
    # Measurements are produced on integer nanosecond ticks. Adding five seconds
    # to a previous floating-point age can differ by a few machine epsilons.
    same_age = np.isclose(age, exact, atol=0.1 / SECOND, rtol=0.0)
    valid = (
        (marks >= 0)
        & (marks <= MAX_MARKS[regime])
        & (marks == np.floor(marks))
        & (pending >= lower)
        & (pending == np.floor(pending))
        & np.isfinite(age)
        & (age >= low_age)
        & (age <= high_age + 1.0e-10)
        & (np.isnan(exact) | same_age)
        & np.where(
            marks == 0,
            np.isnan(delay),
            np.isfinite(delay)
            & (
                delay
                >= np.maximum(_delay_minimum(regime, marks, prev), age - 5.0) - 1.0e-10
            )
            & (delay <= prev_age + 5.0 + 1.0e-10),
        )
    )
    return valid, lower, exact, _band_mask(low_age, high_age)


@dataclass
class Emission:
    designs: tuple
    marks: np.ndarray
    hurdle: np.ndarray
    nb: np.ndarray
    age: _Ordinal
    delay: _Ordinal

    def _score(self, contexts, readings, smoothing):
        if any(
            c.start_ns != r.interval_start_ns
            for c, r in zip(contexts, readings, strict=True)
        ):
            raise ValueError("Misaligned proposed transition and measurement")
        reg, base, prev, prev_age = _context_columns(contexts)
        marks, pending, age, delay = _reading_columns(readings)
        valid, lower, exact, mask = _physical_support(
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
        mass = _mark_logmass(
            marks, x @ self.marks[:-3], np.exp(self.marks[-3:][reg]), MAX_MARKS[reg]
        )
        pending_features = np.column_stack((base, marks))
        x = self.designs[1].matrix(reg, pending_features)
        excess = pending - lower
        pos = excess > 0
        mass += log_expit(np.where(pos, -1, 1) * (x @ self.hurdle))
        mass[pos] += _nb_positive(excess[pos], x[pos] @ self.nb[:-1], self.nb[-1])
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
            _delay_mask(
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
    valid, lower, exact, mask = _physical_support(
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
    marks_fit = _fit_marks(matrix(base, all_rows), reg, marks)
    pending_features = np.column_stack((base, marks))
    hurdle, nb = _fit_pending(matrix(pending_features, all_rows), pending - lower)
    timing_features = np.column_stack((pending_features, pending))
    band = np.searchsorted(BAND_UPPER, age)
    stochastic = np.isnan(exact)
    age_fit = _Ordinal.fit(
        matrix(timing_features, stochastic), band[stochastic], mask[stochastic]
    )
    defined = marks > 0
    dx = matrix(np.column_stack((timing_features, band)), defined)
    delay_fit = _Ordinal.fit(
        dx,
        np.searchsorted(BAND_UPPER, delay[defined]),
        _delay_mask(
            reg[defined], marks[defined], prev[defined], prev_age[defined], age[defined]
        ),
    )
    return Emission(tuple(designs), marks_fit, hurdle, nb, age_fit, delay_fit)
