"""Regularised component distributions for counts and ordered timing bands."""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import betainc, betaincc, gammaln, log_expit, log_ndtr, logsumexp

from .outbox_support import BAND_UPPER


RIDGE = 0.01


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


def normal_interval(lo, hi, centre, scale):
    left = (lo - centre) / scale
    right = (hi - centre) / scale
    use_survival = left > 0
    a = np.where(use_survival, -left, right)
    b = np.where(use_survival, -right, left)
    return _log_difference(log_ndtr(a), log_ndtr(b))


def mark_logmass(mark, centre, scale, cap):
    return normal_interval(mark - 0.5, mark + 0.5, centre, scale) - normal_interval(
        -0.5, cap + 0.5, centre, scale
    )


def _penalty(beta):
    return 0.5 * RIDGE * np.sum(beta[3:] ** 2)


def fit_marks(x, regime, marks, cap):
    intercept = [marks[regime == i].mean() for i in range(3)]
    spreads = [np.log(max(0.5, marks[regime == i].std())) for i in range(3)]
    initial = np.r_[intercept, np.zeros(x.shape[1] - 3), spreads]

    def loss(params):
        beta, spread = params[:-3], params[-3:]
        logp = mark_logmass(marks, x @ beta, np.exp(spread[regime]), cap)
        return -logp.mean() + _penalty(beta)

    return _estimate(
        "discretised normal marks",
        loss,
        initial,
        [(None, None)] * x.shape[1] + [(-3, 8)] * 3,
    )


def nb_log_parameters(mean_log, size_log):
    # Stable even when exp(mean_log) dwarfs exp(size_log) or vice versa.
    log_success = -np.logaddexp(0.0, mean_log - size_log)
    log_failure = -np.logaddexp(0.0, size_log - mean_log)
    size = np.exp(size_log)
    with np.errstate(divide="ignore"):
        log_positive = np.where(
            mean_log < -35.0 + np.minimum(0.0, size_log),
            mean_log,
            np.log(-np.expm1(size * log_success)),
        )
    return size, log_success, log_failure, log_positive


def _nb_logmass(k, size, logp, logq):
    return gammaln(k + size) - gammaln(size) - gammaln(k + 1) + size * logp + k * logq


def _nb_log_survival(k, size, logp, logq, log_positive):
    """Log P(count > k), including tails too small to represent in probability space."""
    if np.all(np.asarray(k) == 0):
        return log_positive
    k, size, logp, logq, log_positive = np.broadcast_arrays(
        k, size, logp, logq, log_positive
    )
    q = np.exp(logq)
    tail = np.array(betainc(k + 1, size, q))
    rounded = q == 1
    tail[rounded] = betaincc(size[rounded], k[rounded] + 1, np.exp(logp[rounded]))
    with np.errstate(divide="ignore"):
        out = np.where(k == 0, log_positive, np.log(tail))
    underflow = (k > 0) & (tail < np.finfo(float).tiny)
    out[underflow] = _nb_underflow_tail(
        k[underflow] + 1, size[underflow], logp[underflow], logq[underflow]
    )
    return out


def _nb_underflow_tail(first, size, logp, logq):
    """Sum the deep upper tail relative to its first mass."""
    q = np.exp(logq)
    term = np.ones(len(first))
    total = term.copy()
    count = first.copy()
    # Ratios converge monotonically to q. In this deep tail their upper bound
    # is below one, so a geometric series bounds all remaining terms.
    ratio = q * (count + size) / (count + 1)
    bound = np.maximum(ratio, q)
    while np.any(term * bound > np.finfo(float).eps * total * (1 - bound)):
        term *= ratio
        total += term
        count += 1
        ratio = q * (count + size) / (count + 1)
    return _nb_logmass(first, size, logp, logq) + np.log(total)


def positive_nb_logmass(k, mean_log, size_log, offset=0):
    """Log mass of excess k > 0 when the underlying count is k + offset."""
    size, logp, logq, log_positive = nb_log_parameters(mean_log, size_log)
    return _nb_logmass(k + offset, size, logp, logq) - _nb_log_survival(
        offset, size, logp, logq, log_positive
    )


@dataclass
class PendingExcess:
    hurdle_logit: np.ndarray
    mean_log: np.ndarray
    size_log: np.ndarray
    offset: np.ndarray

    def log_mass(self, excess):
        excess = np.asarray(excess)
        positive = excess > 0
        mass = np.where(excess == 0, log_expit(self.hurdle_logit), -np.inf)
        mass[positive] = log_expit(-self.hurdle_logit[positive]) + positive_nb_logmass(
            excess[positive],
            self.mean_log[positive],
            self.size_log[positive],
            self.offset[positive],
        )
        return mass

    def survival(self, threshold):
        threshold = np.broadcast_to(threshold, self.offset.shape)
        log_positive = log_expit(-self.hurdle_logit)
        probability = np.where(threshold < 0, 1.0, np.exp(log_positive))
        positive = threshold > 0
        size, logp, logq, log_tail = nb_log_parameters(
            self.mean_log[positive], self.size_log[positive]
        )
        offset = self.offset[positive]
        probability[positive] = np.exp(
            log_positive[positive]
            + _nb_log_survival(threshold[positive] + offset, size, logp, logq, log_tail)
            - _nb_log_survival(offset, size, logp, logq, log_tail)
        )
        return np.clip(probability, 0.0, 1.0)


def fit_pending(x, excess):
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
    nb = fit_positive_pending(
        "positive pending excess",
        x[~at_lower],
        excess[~at_lower],
        0,
        np.zeros(x.shape[1]),
    )
    return hurdle, nb


def fit_positive_pending(name, x, excess, offset, initial, *, intercepts=3):
    def loss(params):
        beta, size = params[:-1], params[-1]
        penalty = 0.5 * RIDGE * np.sum(beta[intercepts:] ** 2)
        return -positive_nb_logmass(excess, x @ beta, size, offset).mean() + penalty

    return _estimate(
        name, loss, np.r_[initial, 0.0], [(None, None)] * x.shape[1] + [(-8, 10)]
    )


@dataclass
class OrdinalBands:
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
        band_count = feasible.sum(axis=1)[:, None]
        uniform = np.divide(
            feasible, band_count, where=band_count > 0, out=np.zeros_like(fitted)
        )
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
