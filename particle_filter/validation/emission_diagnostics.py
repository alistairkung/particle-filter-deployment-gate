"""Held-out one-step checks for the counts-first law, grouped by complete root."""

from collections import defaultdict

import numpy as np
from scipy.special import betainc, expit, log_expit, logsumexp

from simulator.observations import INTERVAL
from ..emissions import (
    BAND_UPPER,
    MAX_MARKS,
    REGIMES,
    _age_support,
    _band_mask,
    _context_columns,
    _mark_logmass,
    _nb_logs,
    _nb_positive,
    _delay_mask,
    _normal_interval,
)


def training_thresholds(samples):
    """Freeze per-regime strict upper-tail thresholds from training readings."""
    grouped = defaultdict(list)
    for context, reading in samples:
        grouped[context.regime].append(
            (5 * reading.publication_throughput_per_s, reading.pending_count)
        )
    return {
        regime: tuple(np.quantile(grouped[regime], 0.9, axis=0)) for regime in REGIMES
    }


def _pending_survival(k, log_excess, mean_log, size_log):
    """P(N > threshold) where k is the threshold minus the physical lower bound."""
    size, _, logq, log_positive = _nb_logs(mean_log, size_log)
    # NB positive-excess conditional survival is sf(k)/(1-pmf(0)) for k > 0.
    # P(N > L) and P(N > a value below L) are available without special functions.
    result = np.where(k < 0, 1.0, np.exp(log_excess))
    positive = k > 0
    if positive.any():
        sf = betainc(k[positive] + 1, size, np.exp(logq[positive]))
        with np.errstate(divide="ignore"):
            result[positive] = np.exp(
                log_excess[positive] + np.log(sf) - log_positive[positive]
            )
    return np.clip(result, 0.0, 1.0)


def _count_events(
    model, reg, base, prev, centre, scale, weights, mark_tail, backlog_tail
):
    total = np.zeros(4)
    # Bound the number of rows in the integration matrix. Sum over every physically
    # possible mark count, including the extrapolated zero-mark tail.
    for start in range(0, len(base), 16):
        part = slice(start, start + 16)
        cap = MAX_MARKS[reg[part]]
        marks = np.arange(cap.max() + 1)
        logmass = _mark_logmass(
            marks, centre[part, None], scale[part, None], cap[:, None]
        )
        mass = np.where(marks <= cap[:, None], np.exp(logmass), 0.0)
        mass *= weights[part, None]
        ncols = len(marks)
        grid = np.broadcast_to(marks, mass.shape).ravel()
        features = np.column_stack((np.repeat(base[part], ncols, axis=0), grid))
        x = model.designs[1].matrix(np.repeat(reg[part], ncols), features)
        lower = np.maximum(0, np.repeat(prev[part], ncols) - grid)
        log_excess = log_expit(-(x @ model.hurdle))
        mean_log = x @ model.nb[:-1]
        # A strictly positive lower bound rules out an empty outbox.
        empty = np.where(lower == 0, expit(x @ model.hurdle), 0.0).reshape(mass.shape)
        backlog = _pending_survival(
            np.floor(backlog_tail) - lower, log_excess, mean_log, model.nb[-1]
        ).reshape(mass.shape)
        high_marks = marks > mark_tail
        total += (
            mass[:, high_marks].sum(),
            (mass * empty).sum(),
            (mass * backlog).sum(),
            (mass[:, high_marks] * backlog[:, high_marks]).sum(),
        )
    return np.clip(total, 0.0, 1.0)


def _condition(weights_log, component):
    posterior = weights_log + component
    normalizer = logsumexp(posterior)
    if not np.isfinite(normalizer):
        raise ValueError("Observed conditional event has no predictive mass")
    return posterior - normalizer


def diagnostic_rows(model, contexts, reading, weights, thresholds, rng):
    """Evaluate one held-out interval under the predictive particle mixture.

    For a teacher-forced replay pass its single physical transition and weight one.
    For a sequential check pass the pre-update particle transitions and weights.
    Each returned pair is (observed statistic, predicted statistic). PIT rows
    store a randomised probability integral transform and its reference mean 1/2.
    """
    if not contexts or any(c.start_ns != reading.interval_start_ns for c in contexts):
        raise ValueError("Prediction needs aligned interval transitions")
    weights = np.asarray(weights, dtype=float)
    if (
        weights.shape != (len(contexts),)
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("Expected normalised predictive weights")
    reg, base, prev, prior_age = _context_columns(contexts)
    x = model.designs[0].matrix(reg, base)
    centre = x @ model.marks[:-3]
    scale = np.exp(model.marks[-3:][reg])
    marks = 5 * reading.publication_throughput_per_s
    pending = reading.pending_count
    mark_tail, backlog_tail = thresholds
    observed = (
        marks > mark_tail,
        pending == 0,
        pending > backlog_tail,
        marks > mark_tail and pending > backlog_tail,
    )
    events = _count_events(model, reg, base, prev, centre, scale, weights, *thresholds)
    rows = {
        k: (float(o), float(p))
        for k, o, p in zip(
            ("mark_tail", "empty", "backlog_tail", "upper_joint_tail"),
            observed,
            events,
            strict=True,
        )
    }

    cap = MAX_MARKS[reg]
    with np.errstate(divide="ignore"):
        logw = np.log(weights)
    logmarks = np.where(marks <= cap, _mark_logmass(marks, centre, scale, cap), -np.inf)
    if marks == 0:
        mark_left = np.zeros(len(reg))
    else:
        mark_left = np.exp(
            _normal_interval(-0.5, np.minimum(marks - 1, cap) + 0.5, centre, scale)
            - _normal_interval(-0.5, cap + 0.5, centre, scale)
        )
    rows["mark_pit"] = (
        float(weights @ (mark_left + rng.random() * np.exp(logmarks))),
        0.5,
    )
    logw = _condition(logw, logmarks)
    mark_weights = np.exp(logw)

    pending_features = np.column_stack((base, np.full(len(reg), marks)))
    x = model.designs[1].matrix(reg, pending_features)
    excess = pending - np.maximum(0, prev - marks)
    logit = x @ model.hurdle
    mean_log = x @ model.nb[:-1]
    logpending = np.full(len(reg), -np.inf)
    at_lower = excess == 0
    positive = excess > 0
    logpending[at_lower] = log_expit(logit[at_lower])
    logpending[positive] = log_expit(-logit[positive]) + _nb_positive(
        excess[positive], mean_log[positive], model.nb[-1]
    )
    pending_left = 1 - _pending_survival(
        excess - 1, log_expit(-logit), mean_log, model.nb[-1]
    )
    rows["pending_pit"] = (
        float(mark_weights @ (pending_left + rng.random() * np.exp(logpending))),
        0.5,
    )
    logw = _condition(logw, logpending)

    age = reading.oldest_pending_age_s
    band = int(np.searchsorted(BAND_UPPER, age))
    low, high, exact = _age_support(prev, marks, pending, prior_age)
    mask = _band_mask(low, high)
    timing_features = np.column_stack((pending_features, np.full(len(reg), pending)))
    probs = model.age.probabilities(model.designs[2].matrix(reg, timing_features), mask)
    deterministic = np.isfinite(exact)
    probs[deterministic] = np.eye(len(BAND_UPPER))[
        np.searchsorted(BAND_UPPER, exact[deterministic])
    ]
    for j, p in enumerate(np.exp(logw) @ probs):
        rows[f"age_band_{j}"] = (float(j == band), float(p))
    if marks:
        compatible = (
            (age >= low)
            & (age <= high + 1.0e-10)
            & (~deterministic | np.isclose(age, exact, atol=1.0e-10, rtol=0.0))
        )
        with np.errstate(divide="ignore"):
            logw = _condition(logw, np.log(np.where(compatible, probs[:, band], 0.0)))
        delay_band = int(np.searchsorted(BAND_UPPER, reading.completed_delay_median_s))
        x = model.designs[3].matrix(
            reg, np.column_stack((timing_features, np.full(len(reg), band)))
        )
        delay_probs = model.delay.probabilities(
            x, _delay_mask(reg, marks, prev, prior_age, age)
        )
        for j, p in enumerate(np.exp(logw) @ delay_probs):
            rows[f"delay_band_{j}_given_age_{band}"] = (
                float(j == delay_band),
                float(p),
            )
    return rows


def fold_diagnostics(records, *, seed, resamples=2000):
    """Whole-root resampling of frequency errors and lag-one centred PIT products.

    Supply (root, generating interval regime, interval start tick, diagnostic rows)
    records from one held-out fold. Report positive errors for underprediction.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for root, regime, start, rows in records:
        groups[regime][root].append((start, rows))
    rng = np.random.default_rng(seed)
    results = {}
    for regime, roots in sorted(groups.items()):
        by_root = []
        for intervals in roots.values():
            metrics = defaultdict(list)
            previous = None
            for start, rows in sorted(intervals, key=lambda item: item[0]):
                for key, pair in rows.items():
                    metrics[key].append(pair)
                if previous is not None and start == previous[0] + INTERVAL:
                    for key in ("mark_pit", "pending_pit"):
                        if key in previous[1] and key in rows:
                            lag = (rows[key][0] - 0.5) * (previous[1][key][0] - 0.5)
                            metrics[key + "_lag1"].append((lag, 0.0))
                previous = start, rows
            by_root.append(metrics)
        names = sorted(set().union(*(set(r) for r in by_root)))
        observed = np.zeros((len(roots), len(names)))
        predicted = np.zeros_like(observed)
        counts = np.zeros_like(observed)
        for i, root_metrics in enumerate(by_root):
            for j, key in enumerate(names):
                pairs = root_metrics.get(key)
                if pairs:
                    observed[i, j], predicted[i, j] = np.sum(pairs, axis=0)
                    counts[i, j] = len(pairs)
        draws = rng.multinomial(
            len(roots), np.full(len(roots), 1 / len(roots)), size=resamples
        )
        boot_counts = draws @ counts
        rates = np.full(boot_counts.shape, np.nan)
        np.divide(
            draws @ (observed - predicted),
            boot_counts,
            where=boot_counts > 0,
            out=rates,
        )
        report = {}
        for j, key in enumerate(names):
            n = counts[:, j].sum()
            represented = np.count_nonzero(counts[:, j])
            nonempty = rates[:, j][np.isfinite(rates[:, j])]
            interval = (
                np.quantile(nonempty, [0.025, 0.975]).tolist()
                if represented >= 2 and len(nonempty)
                else None
            )
            report[key] = {
                "observations": int(n),
                "roots": int(represented),
                "observed": float(observed[:, j].sum() / n),
                "predicted": float(predicted[:, j].sum() / n),
                "error_ci95": interval,
            }
        results[regime] = report
    return results


def repeatable_discrepancies(folds):
    """Flag repeatable development discrepancies for investigation before lock."""
    flags = defaultdict(list)
    for fold, report in enumerate(folds):
        for regime, metrics in report.items():
            for key, check in metrics.items():
                if check["error_ci95"] is None or key.endswith("_pit"):
                    continue
                lower, upper = check["error_ci95"]
                error = check["observed"] - check["predicted"]
                if key.endswith("_lag1"):
                    direction = (
                        "positive"
                        if lower > 0.0
                        else "negative"
                        if upper < 0.0
                        else None
                    )
                else:
                    direction = None
                    if abs(error) > 0.05:
                        if lower > 0.0:
                            direction = "underprediction"
                        elif upper < 0.0:
                            direction = "overprediction"
                if direction:
                    flags[regime, key, direction].append(fold)
    return [
        {"regime": r, "quantity": k, "direction": d, "folds": f}
        for (r, k, d), f in sorted(flags.items())
        if len(f) >= 2
    ]
