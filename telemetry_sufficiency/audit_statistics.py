"""Reconstruct root-paired descriptive and confirmatory statistics from locked forecasts."""

from itertools import combinations
from math import sqrt
from statistics import NormalDist

import numpy as np

from .audit import compare
from .audit_models import CONDITIONS, LEVELS

ACTIONS = ("hold", "promote")
RISK = (*ACTIONS, "increment")
NOMINAL = CONDITIONS[:2]


def claim(groups, candidate):
    """Stratified best-constant comparison with root as the variance unit."""
    constant = float(np.mean([y.mean() for y, _, _ in groups.values()]))
    gains, prior_gains, variances = [], [], []
    counts = {}
    for name, (y, prediction, prior) in groups.items():
        counts[name] = len(y)
        loss = (y - prediction) ** 2
        diffs = (y - constant) ** 2 - loss
        gains.append(float(diffs.mean()))
        prior_gains.append(float(np.mean((y - prior) ** 2 - loss)))
        variances.append(
            0.0 if np.min(diffs) == np.max(diffs) else float(diffs.var(ddof=1)) / len(y)
        )
    g = float(np.mean(gains))
    se = sqrt(sum(variances)) / len(groups)
    lower = g - NormalDist().inv_cdf(0.975) * se
    return dict(
        candidate=candidate,
        n=sum(counts.values()),
        counts=counts,
        constant=constant,
        gain=g,
        prior_gain=float(np.mean(prior_gains)),
        se=se,
        lower=lower,
        passes=bool(se > 0 and lower > 0),
    )


def _stratum(rows, surprise=False):
    prefix = "surprise_" if surprise else ""
    y = {a: np.array([r[prefix + a] for r in rows], float) for a in ACTIONS}
    y["increment"] = y["promote"] - y["hold"]
    if not surprise:
        for t in ("demand", "unfinished"):
            y[t] = np.array([r[t] for r in rows])
        y["condition"] = np.eye(3)[[CONDITIONS.index(r["condition"]) for r in rows]]
    predictions, loss = {}, {}
    for level in LEVELS:
        p = {
            t: np.array([r["predictions"][level][t] for r in rows])
            for t in y
            if level != "evaluator" or t in RISK
        }
        predictions[level] = p
        loss[level] = {t: (v - y[t]) ** 2 for t, v in p.items()}
        if "condition" in p:
            loss[level]["condition"] = loss[level]["condition"].sum(axis=1)
    return y, predictions, loss


def _calibration(predicted, events):
    if np.any((predicted < 0) | (predicted > 1)):
        raise ValueError("action forecast outside probability range")
    index = np.searchsorted([i / 10 for i in range(1, 10)], predicted, side="right")
    result = []
    for b in range(10):
        mask = index == b
        count = int(mask.sum())
        result.append(
            dict(
                count=count,
                mean_prediction=float(predicted[mask].mean()) if count else None,
                event_rate=float(events[mask].mean()) if count else None,
            )
        )
    return result


def _population(strata):
    y = {t: np.concatenate([s[0][t] for s in strata]) for t in strata[0][0]}
    h, p = y["hold"], y["promote"]
    cells = {}
    for name, a, b in (
        ("neither", 0, 0),
        ("promote_only", 0, 1),
        ("hold_only", 1, 0),
        ("both", 1, 1),
    ):
        cells[name] = int(np.sum((h == a) & (p == b)))
    result = dict(
        n=len(h),
        paired=cells,
        rates={a: float(np.mean([s[0][a].mean() for s in strata])) for a in RISK},
        models={},
    )
    for level in LEVELS:
        scores = {
            t: float(np.mean([s[2][level][t].mean() for s in strata])) for t in strata[0][2][level]
        }
        predictions = {t: np.concatenate([s[1][level][t] for s in strata]) for t in scores}
        model = dict(
            scores=scores, calibration={a: _calibration(predictions[a], y[a]) for a in ACTIONS}
        )
        if "condition" in scores:
            classes = np.argmax(y["condition"], axis=1) * 3 + np.argmax(
                predictions["condition"], axis=1
            )
            model["confusion"] = np.bincount(classes, minlength=9).reshape(3, 3).tolist()
            for t in ("demand", "unfinished"):
                scores[t + "_rmse"] = sqrt(scores[t])
        result["models"][level] = model
    return result


def _bootstrap_weights(counts, seed):
    """Express a draw as root multiplicities; reuse across every paired comparison."""
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = {}
    for name in CONDITIONS:
        n = counts[name]
        matrix = np.empty((10000, n), dtype=np.float64)
        # Generate the same row-major PCG64 draws as a single (10000,n) request.
        for start in range(0, 10000, 100):
            sampled = rng.integers(0, n, (100, n)) + (n * np.arange(100))[:, None]
            matrix[start : start + 100] = (
                np.bincount(sampled.ravel(), minlength=100 * n).reshape(100, n) / n
            )
        weights[name] = matrix
    return weights


def _estimates(weights, data):
    """Map a common whole-root resample to its mean for every outcome and loss."""
    truth, _, loss = data
    keys = [("observed", t) for t in (*RISK, "demand") if t in truth]
    values = [truth[t] for _, t in keys]
    for model in LEVELS:
        for target, value in loss[model].items():
            keys.append((model, target))
            values.append(value)
    for target in ("demand", "increment"):
        if target in truth:
            keys.append(("second_moment", target))
            values.append(truth[target] ** 2)
    columns = np.column_stack(values)
    means = dict(zip(keys, columns.mean(axis=0)))
    bootstrap = dict(zip(keys, (weights @ columns).T))
    for target in ("demand", "increment"):
        if target in truth:
            for source in (means, bootstrap):
                source["stratum_mean", target] = (
                    source["second_moment", target] - source["observed", target] ** 2
                )
    return means, bootstrap


def _interval(names, estimates, baseline, candidate, target):
    differences = []
    for part in range(2):
        differences.append(
            np.mean(
                [
                    estimates[n][part][baseline, target] - estimates[n][part][candidate, target]
                    for n in names
                ],
                axis=0,
            )
        )
    lower, upper = np.quantile(differences[1], [0.025, 0.975], method="linear")
    return dict(gain=float(differences[0]), lower=float(lower), upper=float(upper))


def _comparisons(populations, estimates):
    rows = []
    for population, names in populations.items():
        for target in (*RISK, "demand", "unfinished", "condition"):
            pairs = [("current", "history")]
            if target in RISK:
                pairs.append(("prior", "current"))
            if target == "increment":
                pairs.append(("prior", "history"))
            if len(names) == 1 and target in ("demand", "increment"):
                for level in LEVELS:
                    if level != "evaluator" or target == "increment":
                        pairs.append(("stratum_mean", level))
            for baseline, candidate in pairs:
                interval = _interval(names, estimates, baseline, candidate, target)
                rows.append(
                    dict(
                        population=population,
                        target=target,
                        baseline=baseline,
                        candidate=candidate,
                        **interval
                    )
                )
    return rows


def _surprise_change(ordinary, surprise, model, target):
    key = model, target
    lower, upper = np.quantile(surprise[1][key] - ordinary[1][key], [0.025, 0.975], method="linear")
    return dict(
        change=float(surprise[0][key] - ordinary[0][key]), lower=float(lower), upper=float(upper)
    )


def _overlap(groups):
    rows = []
    for left, right in combinations(CONDITIONS, 2):
        for metric, ends in (
            ("oldest_pending_age_s", [0, 0.1, 0.5, 1, 5, 15]),
            ("pending_count", [0, 4, 19, 99, 499]),
        ):
            histograms = []
            for condition in (left, right):
                values = [
                    max(r["history"], key=lambda h: h["interval_end_ns"])[metric]
                    for r in groups[condition]
                ]
                bins = np.searchsorted(ends, values, side="left")
                histograms.append(np.bincount(bins, minlength=len(ends) + 1) / len(values))
            rows.append(
                dict(
                    first=left,
                    second=right,
                    metric=metric,
                    overlap=float(np.minimum(*histograms).sum()),
                )
            )
    return rows


def audit_scores(rows, selected, reported, bootstrap_seed):
    """Rebuild the complete score table, comparison inventory and its bootstrap intervals."""
    groups = {
        n: sorted((r for r in rows if r["condition"] == n), key=lambda r: r["key"])
        for n in CONDITIONS
    }
    strata = {n: _stratum(group) for n, group in groups.items()}
    surprise = _stratum(groups["reference"], surprise=True)
    populations = dict(all=CONDITIONS, nominal_publisher=NOMINAL, **{n: (n,) for n in CONDITIONS})
    summaries = {p: _population([strata[n] for n in names]) for p, names in populations.items()}
    summaries["surprise"] = _population([surprise])
    compare(reported["populations"], summaries, "populations")
    claims = {}
    for target, names in (("demand", NOMINAL), ("increment", CONDITIONS)):
        candidate = selected[target]
        evidence = {
            n: (
                strata[n][0][target],
                strata[n][1][candidate][target],
                strata[n][1]["prior"][target],
            )
            for n in names
        }
        claims[target] = claim(evidence, candidate)
    compare(reported["claims"], claims, "claims")

    weights = _bootstrap_weights({n: len(v) for n, v in groups.items()}, bootstrap_seed)
    estimates = {n: _estimates(weights[n], s) for n, s in strata.items()}
    surprise_est = _estimates(weights["reference"], surprise)
    ordinary = estimates["reference"]
    changes = dict(
        rates={t: _surprise_change(ordinary, surprise_est, "observed", t) for t in RISK},
        scores={
            m: {t: _surprise_change(ordinary, surprise_est, m, t) for t in RISK} for m in LEVELS
        },
    )
    compare(reported["surprise_changes"], changes, "surprise changes")
    comparisons = _comparisons(populations, estimates)

    def order(row):
        return tuple(row[k] for k in ("population", "target", "baseline", "candidate"))

    compare(
        sorted(reported["comparisons"], key=order),
        sorted(comparisons, key=order),
        "paired intervals",
    )

    def overlap_order(row):
        return row["first"], row["second"], row["metric"]

    compare(
        sorted(reported["overlap"], key=overlap_order),
        sorted(_overlap(groups), key=overlap_order),
        "overlap",
    )
    verdict = "READY" if all(c["passes"] for c in claims.values()) else "READY WITH LIMITATIONS"
    return verdict, len(comparisons)
