"""Protected scoring. Each independent root retains all sibling outcomes and forecasts."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.special import ndtri

from .methods import ACTIONS, CONDITIONS, LEVELS, PHYSICAL
from simulator.observations import parse_history

MODELS = (*LEVELS, "evaluator")
RISK = (*ACTIONS, "increment")
BOOTSTRAPS = 10000


@dataclass
class Sample:
    rows: list
    target: dict
    prediction: dict
    loss: dict
    bootstrap_means: dict

    @classmethod
    def from_rows(cls, rows, surprise=False):
        prefix = "surprise_" if surprise else ""
        target = {a: np.array([r[prefix + a] for r in rows], dtype=float) for a in ACTIONS}
        if any(not np.isin(values, [0, 1]).all() for values in target.values()):
            raise ValueError("nonbinary protected endpoint")
        target["increment"] = target["promote"] - target["hold"]
        if not surprise:
            target.update({t: np.asarray([r[t] for r in rows], dtype=float) for t in PHYSICAL})
            target["condition"] = np.eye(3)[[CONDITIONS.index(r["condition"]) for r in rows]]
        prediction, loss = {}, {}
        for level in MODELS:
            for name, actual in target.items():
                if level == "evaluator" and name not in RISK:
                    continue
                values = np.asarray([r["predictions"][level][name] for r in rows], dtype=float)
                if values.shape != actual.shape or not np.isfinite(values).all():
                    raise ValueError(f"invalid protected prediction: {level}/{name}")
                if name in (*ACTIONS, "condition") and np.any((values < 0) | (values > 1)):
                    raise ValueError(f"probability outside unit interval: {level}/{name}")
                if name == "condition" and not np.allclose(values.sum(axis=1), 1., rtol=0, atol=1e-12):
                    raise ValueError(f"invalid class probabilities: {level}")
                error = (values - actual) ** 2
                prediction[level, name] = values
                loss[level, name] = error.sum(axis=1) if name == "condition" else error
                if not np.isfinite(loss[level, name]).all():
                    raise ArithmeticError(f"nonfinite protected loss: {level}/{name}")
            increment = prediction[level, "promote"] - prediction[level, "hold"]
            if not np.allclose(prediction[level, "increment"], increment, rtol=0, atol=1e-14):
                raise ValueError(f"inconsistent signed prediction: {level}")
        return cls(rows, target, prediction, loss, {})


def _mean(arrays):
    """Equally weight stratum means, holding the experimental mixture fixed."""
    return float(np.mean([a.mean() for a in arrays]))


def _calibration(samples, level, action):
    predicted = np.concatenate([s.prediction[level, action] for s in samples])
    actual = np.concatenate([s.target[action] for s in samples])
    bins = np.searchsorted([i / 10 for i in range(1, 10)], predicted, side="right")
    result = []
    for i in range(10):
        selected = bins == i
        count = int(selected.sum())
        result.append(dict(
            count=count,
            mean_prediction=float(predicted[selected].mean()) if count else None,
            event_rate=float(actual[selected].mean()) if count else None,
        ))
    return result


def _population(samples):
    h, p = (np.concatenate([s.target[a] for s in samples]) for a in ACTIONS)
    cells = {cell: int(np.sum((h == hold) & (p == promote)))
             for cell, hold, promote in (("neither", 0, 0), ("promote_only", 0, 1),
                                         ("hold_only", 1, 0), ("both", 1, 1))}
    models = {}
    for level in MODELS:
        scores = {t: _mean([s.loss[level, t] for s in samples])
                  for t in samples[0].target if (level, t) in samples[0].loss}
        for t in PHYSICAL:
            if t in scores:
                scores[t + "_rmse"] = float(np.sqrt(scores[t]))
        models[level] = dict(scores=scores, calibration={
            a: _calibration(samples, level, a) for a in ACTIONS
        })
        if "condition" in scores:
            predicted = np.concatenate([s.prediction[level, "condition"] for s in samples]).argmax(axis=1)
            actual = np.concatenate([s.target["condition"] for s in samples]).argmax(axis=1)
            models[level]["confusion"] = np.bincount(3 * actual + predicted, minlength=9).reshape(3, 3).tolist()
    return dict(n=len(h), paired=cells, rates={a: _mean([s.target[a] for s in samples]) for a in RISK},
                models=models)


def _claim(samples, candidate, target):
    constant = _mean([s.target[target] for s in samples])
    gains = [(s.target[target] - constant) ** 2 - s.loss[candidate, target] for s in samples]
    gain = _mean(gains)
    # Identical losses have exactly zero sample variance, including decimal-valued constants.
    variance_sum = sum(g.var(ddof=1) / len(g) for g in gains if np.ptp(g) != 0)
    se = float(np.sqrt(variance_sum) / len(samples))
    lower = float(gain - ndtri(.975) * se)
    prior_gain = _mean([s.loss["prior", target] - s.loss[candidate, target] for s in samples])
    counts = {s.rows[0]["condition"]: len(s.rows) for s in samples}
    if not np.isfinite([gain, se, lower, prior_gain]).all():
        raise ArithmeticError(f"nonfinite required claim calculation: {target}")
    return dict(candidate=candidate, n=sum(counts.values()), counts=counts, constant=constant,
                gain=gain, se=se, lower=lower, passes=se > 0 and lower > 0, prior_gain=prior_gain)


def score(rows, selected, bootstrap_seed):
    """Join locked forecasts to outcomes exactly once. No fitted parameters are modified."""
    if len(rows) != len({r["key"] for r in rows}):
        raise ValueError("duplicate protected key")
    if any(r["condition"] not in CONDITIONS for r in rows):
        raise ValueError("unknown protected condition")
    if set(selected) != {"demand", "increment"} or any(v not in LEVELS[1:] for v in selected.values()):
        raise ValueError("invalid locked candidates")
    ordinary = []
    for condition in CONDITIONS:
        group = sorted((r for r in rows if r["condition"] == condition), key=lambda r: r["key"])
        if len(group) < 2:
            raise ValueError(f"insufficient protected roots in {condition}")
        ordinary.append(Sample.from_rows(group))
    surprise = Sample.from_rows(ordinary[0].rows, surprise=True)
    populations = {"all": ordinary, "nominal_publisher": ordinary[:2]}
    populations.update({c: [s] for c, s in zip(CONDITIONS, ordinary)})
    populations["surprise"] = [surprise]
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    for i, sample in enumerate(ordinary):
        weights = _resample_weights(len(sample.rows), rng)
        _bootstrap(sample, weights)
        if i == 0:
            _bootstrap(surprise, weights)
    comparisons = []
    for name, samples in populations.items():
        if name != "surprise":
            comparisons.extend(_comparisons(name, samples))
    return dict(populations={k: _population(v) for k, v in populations.items()},
                claims={"demand": _claim(ordinary[:2], selected["demand"], "demand"),
                        "increment": _claim(ordinary, selected["increment"], "increment")},
                comparisons=comparisons, overlap=_overlap(ordinary),
                surprise_changes=_surprise_changes(ordinary[0], surprise))


def _resample_weights(n, rng):
    """Multiplicity of each root for each draw. All methods and siblings share a row."""
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    weights = np.zeros((BOOTSTRAPS, n), dtype=np.float64)
    np.add.at(weights, (np.arange(BOOTSTRAPS)[:, None], indices), 1.)
    return weights / n


def _bootstrap(sample, weights):
    columns = {**sample.loss, **{("outcome", t): sample.target[t] for t in RISK}}
    for t in ("demand", "increment"):
        if t in sample.target:
            columns["response", t] = sample.target[t]
            columns["square", t] = sample.target[t] ** 2
    means = weights @ np.column_stack(list(columns.values()))
    sample.bootstrap_means = dict(zip(columns, means.T))
    for t in ("demand", "increment"):
        if t in sample.target:
            # In each resample, MSE at that resample's own mean equals its variance.
            sample.bootstrap_means["stratum_mean", t] = (
                sample.bootstrap_means["square", t] - sample.bootstrap_means["response", t] ** 2)


def _interval(draws):
    lower, upper = np.quantile(draws, [.025, .975], method="linear")
    return dict(lower=float(lower), upper=float(upper))


def _comparisons(population, samples):
    plans = []
    for t in RISK:
        plans.extend(((t, "prior", "current"), (t, "current", "history")))
    plans.append(("increment", "prior", "history"))
    plans.extend((t, "current", "history") for t in (*PHYSICAL, "condition"))
    if population in CONDITIONS:
        for t in ("demand", "increment"):
            plans.extend((t, "stratum_mean", level) for level in MODELS if (level, t) in samples[0].loss)
    for target, baseline, candidate in plans:
        point, draws = [], []
        for s in samples:
            if baseline == "stratum_mean":
                source = (s.target[target] - s.target[target].mean()) ** 2
            else:
                source = s.loss[baseline, target]
            point.append(source - s.loss[candidate, target])
            draws.append(s.bootstrap_means[baseline, target] - s.bootstrap_means[candidate, target])
        yield dict(population=population, target=target, baseline=baseline, candidate=candidate,
                   gain=_mean(point), **_interval(np.mean(draws, axis=0)))


def _surprise_changes(reference, surprise):
    def change(group, target):
        if group == "outcome":
            values = surprise.target[target] - reference.target[target]
        else:
            values = surprise.loss[group, target] - reference.loss[group, target]
        draws = surprise.bootstrap_means[group, target] - reference.bootstrap_means[group, target]
        return dict(change=float(values.mean()), **_interval(draws))

    return dict(rates={t: change("outcome", t) for t in RISK},
                scores={level: {t: change(level, t) for t in RISK} for level in MODELS})


def _overlap(samples):
    histograms = {}
    for condition, sample in zip(CONDITIONS, samples):
        current = [parse_history(r["history"])[-1] for r in sample.rows]
        for metric, upper in (("oldest_pending_age_s", [0, .1, .5, 1, 5, 15]),
                              ("pending_count", [0, 4, 19, 99, 499])):
            values = [getattr(r, metric) for r in current]
            bins = np.searchsorted(upper, values, side="left")
            histograms[condition, metric] = np.bincount(bins, minlength=len(upper) + 1) / len(values)
    return [
        dict(first=a, second=b, metric=m, overlap=float(np.minimum(histograms[a, m], histograms[b, m]).sum()))
        for a, b in combinations(CONDITIONS, 2) for m in ("oldest_pending_age_s", "pending_count")
    ]
