"""Locked outbox diagnostics. The public prediction path consumes delivered history alone."""

import hashlib
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from .observation import parse_history

CONDITIONS = ("reference", "contention", "publisher")
LEVELS = ("prior", "age", "current", "history")
METRICS = (
    "pending_count", "oldest_pending_age_s",
    "publication_throughput_per_s", "completed_delay_median_s",
)
ACTIONS = ("hold", "promote")
PHYSICAL = ("demand", "unfinished")
PENALTY = 0.01


def features(history, level):
    """Return the ordered public design row, with explicit None before training imputation."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    observed = parse_history(history)
    if level == "prior":
        return {}
    current = observed[-1] if observed else None
    metrics = ("oldest_pending_age_s",) if level == "age" else METRICS
    row = {}
    flags = {}

    def numeric(name, value, log=False):
        flags[f"undefined_{name}"] = int(value is None)
        row[name] = math.log1p(value) if log and value is not None else value

    for metric in metrics:
        numeric("current_" + metric, getattr(current, metric, None), log=True)
    if level == "history":
        window = [r for r in observed if -30 * 10**9 < r.interval_end_ns <= 0]
        for metric in metrics:
            defined = [r for r in window if getattr(r, metric) is not None]
            values = [getattr(r, metric) for r in defined]
            mean = sum(values) / len(values) if values else None
            slope = None
            if len(values) >= 2:
                times = [r.interval_end_ns / 10**9 for r in defined]
                center = sum(times) / len(times)
                numerator = sum((t - center) * (v - mean) for t, v in zip(times, values))
                slope = numerator / sum((t - center) ** 2 for t in times)
            numeric("mean_" + metric, mean, log=True)
            numeric("slope_" + metric, slope)
            row["defined_" + metric] = len(defined)
        row["deliveries"] = len(window)
    return row | flags


SOLVER = {
    "method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000,
    "penalty": PENALTY, "dtype": "float64", "class_order": list(CONDITIONS),
    "positive_target_clip": "max(0,prediction)", "initialization": "zeros",
    "linear": "unpenalised intercept; solve ridge normal equations",
    "multiclass": "symmetric logits, centred intercepts",
}


def _evaluator_features(condition, demand, unfinished):
    if condition not in CONDITIONS or not all(math.isfinite(x) and x >= 0 for x in (demand, unfinished)):
        raise ValueError("invalid evaluator diagnostic variables")
    return {c: int(c == condition) for c in CONDITIONS} | {"demand": demand, "unfinished": unfinished}


def _fit_model(design, response, target):
    """Fit a single authorised response against a numeric matrix with leading intercept."""
    y = np.asarray(response, dtype=float)
    multiclass = target == "condition"
    if multiclass:
        y = np.eye(len(CONDITIONS))[y.astype(int)]
    constant_response = np.all(y == y[0])
    if design.shape[1] == 1 or constant_response:
        value = y[0] if constant_response else y.mean(axis=0)
        return dict(kind="constant", intercept=value.tolist(), coef=[],
                    converged=True, iterations=0)

    n = len(y)
    if target in PHYSICAL:
        normal = design.T @ design / n
        normal[1:, 1:] += PENALTY * np.eye(design.shape[1] - 1)
        parameters = np.linalg.solve(normal, design.T @ y / n)
        iterations, kind = 0, "linear"
    else:
        shape = (design.shape[1], len(CONDITIONS)) if multiclass else (design.shape[1],)

        def objective(weights):
            coefficients = weights.reshape(shape)
            logits = design @ coefficients
            if multiclass:
                loss = np.mean(logsumexp(logits, axis=1) - np.sum(y * logits, axis=1))
                residual = softmax(logits, axis=1) - y
            else:
                loss = np.mean(np.logaddexp(0., logits) - y * logits)
                residual = expit(logits) - y
            gradient = design.T @ residual / n
            gradient[1:] += PENALTY * coefficients[1:]
            loss += PENALTY / 2 * np.sum(coefficients[1:] ** 2)
            return loss, gradient.ravel()

        result = minimize(objective, np.zeros(shape).ravel(), method="L-BFGS-B", jac=True,
                          options={k: SOLVER[k] for k in ("ftol", "gtol", "maxiter")})
        if not result.success:
            raise ArithmeticError(f"required {target} fit failed: {result.message}")
        parameters = result.x.reshape(shape)
        if multiclass:
            parameters[0] -= parameters[0].mean()
        iterations, kind = int(result.nit), "multiclass" if multiclass else "binary"
    if not np.isfinite(parameters).all():
        raise ArithmeticError(f"nonfinite coefficients for {target}")
    return dict(kind=kind, intercept=parameters[0].tolist(), coef=parameters[1:].tolist(),
                converged=True, iterations=iterations)


def _fit_level(feature_names, matrix, responses):
    medians = []
    for column in matrix.T:
        defined = column[~np.isnan(column)]
        medians.append(float(np.median(defined)) if len(defined) else 0.)
    values = np.where(np.isnan(matrix), medians, matrix)
    means = values.mean(axis=0)
    scales = np.where(np.ptp(values, axis=0) == 0, 1., values.std(axis=0))
    design = np.column_stack((np.ones(len(values)), (values - means) / scales))
    return dict(features=feature_names, medians=medians, means=means.tolist(), scales=scales.tolist(),
                fits={t: _fit_model(design, y, t) for t, y in responses.items()})


def _predict_level(model, row):
    if list(row) != model["features"]:
        raise ValueError("locked feature names or order differ from prediction inputs")
    x = np.asarray(list(row.values()), dtype=float)
    x = (np.where(np.isnan(x), model["medians"], x) - model["means"]) / model["scales"]
    outputs = {}
    for target, fit in model["fits"].items():
        value = np.asarray(fit["intercept"])
        if fit["kind"] != "constant":
            value = value + x @ np.asarray(fit["coef"])
        if fit["kind"] == "binary":
            value = expit(value)
        elif fit["kind"] == "multiclass":
            value = softmax(value)
        elif target in PHYSICAL:
            value = np.maximum(0., value)
        if not np.isfinite(value).all():
            raise ArithmeticError(f"nonfinite prediction for {target}")
        outputs[target] = value.tolist()
    outputs["increment"] = outputs["promote"] - outputs["hold"]
    return outputs


def predict(models, history):
    """Produce every outbox forecast using the lock and public evidence alone."""
    return {level: _predict_level(models[level], features(history, level)) for level in LEVELS}


def predict_evaluator(evaluator_models, condition, demand, unfinished):
    """Score the private action diagnostic from its separately authorised five variables."""
    return _predict_level(evaluator_models, _evaluator_features(condition, demand, unfinished))


def _fold_assignments(rows, fold_seed):
    folds = {}
    for condition in CONDITIONS:
        keys = [r["key"] for r in rows if r["condition"] == condition]
        if len(keys) < 5:
            raise ValueError(f"fewer than five development roots in {condition}")
        keys.sort(key=lambda k: hashlib.sha256(f"{fold_seed}/{k}".encode()).digest())
        folds.update({key: i % 5 for i, key in enumerate(keys)})
    return folds


def _selection(oof):
    scores, selected = {}, {}
    for claim in ("demand", "increment"):
        scores[claim] = {}
        conditions = CONDITIONS[:2] if claim == "demand" else CONDITIONS
        strata = [[r for r in oof if r["targets"]["condition"] == c] for c in conditions]
        for level in LEVELS[1:]:
            def loss(entry):
                truth = entry["targets"]
                observed = truth["demand"] if claim == "demand" else truth["promote"] - truth["hold"]
                return (observed - entry["predictions"][level][claim]) ** 2

            scores[claim][level] = float(np.mean([np.mean([loss(r) for r in group]) for group in strata]))
        selected[claim] = min(LEVELS[1:], key=scores[claim].get)
    return scores, selected


def fit_development(rows, fold_seed):
    """Custody adapter: preflight development envelopes, separate inputs, fit and select."""
    required = {"key", "partition", "condition", "history", *PHYSICAL, *ACTIONS}
    if any(set(r) != required or r["partition"] != "development" for r in rows):
        raise ValueError("fit requires development-only envelopes with the exact schema")
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("duplicate development key")
    rows = sorted(rows, key=lambda r: r["key"])
    responses = {
        target: np.asarray([r[target] for r in rows], dtype=float)
        for target in (*ACTIONS, *PHYSICAL)
    }
    if any(not np.isin(responses[a], [0, 1]).all() for a in ACTIONS):
        raise ValueError("expected binary development action endpoints")
    responses["condition"] = np.asarray([CONDITIONS.index(r["condition"]) for r in rows])
    folds = _fold_assignments(rows, fold_seed)
    fold_ids = np.array([folds[r["key"]] for r in rows])

    designs = {level: [features(r["history"], level) for r in rows] for level in LEVELS}
    designs["evaluator"] = [_evaluator_features(r["condition"], r["demand"], r["unfinished"])
                            for r in rows]
    columns = {level: list(design[0]) for level, design in designs.items()}
    matrices = {level: np.asarray([list(row.values()) for row in design], dtype=float)
                for level, design in designs.items()}

    def fit_subset(indices):
        supervised = {name: y[indices] for name, y in responses.items()}
        public = {level: _fit_level(columns[level], matrices[level][indices], supervised)
                  for level in LEVELS}
        evaluator = _fit_level(columns["evaluator"], matrices["evaluator"][indices],
                               {action: supervised[action] for action in ACTIONS})
        return {"models": public, "evaluator_models": evaluator}

    fold_models, oof = {}, []
    for fold in range(5):
        training = fold_ids != fold
        training_keys = [r["key"] for r, train in zip(rows, training) if train]
        training_hash = hashlib.sha256("\n".join(training_keys).encode()).hexdigest()
        bundle = fit_subset(training)
        fold_models[str(fold)] = bundle
        for i in np.flatnonzero(~training):
            row = rows[i]
            predictions = {level: _predict_level(bundle["models"][level], designs[level][i])
                           for level in LEVELS}
            predictions["evaluator"] = _predict_level(bundle["evaluator_models"], designs["evaluator"][i])
            oof.append(dict(key=row["key"], fold=fold, training_hash=training_hash,
                            targets={t: row[t] for t in responses}, predictions=predictions))
    selection_scores, selected = _selection(oof)
    return fit_subset(slice(None)) | dict(
        solver=SOLVER, folds=folds, fold_models=fold_models, oof=oof,
        selection_scores=selection_scores, selected=selected,
    )
