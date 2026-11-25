"""Apply locked fits to independent numeric designs and reconstruct development selection."""

import hashlib
import math
from statistics import fmean

import numpy as np
from scipy.special import expit, softmax

from .audit import compare

CONDITIONS = ("reference", "contention", "publisher")
LEVELS = ("prior", "age", "current", "history", "evaluator")
METRICS = (
    "pending_count",
    "oldest_pending_age_s",
    "publication_throughput_per_s",
    "completed_delay_median_s",
)


def features(level, row):
    if level == "prior":
        return {}
    if level == "evaluator":
        return {
            **{name: float(name == row["condition"]) for name in CONDITIONS},
            "demand": row["demand"],
            "unfinished": row["unfinished"],
        }
    history = row["history"]
    public = set(METRICS) | {"interval_start_ns", "interval_end_ns", "available_at_ns"}
    ends = set()
    last = None
    recent = []
    for record in history:
        end = record["interval_end_ns"]
        delivered = record["available_at_ns"]
        if (
            set(record) != public
            or end in ends
            or end > delivered
            or delivered > 0
            or record["interval_start_ns"] != end - 5_000_000_000
        ):
            raise ValueError("invalid public history")
        ends.add(end)
        if last is None or end > last["interval_end_ns"]:
            last = record
        if end in range(-25_000_000_000, 1, 5_000_000_000):
            recent.append(record)
    values = {}
    flags = {}

    def add(name, value, logarithmic=False):
        flags["undefined_" + name] = float(value is None)
        values[name] = math.nan if value is None else math.log1p(value) if logarithmic else value

    for metric in (METRICS[1:2] if level == "age" else METRICS):
        add("current_" + metric, None if last is None else last[metric], True)
    if level == "history":
        for metric in METRICS:
            pairs = [
                (r["interval_end_ns"] / 1e9, r[metric]) for r in recent if r[metric] is not None
            ]
            mean = fmean(y for _, y in pairs) if pairs else None
            slope = None
            if len(pairs) >= 2:
                mid = fmean(x for x, _ in pairs)
                slope = sum((x - mid) * (y - mean) for x, y in pairs) / sum(
                    (x - mid) ** 2 for x, _ in pairs
                )
            add("mean_" + metric, mean, True)
            add("slope_" + metric, slope)
            values["defined_" + metric] = len(pairs)
        values["deliveries"] = len(recent)
    return values | flags


def _key_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _fold_map(rows, seed):
    result = {}
    for condition in CONDITIONS:
        keys = [r["key"] for r in rows if r["condition"] == condition]
        for i, key in enumerate(sorted(keys, key=lambda k: _key_hash(f"{seed}/{k}"))):
            result[key] = i % 5
    return result


def _design(level, rows):
    data = [features(level, row) for row in rows]
    names = list(data[0])
    return names, np.array([list(row.values()) for row in data], dtype=np.float64)


def _predict(fit, x):
    if fit["converged"] is not True:
        raise ValueError("required fit did not converge")
    intercept = np.asarray(fit["intercept"], dtype=np.float64)
    if fit["kind"] == "constant":
        return np.broadcast_to(intercept, (len(x), *intercept.shape))
    value = intercept + x @ np.asarray(fit["coef"], dtype=np.float64)
    if fit["kind"] == "linear":
        return np.clip(value, 0, None)
    if fit["kind"] == "binary":
        return expit(value)
    if fit["kind"] == "multiclass":
        return softmax(value, axis=1)
    raise ValueError(f"unrecognised fit {fit['kind']}")


def _apply(bundle, level, predictors, responded, trainings):
    names, train = predictors
    _, scored = responded
    compare(bundle["features"], names, f"{level}.features")
    medians = np.array(
        [np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in train.T]
    )
    imputed = np.where(np.isnan(train), medians, train)
    means, scales = imputed.mean(axis=0), imputed.std(axis=0)
    scales[np.ptp(imputed, axis=0) == 0] = 1.0
    for field, expected in (("medians", medians), ("means", means), ("scales", scales)):
        compare(bundle[field], expected.tolist(), f"{level}.{field}")
    x = (np.where(np.isnan(scored), medians, scored) - bundle["means"]) / bundle["scales"]
    targets = (
        ("hold", "promote")
        if level == "evaluator"
        else ("hold", "promote", "demand", "unfinished", "condition")
    )
    compare(sorted(bundle["fits"]), sorted(targets), f"{level}.targets")
    predictions = {}
    for name in targets:
        if level == "prior":
            if name == "condition":
                prior = [fmean(r[name] == c for r in trainings) for c in CONDITIONS]
            else:
                prior = fmean(r[name] for r in trainings)
            compare(bundle["fits"][name]["intercept"], prior, f"prior.{name}")
            compare(bundle["fits"][name]["kind"], "constant", f"prior.{name}.kind")
        predictions[name] = _predict(bundle["fits"][name], x)
    predictions["increment"] = predictions["promote"] - predictions["hold"]
    return predictions


def _check_bundle(bundle, reference_designs, indices, test_designs, test_indices, train, score):
    compare(sorted(bundle["models"]), sorted(LEVELS[:-1]), "public model inventory")
    for level in LEVELS:
        model = bundle["evaluator_models"] if level == "evaluator" else bundle["models"][level]
        names, reference = reference_designs[level]
        _, test = test_designs[level]
        calculated = _apply(
            model, level, (names, reference[indices]), (names, test[test_indices]), train
        )
        for i, r in enumerate(score):
            compare(
                r["predictions"][level],
                {k: v[i].tolist() for k, v in calculated.items()},
                f"{r['key']}.{level}.prediction",
            )


def audit_fits(development, protected, fitted, fold_seed):
    development = sorted(development, key=lambda r: r["key"])
    protected = sorted(protected, key=lambda r: r["key"])
    keys = [r["key"] for r in development + protected]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate or cross-partition key")
    fold = _fold_map(development, fold_seed)
    compare(fitted["folds"], fold, "development.folds")
    oof = {r["key"]: r for r in fitted["oof"]}
    if len(oof) != len(fitted["oof"]) or set(oof) != set(fold):
        raise ValueError("out-of-fold inventory")
    design = {level: _design(level, development) for level in LEVELS}
    test_design = {level: _design(level, protected) for level in LEVELS}
    for i in range(5):
        train_indices, test_indices = [], []
        for j, row in enumerate(development):
            (test_indices if fold[row["key"]] == i else train_indices).append(j)
        train = [development[j] for j in train_indices]
        train_hash = _key_hash("\n".join(sorted(r["key"] for r in train)))
        scored = []
        for j in test_indices:
            r = development[j]
            evidence = oof[r["key"]]
            compare(
                evidence["targets"],
                {k: r[k] for k in ("condition", "demand", "unfinished", "hold", "promote")},
                f"{r['key']}.targets",
            )
            compare(evidence["training_hash"], train_hash, "training inventory")
            compare(evidence["fold"], i, "fold")
            scored.append(evidence)
        _check_bundle(
            fitted["fold_models"][str(i)],
            design,
            train_indices,
            design,
            test_indices,
            train,
            scored,
        )
    selection_scores = {}
    selected = {}
    for target in ("demand", "increment"):
        population = CONDITIONS if target == "increment" else CONDITIONS[:2]
        loss = {}
        for level in ("age", "current", "history"):
            errors = {condition: [] for condition in population}
            for r in development:
                if r["condition"] in errors:
                    truth = (
                        r["demand"] if target == "demand" else int(r["promote"]) - int(r["hold"])
                    )
                    predicted = oof[r["key"]]["predictions"][level][target]
                    errors[r["condition"]].append((truth - predicted) ** 2)
            loss[level] = fmean(fmean(group) for group in errors.values())
        selection_scores[target] = loss
        selected[target] = min(loss, key=loss.get)
    compare(fitted["selection_scores"], selection_scores, "claim development scores")
    compare(fitted["selected"], selected, "claim selections")
    _check_bundle(fitted, design, slice(None), test_design, slice(None), development, protected)
    return selected
