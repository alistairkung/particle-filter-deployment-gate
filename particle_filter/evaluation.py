"""Score frozen predictions after a whole-root join to evaluator outcomes."""

import numpy as np

from telemetry_sufficiency.scoring import BOOTSTRAPS

METHODS = ("full", "ablated", "history", "age", "current", "prior")


def _calibration(actual, probability):
    indices = np.searchsorted(np.arange(1, 10) / 10, probability, side="right")
    result = []
    for i in range(10):
        selected = indices == i
        n = int(selected.sum())
        result.append(
            dict(
                n=n,
                prediction=float(probability[selected].mean()) if n else None,
                event_fraction=float(actual[selected].mean()) if n else None,
            )
        )
    return result


def _losses(rows, method):
    predictions = np.asarray(
        [[r["predictions"][method][a] for a in ("hold", "promote")] for r in rows],
        dtype=float,
    )
    outcomes = np.asarray([[r[a] for a in ("hold", "promote")] for r in rows])
    if not np.isin(outcomes, (0, 1)).all():
        raise ValueError("action outcomes must be binary")
    if not ((0 <= predictions) & (predictions <= 1)).all():
        raise ValueError("invalid action risk")
    error = predictions - outcomes
    return predictions, outcomes, (error[:, 1] - error[:, 0]) ** 2


def _population(rows, backlog_median):
    models = {}
    for method in METHODS:
        pred, truth, signed_loss = _losses(rows, method)
        masks = dict(
            all=np.ones(len(rows), dtype=bool),
            low=np.array(
                [
                    r["pending"] is not None and r["pending"] <= backlog_median
                    for r in rows
                ]
            ),
            high=np.array(
                [
                    r["pending"] is not None and r["pending"] > backlog_median
                    for r in rows
                ]
            ),
            unavailable=np.array([r["pending"] is None for r in rows]),
        )
        hold, promote = np.mean((pred - truth) ** 2, axis=0)
        models[method] = dict(
            hold_brier=float(hold),
            promote_brier=float(promote),
            increment_mse=float(signed_loss.mean()),
            calibration={
                group: {
                    action: _calibration(truth[mask, j], pred[mask, j])
                    for j, action in enumerate(("hold", "promote"))
                }
                for group, mask in masks.items()
            },
        )
    return dict(n=len(rows), models=models)


def _paired_interval(groups, comparator, seed):
    rng = np.random.default_rng(seed)
    draws = np.zeros(BOOTSTRAPS)
    means = []
    degenerate = True
    for group in groups:
        diff = _losses(group, comparator)[2] - _losses(group, "full")[2]
        means.append(diff.mean())
        degenerate &= np.ptp(diff) == 0
        indices = rng.integers(len(diff), size=(BOOTSTRAPS, len(diff)))
        draws += diff[indices].mean(axis=1) / len(groups)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    interpretation = "inconclusive"
    if degenerate:
        interpretation = "degenerate"
    elif lo > 0:
        interpretation = "favours full PF"
    elif hi < 0:
        interpretation = "favours comparator"
    return dict(
        gain=float(np.mean(means)),
        interval=[float(lo), float(hi)],
        achieved_half_width=float((hi - lo) / 2),
        interpretation=interpretation,
    )


def score_roots(rows, *, backlog_median, seed):
    """Use identical root resamples for both predeclared contrasts."""
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("duplicate evaluation root")
    groups = [
        sorted((r for r in rows if r["regime"] == z), key=lambda r: r["key"])
        for z in "ABC"
    ]
    sizes = [len(g) for g in groups]
    if min(sizes) < 2 or len(set(sizes)) != 1 or sum(sizes) != len(rows):
        raise ValueError(
            "expected complete, balanced A/B/C root groups with at least two each"
        )
    populations = {
        z: _population(g, backlog_median) for z, g in zip("ABC", groups, strict=True)
    }
    populations["all"] = _population(rows, backlog_median)
    return dict(
        populations=populations,
        primary=_paired_interval(groups, "history", seed),
        ablation=_paired_interval(groups, "ablated", seed),
    )
