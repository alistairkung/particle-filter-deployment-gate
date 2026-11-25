"""Held-out development validation at the public forecast boundary."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from simulator.config import SECOND
from simulator.observations import INTERVAL
from telemetry_sufficiency.custody import file_hash, save
from telemetry_sufficiency.parquet_evidence import read_fit

from .analysis import score_roots
from .development import (
    DEVELOPMENT,
    development_inventory,
    development_seeds,
    fit_fold,
    replay_root,
)
from .emission_diagnostics import (
    diagnostic_rows,
    fold_diagnostics,
    repeatable_discrepancies,
)
from .forecast import forecast
from .inference import infer

# Declared diagnostic settings. Convergence and method lock are separate stages.
PARTICLES = 128
PAIRS = 64
SEED = 570112
# Perturb the same scheduled reading for each diagnostic copy.
PERTURBED_INTERVAL = 6


def _diagnose_history(root, intervals, fit, seed):
    records = []
    history = [i.reading for i in intervals]
    by_end = {i.reading.interval_end_ns: i for i in intervals}
    diagnostic_rng = np.random.default_rng([SEED, seed, 1])

    def observe(end, contexts, weights, marks, work_s):
        truth = by_end[end]
        statistics = diagnostic_rows(
            fit.likelihood,
            contexts,
            truth.reading,
            weights,
            fit.thresholds[truth.regime],
            diagnostic_rng,
        )
        records.append((root.key, truth.regime, end - INTERVAL, statistics))

    start = perf_counter()
    posterior = infer(
        history,
        fit.closure,
        fit.likelihood,
        particles=PARTICLES,
        seed=seed,
        on_prediction=observe,
    )
    filtering_s = perf_counter() - start
    # Distinct filter/forecast streams; identical forecast seeds for full and ablated work.
    start = perf_counter()
    predictions = {
        name: forecast(
            posterior, fit.closure, pairs=PAIRS, seed=seed, work_by_regime=law
        )
        for name, law in (("full", None), ("ablated", fit.cut_work_ns))
    }
    time_s = perf_counter() - start
    cells = {name: asdict(prediction) for name, prediction in predictions.items()}
    risks = {
        name: dict(hold=p.hold, promote=p.promote) for name, p in predictions.items()
    }
    result = dict(
        key=root.key,
        regime=intervals[-1].regime,
        hold=root.hold,
        promote=root.promote,
        pending=history[-1].pending_count,
        work_s=root.cut_work_s,
        backlog_median=fit.backlog_median,
        predictions=risks,
        paired_cells=cells,
        posterior_regime=posterior.regime_probabilities,
        posterior_work_s=posterior.mean_work_s,
        truth_regime=intervals[-1].regime,
        conditional_mean_work_s=float(np.mean(fit.cut_work_ns[intervals[-1].regime]))
        / SECOND,
        trace=[
            asdict(step) | {"regime_counts": list(map(int, step.regime_counts))}
            for step in posterior.steps
        ],
        filtering_s=filtering_s,
        forecast_s=time_s,
    )
    return records, result


def _delivery_checks(intervals, fit, seed):
    history = [i.reading for i in intervals]
    r = PERTURBED_INTERVAL - 1
    delayed = history.copy()
    delayed[r] = replace(
        history[r], available_at_ns=history[r].available_at_ns + INTERVAL
    )
    copies = dict(
        missing_first=history[1:],
        deleted=history[:r] + history[r + 1 :],
        delayed=delayed,
    )
    result = {}
    for kind, readings in copies.items():
        post = infer(
            readings, fit.closure, fit.likelihood, particles=PARTICLES, seed=seed
        )
        result[kind] = dict(
            mean_work_s=post.mean_work_s,
            regime=post.regime_probabilities,
            surviving_ancestors=post.steps[-1].unique_ancestors,
        )
    return result


def _validate_fold(fold, output):
    roots = development_inventory()
    seeds = development_seeds(roots, SEED)
    training = [r for r in roots if r.fold != fold]
    held = [r for r in roots if r.fold == fold and r.panel]
    fit = fit_fold(training, {r.key: replay_root(r) for r in training})
    records, results, delivery = [], [], {}
    # Retrieve existing out-of-fold direct predictions, keyed only on the evaluator side.
    frozen = read_fit(DEVELOPMENT)
    direct = {r["key"]: r["predictions"] for r in frozen["oof"]}
    for root in held:
        intervals = replay_root(root)
        observations, result = _diagnose_history(root, intervals, fit, seeds[root.key])
        for name in ("history", "prior", "current", "age"):
            result["predictions"][name] = {
                a: direct[root.key][name][a] for a in ("hold", "promote")
            }
        records.extend(observations)
        results.append(result)
        if root.condition not in delivery:
            delivery[root.condition] = _delivery_checks(intervals, fit, seeds[root.key])
    report = dict(
        training_digest=fit.training_digest,
        held_out=[r.key for r in held],
        diagnostics=fold_diagnostics(records, seed=SEED + fold),
        roots=results,
        delivery_checks=delivery,
    )
    save(output / f"fold-{fold}.json", report)
    return report


def _state_diagnostics(rows):
    results = {}
    for regime in ("all", "A", "B", "C"):
        group = [r for r in rows if regime == "all" or r["regime"] == regime]
        truth = np.asarray([r["work_s"] for r in group])
        probs = np.asarray([r["posterior_regime"] for r in group])
        target = np.asarray([[r["regime"] == z for z in "ABC"] for r in group])
        results[regime] = dict(
            regime_brier=float(np.mean(np.sum((probs - target) ** 2, axis=1))),
            work_rmse_s=float(
                np.sqrt(np.mean((truth - [r["posterior_work_s"] for r in group]) ** 2))
            ),
            regime_mean_work_rmse_s=float(
                np.sqrt(
                    np.mean(
                        (truth - [r["conditional_mean_work_s"] for r in group]) ** 2
                    )
                )
            ),
        )
    nominal = [r for r in rows if r["regime"] in "AB"]
    results["nominal_discrimination"] = {
        z: [
            sum(
                r["regime"] == z and int(np.argmax(r["posterior_regime"][:2])) == i
                for r in nominal
            )
            for i in range(2)
        ]
        for z in "AB"
    }
    return results


def run(output, workers):
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    start = perf_counter()
    try:
        with ProcessPoolExecutor(workers) as pool:
            folds = list(pool.map(_validate_fold, range(5), [output] * 5))
        rows = [r for fold in folds for r in fold["roots"]]
        flags = repeatable_discrepancies([f["diagnostics"] for f in folds])
        # Each OOF root is scored against exactly its own held-out action outcomes.
        report = dict(
            status="requires-review",
            particles=PARTICLES,
            forecast_pairs=PAIRS,
            roots=len(rows),
            flags=flags,
            state=_state_diagnostics(rows),
            actions=score_roots(rows, seed=SEED),
            hardware=platform.machine(),
            workers=workers,
            elapsed_s=perf_counter() - start,
            fold_evidence={
                str(i): file_hash(output / f"fold-{i}.json") for i in range(5)
            },
        )
        save(output / "sequential.json", report)
    except Exception as exc:
        save(
            output / "failure.json",
            dict(status="failed", error=str(exc), elapsed_s=perf_counter() - start),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.workers)
