"""Compare the approved forecast variants on the held-out development panel."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from telemetry_sufficiency.custody import save
from telemetry_sufficiency.parquet_evidence import read_fit

from .config import load_variant
from .development import DEVELOPMENT, development_inventory, fit_fold, replay_root
from .forecast import forecast
from .inference import infer


def predict_variants(history, fit, variants, *, particles, pairs, seed):
    """The same posterior feeds full and ablated forecasts at a given half-life."""
    posteriors = {}
    results = {}
    for variant in variants:
        if variant.half_life not in posteriors:
            posteriors[variant.half_life] = infer(
                history,
                fit.closure,
                fit.likelihood,
                particles=particles,
                half_life=variant.half_life,
                seed=seed,
            )
        posterior = posteriors[variant.half_life]
        risk = forecast(
            posterior,
            fit.closure,
            pairs=pairs,
            seed=seed,
            work_by_regime=fit.cut_work_ns if variant.replace_cut_work else None,
        )
        results[variant.name] = dict(
            cells=asdict(risk),
            hold=risk.hold,
            promote=risk.promote,
            increment=risk.delta,
            mean_work_s=posterior.mean_work_s,
            regime_probabilities=posterior.regime_probabilities,
        )
    return results


def run_experiment(output, names, *, particles, pairs, seed):
    """Fit on four folds, predict the held-out fold, then join development outcomes."""
    from .analysis import score_roots
    from telemetry_sufficiency.parquet_evidence import histories

    output = Path(output)
    variants = [load_variant(name) for name in names]
    if not variants or len(set(names)) != len(names) or min(particles, pairs) <= 0:
        raise ValueError("choose distinct variants and positive numerical budgets")
    output.mkdir(parents=True, exist_ok=False)
    roots = development_inventory()
    replays = {r.key: replay_root(r) for r in roots}
    panel = [r for r in roots if r.panel]
    seeds = np.random.default_rng(seed).bit_generator.random_raw(len(panel)).tolist()
    seeds = dict(zip((r.key for r in panel), seeds, strict=True))
    direct = {r["key"]: r["predictions"] for r in read_fit(DEVELOPMENT)["oof"]}
    scored = []
    for fold in range(5):
        training = [r for r in roots if r.fold != fold]
        fit = fit_fold(training, replays)
        predictions = []
        for root in panel:
            if root.fold != fold:
                continue
            history = [i.reading for i in replays[root.key]]
            estimates = predict_variants(
                history,
                fit,
                variants,
                particles=particles,
                pairs=pairs,
                seed=seeds[root.key],
            )
            predictions.append(dict(key=root.key, variants=estimates))
            if "full" in estimates and "regime-conditional-work" in estimates:
                predictions_for_score = {
                    m: direct[root.key][m]
                    for m in ("prior", "age", "current", "history")
                }
                predictions_for_score.update(
                    full=estimates["full"], ablated=estimates["regime-conditional-work"]
                )
                scored.append(
                    dict(
                        key=root.key,
                        regime=replays[root.key][-1].regime,
                        pending=history[-1].pending_count,
                        hold=root.hold,
                        promote=root.promote,
                        predictions=predictions_for_score,
                    )
                )
        save(
            output / f"fold-{fold}.json",
            dict(training_digest=fit.training_digest, predictions=predictions),
        )
    report = dict(
        partition="development",
        roots=len(panel),
        variants=names,
        seed=seed,
        particles=particles,
        forecast_pairs=pairs,
    )
    if scored:
        counts = [
            r["history"][-1]["pending_count"]
            for r in histories(DEVELOPMENT, [{"key": x.key} for x in roots])
        ]
        report["comparison"] = score_roots(
            scored, backlog_median=float(np.median(counts)), seed=seed
        )
    save(output / "experiment.json", report)
