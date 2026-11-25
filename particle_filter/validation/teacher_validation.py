"""Replay root-disjoint development folds and assess the interval likelihood at physical states."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import json
from pathlib import Path
import pickle
from time import monotonic

import numpy as np

from telemetry_sufficiency.archive import protect_destination
from telemetry_sufficiency.custody import file_hash
from simulator.observations import INTERVAL
from ..model.likelihood import _context_columns

from .development import (
    development_inventory,
    fit_fold,
    replay_root,
    training_samples,
    model_digest,
    protocol_digest,
)
from .emission_diagnostics import (
    _count_events,
    diagnostic_rows,
    fold_diagnostics,
    repeatable_discrepancies,
)


def _dump_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _compare_pending(model, baseline, context, reading, thresholds, current):
    """Paired count diagnostics at the same physical state and held-out reading."""
    reg, base, previous, _ = _context_columns([context])
    marks = np.array([5 * reading.publication_throughput_per_s])
    lower = np.maximum(0, previous - marks)
    excess = reading.pending_count - lower
    tail_at = np.floor(thresholds[1]) - lower
    before = baseline.pending_excess(reg, base, previous, marks)
    after = model.pending_excess(reg, base, previous, marks)
    observed_tail, integrated_after = current["backlog_tail"]
    integrated_before = integrated_after
    if context.regime == "C":
        design = baseline.designs[0].matrix(reg, base)
        _, _, integrated_before, _ = _count_events(
            baseline,
            reg,
            base,
            previous,
            design @ baseline.marks[:-3],
            np.exp(baseline.marks[-3:][reg]),
            np.ones(1),
            *thresholds,
        )
    result = {
        "conditional_log_mass_after_minus_before": (
            float(after.log_mass(excess)[0]),
            float(before.log_mass(excess)[0]),
        ),
        "given_marks_before": (observed_tail, float(before.survival(tail_at)[0])),
        "given_marks_after": (observed_tail, float(after.survival(tail_at)[0])),
        "integrated_before": (observed_tail, float(integrated_before)),
        "integrated_after": (observed_tail, integrated_after),
    }
    return result


def _heldout_fold(
    fold, roots, replays, output, cache_fits, producing_model, producing_protocol
):
    protect_destination(output)
    if (
        model_digest() != producing_model
        or protocol_digest("teacher") != producing_protocol
    ):
        raise ValueError("Teacher worker model differs from run start")
    rng = np.random.default_rng([57012026, fold])
    training = [root for root in roots if root.fold != fold]
    heldout = [root for root in roots if root.fold == fold]
    start = monotonic()
    fit = fit_fold(training, replays)
    baseline = replace(fit.likelihood, new_commits=None)
    # Local fit cache for sequential checks from the same run. Its manifest
    # names the training inventory. Never unpickle externally supplied models.
    if cache_fits:
        with (output / f"fit-{fold}.pickle").open("xb") as stream:
            pickle.dump(fit, stream)
    records = []
    scores = []
    unsmoothed_scores = []
    pending_records = []
    with gzip.open(output / f"intervals-{fold}.jsonl.gz", "xt") as stream:
        for root in heldout:
            for context, reading in training_samples([replays[root.key]]):
                logp = fit.likelihood.log_likelihood(context, reading)
                if not np.isfinite(logp):
                    raise ValueError(
                        f"Nonfinite held-out log mass in fold {fold}, root {root.key}, boundary {context.start_ns}"
                    )
                metric = diagnostic_rows(
                    fit.likelihood,
                    [context],
                    reading,
                    [1.0],
                    fit.thresholds[context.regime],
                    rng,
                )
                comparison = _compare_pending(
                    fit.likelihood,
                    baseline,
                    context,
                    reading,
                    fit.thresholds[context.regime],
                    metric,
                )
                if context.regime == "C" and reading.interval_end_ns in (-INTERVAL, 0):
                    window = "penultimate" if reading.interval_end_ns else "final"
                    comparison |= {
                        f"{window}_{name}": pair for name, pair in comparison.items()
                    }
                pending_records.append(
                    (root.key, context.regime, context.start_ns, comparison)
                )
                record = root.key, context.regime, context.start_ns, metric
                with np.errstate(divide="ignore"):
                    unsmoothed_scores.append(
                        fit.likelihood.log_likelihood(context, reading, smoothing=0.0)
                    )
                records.append(record)
                scores.append(logp)
                stream.write(json.dumps([*record, logp], allow_nan=False) + "\n")
    unsmoothed = np.array(unsmoothed_scores)
    finite = np.isfinite(unsmoothed)
    report = {
        "model_digest": producing_model,
        "protocol_digest": producing_protocol,
        "fold": fold,
        "training_roots": len(training),
        "heldout_roots": len(heldout),
        "training_digest": fit.training_digest,
        "intervals": len(scores),
        "thresholds": fit.thresholds,
        "smoothing_diagnostic": {
            "zero_mass_without_smoothing": int((~finite).sum()),
            "mean_log_score_change_on_common_support": float(
                np.mean(np.asarray(scores)[finite] - unsmoothed[finite])
            ),
        },
        "mean_joint_log_mass": float(np.mean(scores)),
        "minimum_joint_log_mass": min(scores),
        "seconds": monotonic() - start,
        "diagnostics": fold_diagnostics(records, seed=570120261 + fold),
        "pending_comparison": fold_diagnostics(pending_records, seed=570120261 + fold),
    }
    _dump_json(output / f"fold-{fold}.json", report)
    return report


def plot_diagnostics(reports, output):
    """Plot held-out fold errors with their whole-root percentile intervals."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        (
            "Frequency error (observed − predicted)",
            ("empty", "mark_tail", "backlog_tail", "upper_joint_tail"),
        ),
        ("Lag-one centred PIT product", ("mark_pit_lag1", "pending_pit_lag1")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained", sharey="row")
    for row, (label, metrics) in enumerate(groups):
        for col, regime in enumerate("ABC"):
            ax = axes[row, col]
            ax.axhline(0.0, color=".6", linewidth=0.7)
            for j, metric in enumerate(metrics):
                results = [r["diagnostics"][regime][metric] for r in reports]
                errors = np.array([v["observed"] - v["predicted"] for v in results])
                intervals = np.array([v["error_ci95"] for v in results])
                position = np.arange(len(reports)) + (j - (len(metrics) - 1) / 2) * 0.12
                points = ax.scatter(
                    position, errors, s=11, label=metric.replace("_", " ")
                )
                ax.vlines(
                    position,
                    intervals[:, 0],
                    intervals[:, 1],
                    colors=points.get_facecolors(),
                    linewidth=1,
                )
            ax.set(xticks=range(len(reports)), xlabel="Held-out fold")
            if row == 0:
                ax.set_title(f"Regime {regime}")
            if col == 0:
                ax.set_ylabel(label)
            if col == 2:
                ax.legend(fontsize="small")
    fig.savefig(output / "tail-and-serial-residuals.png", dpi=150)
    plt.close(fig)


def run_teacher_validation(output, workers=4, *, cache_fits=False):
    """Run physical-state diagnostics only, with fresh output and no protected inputs."""
    protect_destination(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Teacher-validation output directory must be empty")
    producing_model = model_digest()
    producing_protocol = protocol_digest("teacher")
    roots = development_inventory()
    start = monotonic()
    reports = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            replays = dict(
                zip(
                    (r.key for r in roots),
                    pool.map(replay_root, roots, chunksize=4),
                    strict=True,
                )
            )
        # Every worker receives the identical replay panel and fits only its own
        # training roots. Pool processes bound simultaneous solver memory use.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks = [
                pool.submit(
                    _heldout_fold,
                    fold,
                    roots,
                    replays,
                    output,
                    cache_fits,
                    producing_model,
                    producing_protocol,
                )
                for fold in range(5)
            ]
            reports = [task.result() for task in tasks]
        if (
            model_digest() != producing_model
            or protocol_digest("teacher") != producing_protocol
        ):
            raise ValueError("Teacher model changed during validation")
        flags = repeatable_discrepancies([r["diagnostics"] for r in reports])
        summary = {
            "model_digest": producing_model,
            "protocol_digest": producing_protocol,
            "kind": "held-out teacher-forced development diagnostics",
            "roots": len(roots),
            "intervals": sum(r["intervals"] for r in reports),
            "folds": len(reports),
            "fold_evidence": {
                str(i): file_hash(output / f"fold-{i}.json") for i in range(5)
            },
            "bootstrap_resamples": 2000,
            "seconds": monotonic() - start,
            "gate": "investigate" if flags else "no repeatable teacher-forced trigger",
            "repeatable_discrepancies": flags,
        }
        plot_diagnostics(reports, output)
        _dump_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        _dump_json(
            output / "failure.json",
            {
                "kind": type(exc).__name__,
                "message": str(exc),
                "seconds": monotonic() - start,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-fits",
        action="store_true",
        help="Export local development fits for a repeat diagnostic in the same environment",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_teacher_validation(
                args.output, args.workers, cache_fits=args.cache_fits
            ),
            indent=2,
        )
    )
