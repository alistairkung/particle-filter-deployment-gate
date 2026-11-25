"""Select numerical budgets and measure runtime using held-out development roots."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from telemetry_sufficiency import CONDITIONS
from telemetry_sufficiency.custody import save

from .budgets import particle_change, rollout_variation
from .development import development_inventory, fit_fold, replay_root
from .panel_execution import (
    Panel,
    PublicCase,
    ResourceLimit,
    compute_posterior,
    compute_risk,
    initialise_worker,
    peak_rss_mib,
    time_all_variants,
)


def validate_evidence(path, expected_digest, *, diagnostics_only=False):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError("Validation evidence digest differs from the requested report")
    report = json.loads(payload)
    if report.get("status") not in {
        "validated",
        "requires-review",
        "failed",
        "investigate",
    }:
        raise ValueError("Unrecognised calibration status")
    if not diagnostics_only and report["status"] != "validated":
        raise ValueError(
            "Numerical budget selection requires validated development evidence"
        )
    return report


def paired_loss_change(smaller, larger, outcomes):
    """Pair N and 2N forecasts on each signed outcome and across four seed runs."""
    if (
        smaller.shape != (3, 12, 4)
        or larger.shape != smaller.shape
        or outcomes.shape != (3, 12)
    ):
        raise ValueError("Expected A/B/C by 12 roots by four seeds")
    y = outcomes[..., None]
    if not np.isin(y, (-1, 0, 1)).all():
        raise ValueError("Expected a signed paired breach indicator")
    changes = np.square(larger - y) - np.square(smaller - y)
    seeds = changes.mean(axis=(0, 1))
    return dict(
        larger_minus_smaller_mse=float(seeds.mean()),
        seed_sd=float(seeds.std(ddof=1)),
        by_seed=seeds.tolist(),
        per_regime=changes.mean(axis=(1, 2)).tolist(),
    )


def _particle_panel(panel, n, *, estimate):
    jobs = [
        (slot, n, seed, 120)
        for slot, seeds in enumerate(panel.seeds("filter", 4))
        for seed in seeds
    ]
    posteriors, elapsed = panel.run(
        f"particles-{n}", compute_posterior, jobs, estimate=estimate
    )
    summary = np.array(
        [(*p.regime_probabilities, p.mean_work_s) for p in posteriors]
    ).reshape(3, 12, 4, 4)
    panel.evidence(
        f"particles-{n}.json",
        dict(
            states=summary.tolist(),
            max_weights=[max(u.max_weight for u in p.steps) for p in posteriors],
            min_ess=[min(u.ess for u in p.steps) for p in posteriors],
            final_unique_ancestors=[p.steps[-1].unique_ancestors for p in posteriors],
            regime_extinction_steps=[
                sum(0 in u.regime_counts for u in p.steps) for p in posteriors
            ],
            seed_sd_by_regime=np.std(summary, axis=2, ddof=1).mean(axis=1).tolist(),
        ),
    )
    return posteriors, summary, elapsed


def _risk_panel(panel, posteriors, k, *, ablated, fixed, name, purpose, estimate):
    seeds = panel.seeds(purpose, 4)
    jobs = [
        (slot, posteriors[4 * slot + (0 if fixed else repeat)], k, batch_seed, ablated)
        for slot, batch in enumerate(seeds)
        for repeat, batch_seed in enumerate(batch)
    ]
    probabilities, elapsed = panel.run(name, compute_risk, jobs, estimate=estimate)
    values = np.array(probabilities).reshape(3, 12, 4, 3)
    panel.evidence(
        name + ".json",
        dict(action_order=["hold", "promote", "increment"], risks=values.tolist()),
    )
    return values, elapsed


def _select_particles(panel, estimated, first_comparison_only):
    previous, first, elapsed = _particle_panel(panel, 128, estimate=estimated)
    for n in (256, 512, 1024):
        current, second, elapsed = _particle_panel(panel, n, estimate=2 * elapsed)
        comparison = particle_change(first, second)
        panel.report["particle_change"][str(n)] = comparison
        if first_comparison_only:
            panel.report["reason"] = (
                "Measured the 128-to-256 development comparison while "
                "calibration remains unresolved. Neither N nor K is selected."
            )
            return None, None, None
        if comparison["passes"]:
            return n // 2, previous, current
        previous, first = current, second
    raise ResourceLimit(
        "No particle-count doubling through 1024 passed the state tolerances"
    )


def _select_rollouts(panel, posteriors, estimated):
    k, previous_pass = 64, False
    while True:
        checks = {}
        runtimes = []
        for label in ("full", "ablated"):
            values, elapsed = _risk_panel(
                panel,
                posteriors,
                k,
                ablated=label == "ablated",
                fixed=True,
                name=f"fixed-{label}-{k}",
                purpose=f"fixed-K-{k}",
                estimate=estimated,
            )
            checks[label] = rollout_variation(values)
            runtimes.append(elapsed)
        panel.report["seed_variation"][str(k)] = checks
        passes = all(check["passes"] for check in checks.values())
        if previous_pass and passes:
            return k // 2, max(runtimes) / 2
        previous_pass = passes
        k *= 2
        estimated = 2 * max(runtimes)


def _compare_loss(panel, small, large, k, targets, forecast_estimate):
    values = []
    for label, posterior in (("N", small), ("2N", large)):
        risk, _ = _risk_panel(
            panel,
            posterior,
            k,
            ablated=False,
            fixed=False,
            name=f"paired-loss-{label}",
            purpose="coupled-loss",
            estimate=forecast_estimate,
        )
        values.append(risk[..., 2])
    return paired_loss_change(*values, targets)


def _run_panel(args, report, deadline):
    from telemetry_sufficiency.parquet_evidence import read_fit
    from .development import DEVELOPMENT

    training_population = development_inventory()
    roots = sorted(
        (r for r in training_population if r.panel),
        key=lambda root: (CONDITIONS.index(root.condition), root.index),
    )
    expected = [(z, i) for z in CONDITIONS for i in range(12)]
    if [(r.condition, r.index) for r in roots] != expected:
        raise ValueError(
            "The development panel must contain the 12 predeclared roots per condition"
        )
    replays = {r.key: replay_root(r) for r in training_population}
    fits = [
        fit_fold([r for r in training_population if r.fold != f], replays)
        for f in range(5)
    ]
    cases = [
        PublicCase(tuple(i.reading for i in replays[r.key]), r.fold) for r in roots
    ]
    del replays
    direct = read_fit(DEVELOPMENT)["models"]
    report["training_inventory_digests"] = [fit.training_digest for fit in fits]
    report["slots"] = [
        dict(root=r.key, condition=r.condition, index=r.index, fold=r.fold)
        for r in roots
    ]
    targets = np.array([r.promote - r.hold for r in roots]).reshape(3, 12)
    with ProcessPoolExecutor(
        args.workers, initializer=initialise_worker, initargs=(fits, cases, direct)
    ) as pool:
        panel = Panel(pool, report, args.output, deadline, args.seed)
        first, prefix_s = panel.run(
            "benchmark-one-prefix", compute_posterior, [(0, 128, args.seed, 120)]
        )
        _, pair_s = panel.run(
            "benchmark-one-pair", compute_risk, [(0, first[0], 1, args.seed, False)]
        )
        target_jobs = 144 / args.workers
        n, small, large = _select_particles(
            panel, prefix_s * target_jobs, args.state_check_only
        )
        if n is None:
            report["status"] = "diagnostics-only"
            return
        k, risk_time = _select_rollouts(panel, small, pair_s * target_jobs * 64)
        report["particle_loss_change"] = _compare_loss(
            panel, small, large, k, targets, risk_time
        )
        filter_seeds = panel.seeds("end-to-end-filter", 1)
        forecast_seeds = panel.seeds("end-to-end-forecast", 1)
        jobs = [(j, n, k, filter_seeds[j][0], forecast_seeds[j][0]) for j in range(36)]
        estimate = 3 * report["stages"][f"particles-{n}"]["wall_s"] / 4 + risk_time
        _, wall = panel.run(
            "end-to-end-all-variants", time_all_variants, jobs, estimate=estimate
        )
    report["candidate"] = dict(particles=n, pairs=k)
    projected = wall * 300 / 36
    report["runtime"] = dict(
        roots_per_s=36 / wall,
        matched_pairs_per_s=36 * 4 * k / wall,
        workers=args.workers,
        wall_s_36_development_roots=wall,
        projected_300_forecast_s=projected,
        projected_verification_replay_s=projected,
        max_worker_rss_mib=report["stages"]["end-to-end-all-variants"][
            "max_worker_rss_mib"
        ],
        parent_peak_rss_mib=peak_rss_mib(),
        timing_scope="Three inference runs, four paired forecast variants and four direct predictions per root on the intended workers. Independent replay forecast cost projected as a second pass.",
    )
    report["status"] = "budget-candidate"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--validation-sha256")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Collect development measurements while validation remains open.",
    )
    parser.add_argument(
        "--state-check-only",
        action="store_true",
        help="Measure the first particle doubling when calibration prevents a lock.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-hours", required=True, type=float, help="Available development wall time"
    )
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--seed", type=int, default=5_701_012)
    args = parser.parse_args(argv)
    if not 1 <= args.workers or not (0 < args.max_hours < math.inf):
        parser.error("Expected positive worker and wall-time budgets")
    if bool(args.validation) != bool(args.validation_sha256):
        parser.error("Supply validation evidence and its digest together")
    if args.state_check_only and not args.diagnostics_only:
        parser.error("--state-check-only requires --diagnostics-only")
    if not args.validation and not args.diagnostics_only:
        parser.error("A budget candidate requires development validation evidence")
    from telemetry_sufficiency.archive import protect_destination

    protect_destination(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    now = time.monotonic()
    report = dict(
        status="running",
        development_only=True,
        state_check_only=args.state_check_only,
        diagnostics_only=args.diagnostics_only,
        seed=args.seed,
        max_hours=args.max_hours,
        workers=args.workers,
        host=platform.platform(),
        implementation=platform.python_version(),
        validation_sha256=args.validation_sha256,
        particle_change={},
        seed_variation={},
        stages={},
        evidence_digests={},
    )
    try:
        if args.validation:
            validation = validate_evidence(
                args.validation,
                args.validation_sha256,
                diagnostics_only=args.diagnostics_only,
            )
            report["validation_status"] = validation["status"]
            from .development import study_source_digest

            if validation.get("source_digest") != study_source_digest():
                raise ValueError(
                    "Development validation source digest differs from the current implementation"
                )
            for name, expected in validation["evidence_digests"].items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Invalid validation evidence location")
                with (args.validation.parent / relative).open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(
                            f"Validation evidence digest mismatch for {name}"
                        )
        else:
            report["validation_status"] = "requires-review"
        from .development import study_source_digest

        report["source_digest"] = study_source_digest()
        save(args.output / "convergence.json", report)
        _run_panel(args, report, now + 3600 * args.max_hours)
        if args.diagnostics_only:
            report["status"] = "diagnostics-only"
    except KeyboardInterrupt:
        report.update(
            status="interrupted", reason="Development run interrupted before completion"
        )
        raise
    except ResourceLimit as exc:
        report.update(status="requires-review", reason=str(exc))
    except Exception as exc:
        report.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_s"] = time.monotonic() - now
        save(args.output / "convergence.json", report)
    return report


if __name__ == "__main__":
    main()
