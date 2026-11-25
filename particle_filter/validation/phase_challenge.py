"""Development-only experiment for the omitted publisher phase and acknowledgement residual."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from simulator.config import SECOND
from .development import development_inventory, physical_digest, replay_root
from .development_replay import PhaseDiagnostic
from ..physical import Closure, matched_pair, propagate

SEED = 413026193
INTERVAL_REPEATS = 4
ACTION_PAIRS = 16
RESAMPLES = 2000


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups["regime=" + row["regime"]].append(row)
        groups["backlog=" + row["backlog"]].append(row)
        groups["phase=" + row["phase"] + ",backlog=" + row["backlog"]].append(row)
    return groups


def _bootstrap_gain(rows, gains, rng):
    """Resample matched root sums and their counts together, stratified by assigned condition."""
    root_sums = defaultdict(lambda: np.zeros(3))
    for row, pair in zip(rows, gains, strict=True):
        aggregate = root_sums[row["condition"], row["root"]]
        aggregate[:2] += pair
        aggregate[2] += 1
    by_condition = defaultdict(list)
    for (condition, _), totals in root_sums.items():
        by_condition[condition].append(totals)
    draws = np.zeros((RESAMPLES, 3))
    for subset in by_condition.values():
        values = np.array(subset)
        # Limit peak memory when a stratum has several hundred independent roots.
        for start in range(0, RESAMPLES, 100):
            selections = rng.integers(len(values), size=(100, len(values)))
            draws[start : start + 100] += values[selections].sum(axis=1)
    return np.quantile(draws[:, :2] / draws[:, 2, None], [0.025, 0.975], axis=0)


def interval_summary(rows, rng):
    summary = {}
    for group, subset in _groups(rows).items():
        actual = np.array([row["target"] for row in subset])
        ordinary = np.abs(np.array([r["ordinary"] for r in subset]) - actual)
        revealed = np.abs(np.array([r["revealed"] for r in subset]) - actual)
        gains = ordinary - revealed
        intervals = _bootstrap_gain(subset, gains, rng)
        stats = dict(roots=len({r["root"] for r in subset}), intervals=len(subset))
        for j, outcome in enumerate(("marks", "work_s")):
            stats[outcome] = dict(
                ordinary_mae=float(ordinary[:, j].mean()),
                revealed_mae=float(revealed[:, j].mean()),
                mae_gain=float(gains[:, j].mean()),
                mae_gain_95=intervals[:, j].tolist(),
            )
        summary[group] = stats
    return summary


def _action_summary(rows):
    summary = {}
    for group, subset in _groups(rows).items():
        actual = np.array([row["target"] for row in subset])
        stats = dict(roots=len(subset))
        for j, action in enumerate(("hold", "promote")):
            event_rate = float(actual[:, j].mean())
            stats[action] = dict(event_fraction=event_rate)
            for model in ("ordinary", "revealed"):
                forecasts = np.array([r[model][j] for r in subset])
                stats[action][model] = dict(
                    mean_risk=float(forecasts.mean()),
                    brier=float(np.mean((forecasts - actual[:, j]) ** 2)),
                )
        summary[group] = stats
    return summary


def _paired_predictions(
    start, previous_regime, regime, closure, diagnostic, random, *, cut
):
    draws = ACTION_PAIRS if cut else INTERVAL_REPEATS
    ordinary = np.zeros(2)
    revealed = np.zeros(2)
    fallback_count = unsupported = 0
    for _ in range(draws):
        donor_seed, tape_seed = random.integers(0, 2**63, size=2)
        base, fallback = closure.reconstruct(
            previous_regime,
            start.work_ns,
            start.pending,
            np.random.default_rng(donor_seed),
        )
        oracle = diagnostic.reconstruct(
            previous_regime,
            start.work_ns,
            start.pending,
            start.phase,
            start.ack_residual_ns,
            np.random.default_rng(donor_seed),
        )
        fallback_count += fallback
        if oracle is None:
            unsupported += 1
            continue
        for initial, total in ((base, ordinary), (oracle, revealed)):
            workload_rng = np.random.default_rng(tape_seed)
            if cut:
                total += matched_pair(initial, regime, workload_rng)
            else:
                end, marks = propagate(initial, regime, workload_rng)
                total += (marks, end.work_ns / SECOND)
    valid = draws - unsupported
    pair = dict(ordinary=ordinary / valid, revealed=revealed / valid) if valid else None
    return pair, fallback_count, unsupported


def _run_fold(task):
    fold, training, validation = task
    donors = [(i.regime, i.end) for history in training for i in history]
    closure = Closure(donors)
    diagnostic = PhaseDiagnostic(donors)
    pending_median = float(np.median([history[-1].end.pending for history in training]))
    steps, actions, coverage = [], [], defaultdict(int)
    for root, history, seed in validation:
        random = np.random.default_rng(seed)
        previous = "A"
        for interval in history:
            state = interval.start
            predictions, fallbacks, unsupported = _paired_predictions(
                state, previous, interval.regime, closure, diagnostic, random, cut=False
            )
            previous = interval.regime
            coverage["interval_fallback_draws"] += fallbacks
            coverage["interval_unsupported_draws"] += unsupported
            if predictions:
                steps.append(
                    dict(
                        root=root.key,
                        condition=root.condition,
                        regime=interval.regime,
                        phase=state.phase,
                        backlog="low" if state.pending <= pending_median else "high",
                        target=(
                            round(interval.reading.publication_throughput_per_s * 5),
                            interval.end.work_ns / SECOND,
                        ),
                        **predictions,
                    )
                )
        if root.panel:
            state = history[-1].end
            predictions, fallbacks, unsupported = _paired_predictions(
                state, previous, previous, closure, diagnostic, random, cut=True
            )
            coverage["action_fallback_draws"] += fallbacks
            coverage["action_unsupported_draws"] += unsupported
            if predictions:
                actions.append(
                    dict(
                        root=root.key,
                        condition=root.condition,
                        regime=previous,
                        phase=state.phase,
                        backlog="low" if state.pending <= pending_median else "high",
                        target=(root.hold, root.promote),
                        **predictions,
                    )
                )
    return (
        steps,
        actions,
        dict(
            fold=fold,
            training_roots=len(training),
            validation_roots=len(validation),
            interval_draws=len(validation) * 12 * INTERVAL_REPEATS,
            action_draws=sum(root.panel for root, _, _ in validation) * ACTION_PAIRS,
            pending_median=pending_median,
            **coverage,
        ),
    )


def phase_challenge(workers=3):
    """Fit reconstruction on four folds; challenge it on identical histories in the fifth."""
    producing_physics = physical_digest()
    roots = development_inventory()
    random = np.random.default_rng(SEED)
    assigned = random.permutation(len(roots))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        histories = list(pool.map(replay_root, roots, chunksize=8))
        tasks = []
        for fold in range(5):
            train = [history for r, history in zip(roots, histories) if r.fold != fold]
            valid = [
                (r, history, [SEED, int(slot)])
                for r, history, slot in zip(roots, histories, assigned)
                if r.fold == fold
            ]
            tasks.append((fold, train, valid))
        results = list(pool.map(_run_fold, tasks))
    if physical_digest() != producing_physics:
        raise ValueError("Physical model changed during the phase challenge")
    steps = [r for intervals, _, _ in results for r in intervals]
    actions = [r for _, cuts, _ in results for r in cuts]
    return dict(
        physical_digest=producing_physics,
        seed=SEED,
        development_roots=len(roots),
        interval_repeats=INTERVAL_REPEATS,
        action_pairs=ACTION_PAIRS,
        root_bootstraps=RESAMPLES,
        folds=[f for _, _, f in results],
        interval=interval_summary(steps, random),
        action=_action_summary(actions),
    )


if __name__ == "__main__":
    print(json.dumps(phase_challenge(), sort_keys=True, indent=2, allow_nan=False))
