"""Independent reconstruction from FIFO admissions, completed write effects and fit evidence."""

import hashlib
import itertools
import json
from statistics import median


SECOND = 1_000_000_000
INTERVAL = 5 * SECOND
PRIMARY_DELIVERIES = {end: end for end in range(-55 * SECOND, 1, INTERVAL)}


def history_from_writes(writes, deliveries=PRIMARY_DELIVERIES, cutoff=0):
    """Reconstruct snapshots and interval statistics from completed lifecycle timestamps."""
    commits = [
        (w.commit, w.mark)
        for w in writes.values()
        if w.eligible and w.commit is not None and w.commit <= cutoff
    ]
    history = []
    for end, available in sorted(deliveries.items()):
        if available is None or available > cutoff:
            continue
        if available < end:
            raise ValueError("delivery before measurement")
        pending = [c for c, m in commits if c <= end and (m is None or m > end)]
        delays = [m - c for c, m in commits if m is not None and end - INTERVAL < m <= end]
        history.append(
            dict(
                interval_start_ns=end - INTERVAL,
                interval_end_ns=end,
                available_at_ns=available,
                pending_count=len(pending),
                oldest_pending_age_s=(end - min(pending)) / SECOND if pending else 0.0,
                publication_throughput_per_s=len(delays) / 5,
                completed_delay_median_s=median(delays) / SECOND if delays else None,
            )
        )
    return history


def windows_from_writes(writes):
    """Decode the frozen pooled-write endpoint directly from arrival and commit ticks."""
    eligible = [0] * 4
    misses = [0] * 4
    for w in writes.values():
        deadline = w.arrival + SECOND // 4
        if w.eligible and 0 < w.arrival and deadline <= 120 * SECOND:
            window = (deadline - 1) // (30 * SECOND)
            eligible[window] += 1
            misses[window] += w.commit is None or w.commit > deadline
    counts = list(zip(eligible, misses))
    return counts, int(any(20 * m > n for n, m in counts))


def truth_from_visits(visits):
    work = queued = active = 0
    duration = 30 * SECOND
    for kind, _, enqueue, start, end in visits:
        if enqueue > 0:
            break
        if kind == 1 and -duration < enqueue:
            work += end - start
        if start > 0:
            queued += end - start
        elif end > 0:
            active += end
    return dict(
        query_work_ns=work,
        duration_ns=duration,
        queued_ns=queued,
        active_residual_ns=active,
        demand=work / duration,
        unfinished=(active + queued) / SECOND,
    )


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def check_admissions(model, prefix, future, branch):
    """Bind regenerated primitives to actual FIFO visits in admission order."""
    external = (v for v in model.db.visits if v[0] in (0, 1))
    observed = []
    for raw, visit in itertools.zip_longest(itertools.chain(prefix, future), external):
        if raw is None or visit is None:
            raise ValueError("missing or extra admission")
        tick, source, identity, *draws = raw
        kind, physical_id, queued, start, end = visit
        if source == "write":
            uniform, business = draws
            share = 0.60 if tick > 0 and branch.endswith("promote") else 0.25
            java = uniform < share
            mean = model.c.java_ms if java else model.c.ruby_ms
            work = round(business * mean * 1_000_000) + model.c.insert_ms * 1_000_000
            w = model.writes.get(physical_id)
            if w is None or (w.arrival, w.java, w.demand) != (tick, java, work):
                raise ValueError("write admission changed identity, route or demand")
            expected = (0, identity, tick, work)
        else:
            if source not in ("reference", "extra"):
                raise ValueError(f"unknown admission source {source}")
            java = False
            (work,) = draws
            expected = (1, 2 * identity + (source == "extra"), tick, work)
        if (kind, physical_id, queued, end - start) != expected:
            raise ValueError(f"unmatched {source} admission at {tick}")
        observed.append((queued, source, identity, end - start, java))
    return digest(observed), len(observed)


def _regenerate(condition, partition, index, phase, surprise, config):
    """Draw fresh primitives directly from pilot streams; transform only actual admissions."""
    import numpy as np
    from simulator.inputs import arrivals, rng

    def generator(stream):
        return rng(config.seed, "outbox-sufficiency-v1", condition, partition, index, phase, stream)

    is_prefix = phase == "prefix"
    seconds = 60 if is_prefix else 120
    offset = -60 * SECOND if is_prefix else 0
    times = arrivals(generator("application"), config.writes, seconds)
    uniforms = generator("routing").random(len(times))
    works = generator("business").exponential(size=len(times))
    first = -len(times) if is_prefix else 0
    tape = [
        (offset + t, "write", first + i, float(uniforms[i]), float(works[i]))
        for i, t in enumerate(times)
    ]
    sources = [("reference", config.reference)]
    if condition == "contention" or (surprise and not is_prefix):
        sources.append(("extra", config.extra))
    for name, rate in sources:
        ticks = arrivals(generator(name + "-arrivals"), rate, seconds)
        work = generator(name + "-work").exponential(config.query_ms * 1_000_000, len(ticks))
        for i, (t, demand) in enumerate(zip(ticks, work)):
            tick = offset + t
            if name == "extra" and is_prefix and tick <= -30 * SECOND:
                continue
            tape.append((tick, name, -(i + 1) if is_prefix else i, max(1, int(np.rint(demand)))))
    return sorted(tape, key=lambda item: item[0])


def _native_replay(config, condition, branch, prefix, future):
    """Run the frozen FIFO without invoking the producer, collector, or evaluator."""
    from dataclasses import replace
    from simulator.model import Model

    def route(tape, share):
        for tick, kind, identity, draw, *extra in tape:
            if kind == "write":
                java = draw < share
                ms = config.java_ms if java else config.ruby_ms
                work = round(extra[0] * ms * 1_000_000) + config.insert_ms * 1_000_000
                yield tick, kind, identity, work, java
            else:
                yield tick, kind, identity, draw, False

    model = Model(config, -60 * SECOND)
    model.schedule(route(prefix, 0.25))
    model.advance(-30 * SECOND)
    if condition == "publisher":
        model.c = replace(config, ack_ms=config.slow_ms)
    model.advance(0)
    model.schedule(route(future, 0.60 if branch.endswith("promote") else 0.25))
    model.advance(120 * SECOND)
    return model


def _cut_state(model, condition):
    """Time-filter the horizon ledger, including partial publisher progress."""
    visits = [v for v in model.db.visits if v[2] <= 0]
    rows = []
    eligible = []
    pending = []
    for identity, w in model.writes.items():
        if w.arrival > 0:
            continue
        commit, claim, ack, mark = (
            t if t is not None and t <= 0 else None for t in (w.commit, w.claim, w.ack, w.mark)
        )
        eligible.append(w.eligible)
        rows.append((identity, w.arrival, w.demand, commit, claim, ack, mark, w.java))
        if w.eligible and commit is not None and mark is None:
            pending.append((commit, identity, claim, ack))
    pending.sort(key=lambda item: item[0])
    phase, due = "idle", None
    ack_ms = model.c.slow_ms if condition == "publisher" else 2
    if pending:
        _, _, claim, ack = pending[0]
        if claim is None:
            phase = "claim"
        elif ack is None:
            phase = "ack"
            due = claim + (ack_ms if claim > -30 * SECOND else 2) * 1_000_000
        else:
            phase = "mark"
    physical = (
        0,
        visits[-1][4] if visits else -60 * SECOND,
        visits,
        rows,
        [p[1] for p in pending],
        phase,
        due,
    )
    physical_hash = hashlib.sha256(repr(physical).encode()).hexdigest()
    enriched = dict(
        physical_digest=physical_hash,
        ack_ms=ack_ms,
        ack_due=due,
        reference_per_s=model.c.reference,
        extra_per_s=model.c.extra if condition == "contention" else 0.0,
        share=model.c.hold,
        eligible=eligible,
        outstanding=[v for v in visits if v[4] > 0],
    )
    return physical_hash, digest(enriched)


def _equal(actual, expected, path):
    if actual != expected:
        raise ValueError(f"{path}: evidence does not match reconstruction")


def audit_root(condition, partition, index, record):
    """Replay all expected branches and compare physical evidence before any scoring."""
    from pathlib import Path
    from experiments.config import read_config

    config = read_config(
        Path(__file__).resolve().parents[1] / "feasibility_pilot/configs/pilot.toml"
    )

    _equal(
        (record["condition"], record["partition"], record["index"]),
        (condition, partition, index),
        "root identity",
    )
    branches = ["hold", "promote"]
    if partition in ("protected", "fixture") and condition == "reference":
        branches.extend(["surprise_hold", "surprise_promote"])
    _equal(set(record["branches"]), set(branches), "branch inventory")
    prefix = _regenerate(condition, partition, index, "prefix", False, config)
    futures = {False: _regenerate(condition, partition, index, "future", False, config)}
    if "surprise_hold" in branches:
        shock = _regenerate(condition, partition, index, "future", True, config)
        _equal([v for v in shock if v[1] != "extra"], futures[False], "surprise coupling")
        futures[True] = shock
    receipts = []
    admissions = 0
    for branch in branches:
        surprise = branch.startswith("surprise")
        future = futures[surprise]
        model = _native_replay(config, condition, branch, prefix, future)
        admitted_hash, admitted = check_admissions(model, prefix, future, branch)
        history = history_from_writes(model.writes)
        _equal(canonical(record["history"]), canonical(history), f"{branch}.history")
        truth = truth_from_visits(model.db.visits)
        compare(record["truth"], truth, f"{branch}.truth")
        physical, state = _cut_state(model, condition)
        evidence = dict(
            physical_digest=physical,
            state_digest=state,
            prefix_digest=digest(prefix),
            history_digest=digest(history),
        )
        _equal(record["evidence"], evidence, f"{branch}.cut")
        counts, breach = windows_from_writes(model.writes)
        windows = [dict(eligible=n, misses=m) for n, m in counts]
        expected = dict(
            windows=windows,
            breach=breach,
            cut_digest=state,
            future_digest=digest(future),
            admissions_digest=admitted_hash,
        )
        _equal(record["branches"][branch], expected, f"{branch}.outcome")
        admissions += admitted
        receipts.append([branch, evidence, truth, expected])
    return dict(
        condition=condition,
        partition=partition,
        index=index,
        branches=len(branches),
        replay_digest=digest(receipts),
        admissions=admissions,
    )


def compare(actual, expected, path):
    """Scientific values have tight tolerances. Counts, decisions and inventory are exact."""
    import math

    if isinstance(expected, dict):
        _equal(set(actual), set(expected), path)
        for key, value in expected.items():
            compare(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        _equal(len(actual), len(expected), path)
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, f"{path}[{i}]")
    elif isinstance(expected, float):
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isfinite(expected)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError(f"{path}: {actual!r} differs from {expected!r}")
    else:
        _equal(actual, expected, path)


def audit_analysis(
    development_rows, protected_score_rows, fitted, scores, fold_seed, bootstrap_seed
):
    """Apply all stored fits and reconstruct the independent conditioned-on-validity verdict."""
    from .audit_models import audit_fits
    from .audit_statistics import audit_scores

    selected = audit_fits(development_rows, protected_score_rows, fitted, fold_seed)
    verdict, comparisons = audit_scores(protected_score_rows, selected, scores, bootstrap_seed)
    # Four public models × (five scalar forecasts and one condition vector),
    # plus three scalar action forecasts from the evaluator-only diagnostic.
    checked = (4 * 6 + 3) * (len(development_rows) + len(protected_score_rows))
    return dict(
        verdict=verdict,
        development_roots=len(development_rows),
        protected_roots=len(protected_score_rows),
        predictions_checked=checked,
        comparisons_checked=comparisons,
    )
