"""C0 analytical calibration and C1 physical placebo validation."""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import simpy
from scipy import stats

from experiments.config import MS, SECOND
from experiments.runner import arms, run_root
from experiments.storage import save_samples
from simulator.inputs import inputs, rng
from simulator.model import Model, QUERY, SerialDB


def isolated_wait_ms(ticks, work, discard):
    """Score completed visits from the common DB model, retaining warm-up work."""
    env = simpy.Environment()
    db = SerialDB(env)
    waits = []

    def complete(_):
        # Identical FIFO admission and completion orders give a stable visit ID.
        _, _, offered, start, end = db.visits[len(waits)]
        if end != env.now:
            raise AssertionError("Isolated DB visit completed off schedule")
        waits.append(start - offered)

    def arrivals():
        for identity, (tick, demand) in enumerate(zip(ticks, work, strict=True)):
            yield env.timeout(int(tick) - env.now)
            db.visit(QUERY, identity, int(demand), complete)

    env.process(arrivals())
    env.run()
    if len(waits) != len(ticks) or not 0 <= discard < len(waits):
        raise ValueError("Isolated queue cohort is incomplete")
    return sum(waits[discard:]) / ((len(waits) - discard) * MS)


def _queue_root(task):
    config, distribution, rate, root = task
    key = (config.seed, "C0", distribution, rate, root)
    count = config.queue_warmup + config.queue_scored
    intervals = rng(*key, "arrival").exponential(SECOND / rate, count)
    ticks = np.maximum(1, np.rint(intervals).astype(np.int64)).cumsum()
    mean_work = SECOND / config.queue_service_rate
    if distribution == "M/M/1":
        work = rng(*key, "service").exponential(mean_work, count)
        work = np.maximum(1, np.rint(work).astype(np.int64))
    else:
        work = np.full(count, round(mean_work), dtype=np.int64)
    return isolated_wait_ms(ticks, work, config.queue_warmup)


def queue_cell(distribution, rate, service_rate, seed_means_ms, tolerance):
    expected = 1000 * rate / (service_rate * (service_rate - rate))
    if distribution == "M/D/1":
        expected /= 2
    mean = float(np.mean(seed_means_ms))
    n = len(seed_means_ms)
    half_width = float(
        stats.t.ppf(0.975, n - 1) * np.std(seed_means_ms, ddof=1) / n**0.5
    )
    low, high = mean - half_width, mean + half_width
    target_low, target_high = expected * (1 - tolerance), expected * (1 + tolerance)
    if target_low <= low and high <= target_high:
        status = "pass"
    elif low > target_high or high < target_low:
        status = "disagreement"
    else:
        status = "insufficient precision"
    return dict(
        distribution=distribution,
        rate=rate,
        expected_ms=expected,
        mean_ms=mean,
        interval_ms=[low, high],
        tolerance_ms=[target_low, target_high],
        passed=status == "pass",
        status=status,
        seed_means_ms=seed_means_ms,
    )


def replay_check(config):
    """A long query, waiting write and unfinished sink acknowledgement span the cut."""
    prefix = [
        (-3 * MS, "write", -2, MS, False),
        (-MS // 2, "reference", -1, 10 * MS, False),
        (-MS // 4, "write", -1, MS, True),
    ]
    future = [(2 * MS, "write", 0, MS, True)]
    uninterrupted = Model(config, -4 * MS)
    uninterrupted.schedule(prefix + future)
    uninterrupted.advance(100 * MS)

    replay = Model(config, -4 * MS)
    replay.schedule(prefix)
    replay.advance(0)
    active_query = next(visit for visit in replay.db.visits if visit[0] == QUERY)
    cut = dict(
        cut_digest=replay.physical_digest(True),
        cut_phase=replay.phase,
        inherited_ack_due_ns=replay.ack_due,
        active_query_residual_ns=active_query[-1],
        outstanding_write=replay.writes[-1].commit is None,
        pending_ids=list(replay.pending),
    )
    replay.schedule(future)
    replay.advance(100 * MS)
    end = replay.physical_digest(True)
    return dict(
        **cut,
        passed=(
            cut["cut_phase"] == "ack"
            and cut["outstanding_write"]
            and cut["pending_ids"] == [-2]
            and cut["inherited_ack_due_ns"] == MS
            and cut["active_query_residual_ns"] == 9_500_000
            and end == uninterrupted.physical_digest(True)
        ),
        replay_end=end,
        uninterrupted_end=uninterrupted.physical_digest(True),
    )


def publisher_release_check(config):
    model = Model(config, 0, slow=True)
    model.schedule([(MS, "write", 0, MS, False), (8 * MS, "write", 1, MS, True)])
    model.advance(60 * MS)
    first = model.writes[0]
    second = model.writes[1]
    return dict(
        passed=(
            first.claim == 3 * MS
            and first.ack == 53 * MS
            and second.commit == 9 * MS
            and first.mark == 54 * MS
        ),
        claim_ns=first.claim,
        intervening_commit_ns=second.commit,
        ack_ns=first.ack,
        mark_ns=first.mark,
    )


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n")
    temp.replace(path)


def _placebo_root(config, out_dir, root):
    try:
        branches, samples = run_root(config, "C1", root)
        reference = branches["hold"]
        passed = all(
            arm["windows"] == reference["windows"]
            and arm["breach"] == reference["breach"]
            for arm in branches.values()
        )
        path = out_dir / "samples" / "C1" / f"{root:04d}.npz"
        sample_digest = save_samples(path, samples)
        row = dict(
            root_id=root, passed=passed, arms=branches, sample_digest=sample_digest
        )
    except (AssertionError, ValueError) as error:
        row = dict(root_id=root, passed=False, error=repr(error))
    if not row["passed"]:
        row["divergence"] = _first_placebo_difference(config, root)
        _write_json(out_dir / "validation" / f"failed-placebo-{root}.json", row)
    return row


def _trace_events(model):
    """Construct a small diagnostic comparison only if a trace hash disagrees."""
    times = []
    for kind, identity, request, start, end in model.db.visits:
        times.extend(
            [
                (request, "request", kind, identity),
                (start, "start", kind, identity),
                (end, "end", kind, identity),
            ]
        )
    for identity, write in model.writes.items():
        for stage in ("commit", "claim", "ack", "mark"):
            stamp = getattr(write, stage)
            if stamp is not None:
                times.append((stamp, stage, -1, identity))
    return sorted(event for event in times if event[0] <= model.env.now)


def _first_placebo_difference(config, root):
    from itertools import zip_longest
    from experiments.observations import finish_latencies, score_window

    prefix = inputs(config, "C1", root, "prefix", config.hold)
    reference = None
    for arm in arms(config, "C1"):
        model = Model(config, -config.warmup * SECOND)
        model.schedule(prefix)
        model.advance(0)
        model.schedule(inputs(config, "C1", root, "future", arm.share))
        model.advance(config.horizon * SECOND)
        cohort = [
            (identity, window)
            for identity, w in model.writes.items()
            if (window := score_window(w.arrival, w.eligible, config)) is not None
        ]
        finish_latencies(model, cohort)
        observed = _trace_events(model)
        if reference is None:
            reference = observed
            continue
        for expected, actual in zip_longest(reference, observed):
            if actual != expected:
                return dict(arm=arm.name, reference=expected, actual=actual)
    return dict(
        message="No timestamp divergence; inspect the recorded scoring or state failure."
    )


def validate(config, out_dir, workers=1):
    """Run the isolated queue prerequisite and the physical controls, saving evidence."""
    out_dir = Path(out_dir)
    checks = dict(replay=replay_check(config), release=publisher_release_check(config))
    _write_json(out_dir / "validation/checks.json", checks)
    cells = [
        (distribution, rate)
        for distribution in ("M/M/1", "M/D/1")
        for rate in config.queue_rates
    ]
    tasks = [
        (config, distribution, rate, root)
        for distribution, rate in cells
        for root in range(config.queue_roots)
    ]
    placebo_fn = partial(_placebo_root, config, out_dir)

    def run(map_tasks):
        means = list(map_tasks(_queue_root, tasks))
        queue = []
        for index, (distribution, rate) in enumerate(cells):
            offset = index * config.queue_roots
            cell = queue_cell(
                distribution,
                rate,
                config.queue_service_rate,
                means[offset : offset + config.queue_roots],
                config.queue_tolerance,
            )
            queue.append(cell)
        _write_json(out_dir / "validation/queue.json", queue)
        placebo = list(map_tasks(placebo_fn, range(config.placebo_roots)))
        _write_json(out_dir / "validation/placebo.json", placebo)
        results = [*queue, *placebo, *checks.values()]
        return dict(
            passed=all(item["passed"] for item in results),
            queue=queue,
            placebo=placebo,
            checks=checks,
        )

    if workers == 1:
        return run(map)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return run(pool.map)
