"""Study workload tapes, passive plant adapter and evaluator-owned paired replay."""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

from experiments.config import SECOND, read_config
from experiments.observations import horizon_records, score_window
from simulator.model import QUERY, WRITE
from simulator.replay import ObservedModel, replay_prefix, run_future
from simulator.workload import admissions as routed_admissions, primitive_tape

from . import CONDITIONS

NAMESPACE = "outbox-sufficiency-v1"
CONFIG = read_config(
    Path(__file__).resolve().parents[1] / "feasibility_pilot/configs/pilot.toml"
)
ONSET = -30 * SECOND


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _key(condition, partition, index):
    if condition not in CONDITIONS or partition not in (
        "development",
        "protected",
        "fixture",
    ):
        raise ValueError("Unknown condition or partition")
    if type(index) is not int or not 0 <= index < 400:
        raise ValueError("Root index must be an integer in [0,399]")
    return condition, partition, index


def raw_tape(condition, partition, index, phase, *, surprise=False, config=CONFIG):
    """Materialise primitive application draws and assigned, route-independent query work."""
    key = _key(condition, partition, index)
    if phase not in ("prefix", "future"):
        raise ValueError("Unknown tape phase")
    if surprise and (
        condition != "reference" or partition == "development" or phase != "future"
    ):
        raise ValueError(
            "Surprise is a future of protected reference or reference fixtures"
        )

    return primitive_tape(
        config,
        (NAMESPACE, *key),
        phase,
        reference_rate=config.reference,
        extra_rate=config.extra if condition == "contention" or surprise else 0,
        extra_after_ns=ONSET if phase == "prefix" else None,
    )


def admissions(tape, share, config=CONFIG):
    return routed_admissions(tape, share, config)


def _cut_truth(visits):
    query_work = sum(
        end - start
        for kind, _, queued, start, end in visits
        if kind == QUERY and ONSET < queued <= 0
    )
    active = sum(end for _, _, _, start, end in visits if start <= 0 < end)
    queued = sum(end - start for _, _, _, start, end in visits if start > 0)
    return dict(
        demand=query_work / -ONSET,
        unfinished=(active + queued) / SECOND,
        query_work_ns=query_work,
        duration_ns=-ONSET,
        queued_ns=queued,
        active_residual_ns=active,
    )


def start_prefix(condition, prefix, *, config=CONFIG, passive=True, deliveries=None):
    """Stop after all events at zero; freeze the public history before supplying any future."""
    model, readings = replay_prefix(
        config,
        prefix,
        config.hold,
        passive=passive,
        deliveries=deliveries,
        ack_change=(ONSET, config.slow_ms) if condition == "publisher" else None,
    )
    history = [asdict(r) for r in readings]
    physical = model.physical_digest(with_routes=True)
    state = dict(
        physical_digest=physical,
        ack_ms=model.c.ack_ms,
        ack_due=model.ack_due,
        reference_per_s=config.reference,
        extra_per_s=config.extra if condition == "contention" else 0.0,
        share=config.hold,
        eligible=[w.eligible for w in model.writes.values()],
        outstanding=[v for v in model.db.visits if v[-1] > 0],
    )
    cut = dict(
        history=history,
        truth=_cut_truth(model.db.visits),
        physical_digest=physical,
        state_digest=_digest(state),
        prefix_digest=_digest(prefix),
        history_digest=_digest(history),
    )
    return model, cut


def verify_admissions(model, prefix, future, share):
    """Bind each raw request and query to its actual external FIFO visit and write."""
    expected = admissions(prefix, model.c.hold, model.c) + admissions(
        future, share, model.c
    )
    actual = []
    for kind, identity, tick, start, end in model.db.visits:
        if kind == WRITE:
            w = model.writes[identity]
            if w.arrival != tick or w.demand != end - start:
                raise ValueError("Write and FIFO admission disagree")
            actual.append((tick, "write", identity, w.demand, w.java))
        elif kind == QUERY:
            source = "extra" if identity % 2 else "reference"
            actual.append((tick, source, identity // 2, end - start, False))
    if actual != expected:
        raise ValueError("Actual external admissions differ from routed primitives")
    return _digest(actual)


def _branch_names(condition, partition):
    ordinary = ("hold", "promote")
    if condition == "reference" and partition in ("protected", "fixture"):
        return ordinary + ("surprise_hold", "surprise_promote")
    return ordinary


def replay(
    condition,
    partition,
    index,
    branch,
    *,
    prefix=None,
    future=None,
    config=CONFIG,
    passive=True,
    deliveries=None,
):
    """Replay one branch from a live prefix and freeze at the horizon, before drain."""
    _key(condition, partition, index)
    if branch not in _branch_names(condition, partition):
        raise ValueError("Unknown or unauthorised branch")
    if prefix is None:
        prefix = raw_tape(condition, partition, index, "prefix", config=config)
    model, cut = start_prefix(
        condition, prefix, config=config, passive=passive, deliveries=deliveries
    )
    if future is None:
        future = raw_tape(
            condition,
            partition,
            index,
            "future",
            surprise=branch.startswith("surprise_"),
            config=config,
        )
    share = config.promote if branch.endswith("promote") else config.hold
    run_future(model, future, share)
    model.admissions_digest = verify_admissions(model, prefix, future, share)
    return model, prefix, future, cut


def endpoint(model):
    c = model.c
    if model.env.now != c.horizon * SECOND:
        raise ValueError("Freeze outcomes exactly at the horizon")
    cohort = [
        (i, window)
        for i, w in model.writes.items()
        if (window := score_window(w.arrival, w.eligible, c)) is not None
    ]
    records = horizon_records(model, cohort)
    return dict(
        windows=[dict(eligible=r["eligible"], misses=r["misses"]) for r in records],
        breach=int(any(r["breach"] for r in records)),
    )


def simulate_root(condition, partition, index, *, order=None, config=CONFIG):
    """Run the complete, matched set of continuations for one independent root."""
    names = _branch_names(condition, partition)
    order = names if order is None else tuple(order)
    if sorted(order) != sorted(names):
        raise ValueError("Every expected branch must be run exactly once")
    prefix = raw_tape(condition, partition, index, "prefix", config=config)
    futures = {False: raw_tape(condition, partition, index, "future", config=config)}
    if "surprise_hold" in names:
        futures[True] = raw_tape(
            condition, partition, index, "future", surprise=True, config=config
        )
        if [r for r in futures[True] if r[1] != "extra"] != futures[False]:
            raise ValueError("Surprise changed ordinary future primitives")
    future_digests = {scenario: _digest(tape) for scenario, tape in futures.items()}
    shared_cut = None
    branches = {}
    for name in order:
        surprise = name.startswith("surprise_")
        model, _, _, cut = replay(
            condition,
            partition,
            index,
            name,
            prefix=prefix,
            future=futures[surprise],
            config=config,
        )
        if shared_cut is None:
            shared_cut = cut
        elif cut != shared_cut:
            raise ValueError(
                "Prefix replay changed physical state or available history"
            )
        branches[name] = dict(
            **endpoint(model),
            cut_digest=cut["state_digest"],
            future_digest=future_digests[surprise],
            admissions_digest=model.admissions_digest,
        )
    history = shared_cut.pop("history")
    truth = shared_cut.pop("truth")
    return dict(
        condition=condition,
        partition=partition,
        index=index,
        history=history,
        truth=truth,
        evidence=shared_cut,
        branches=branches,
    )


def example_trace(condition, partition, index, branch):
    """Private native evidence for a predesignated example root, including incomplete writes."""
    model, prefix, future, cut = replay(condition, partition, index, branch)
    return dict(
        condition=condition,
        partition=partition,
        index=index,
        branch=branch,
        prefix=prefix,
        future=future,
        visits=model.db.visits,
        writes=[[i, asdict(w)] for i, w in model.writes.items()],
        ack_due=model.ack_due,
        phase=model.phase,
        cut_digest=cut["state_digest"],
        horizon_ns=model.env.now,
    )
