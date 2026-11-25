from particle_filter.validation.development_replay import replay_development
from simulator.config import SECOND
from telemetry_sufficiency.plant import raw_tape, start_prefix


def test_interval_replay_matches_the_frozen_development_prefix():
    intervals = replay_development("publisher", 2)
    tape = raw_tape("publisher", "development", 2, "prefix")
    model, cut = start_prefix("publisher", tape)
    assert len(intervals) == 12
    assert [i.regime for i in intervals] == ["A"] * 6 + ["C"] * 6
    assert intervals[0].start.work_ns == intervals[0].start.pending == 0
    assert [i.end for i in intervals[:-1]] == [i.start for i in intervals[1:]]
    assert [i.reading.pending_count for i in intervals] == [
        r["pending_count"] for r in cut["history"]
    ]
    assert intervals[-1].end.work_ns / SECOND == cut["truth"]["unfinished"]
    assert intervals[-1].end.phase == model.phase


def test_reconstruction_preserves_admitted_claim_and_insert_costs_and_fifo_order():
    from numpy.random import default_rng
    from particle_filter.model.physical import Boundary, Closure, Visit
    from simulator.config import MS
    from simulator.model import CLAIM, QUERY, WRITE

    donor = Boundary(1, (Visit(QUERY, 5), Visit(WRITE, MS), Visit(CLAIM, MS)), "claim")
    closure = Closure([("A", donor)])
    recovered, fallback = closure.reconstruct("A", 2 * MS + 12, 3, default_rng(0))
    assert not fallback
    assert recovered == Boundary(
        3, (Visit(QUERY, 12), Visit(WRITE, MS), Visit(CLAIM, MS)), "claim"
    )


def test_reconstructed_work_finishes_before_new_claim_enters_and_inflight_ack_keeps_duration():
    from particle_filter.model.physical import Boundary, Visit, replay_hypothesis
    from simulator.config import MS
    from simulator.model import MARK, WRITE

    # Two admitted visits (one ongoing mark, one write) produce exactly one old mark and one commit.
    start = Boundary(1, (Visit(MARK, MS), Visit(WRITE, 2 * MS)), "mark")
    step = replay_hypothesis(start, "A", [], share=0.25, horizon_ns=3 * MS)
    assert step.marks == 1
    assert step.end.pending == 1
    assert step.end.phase == "claim" and step.end.work_ns == MS
    # A slow acknowledgement already in progress keeps its residual under the nominal regime.
    waiting = replay_hypothesis(
        Boundary(1, (), "ack", 30 * MS), "A", [], share=0.25, horizon_ns=29 * MS
    )
    assert waiting.end == Boundary(1, (), "ack", MS)


def test_phase_challenge_conditions_on_revealed_phase_with_same_work_and_pending():
    from numpy.random import default_rng
    from particle_filter.validation.development_replay import PhaseDiagnostic
    from particle_filter.model.physical import Boundary, Visit
    from simulator.config import MS
    from simulator.model import CLAIM, QUERY

    donors = [
        ("A", Boundary(1, (Visit(CLAIM, MS),), "claim")),
        ("A", Boundary(1, (Visit(QUERY, MS),), "ack", MS)),
    ]
    diagnostic = PhaseDiagnostic(donors)
    assert diagnostic.reconstruct("A", MS, 2, "ack", 19, default_rng(12)) == Boundary(
        2, (Visit(QUERY, MS),), "ack", 19
    )
    assert diagnostic.reconstruct("A", MS, 2, "mark", 0, default_rng(12)) is None


def test_restoring_each_publisher_phase_reproduces_frozen_action_endpoints_on_development_tapes():
    from particle_filter.model.physical import anonymous_boundary, replay_hypothesis
    from telemetry_sufficiency.plant import CONFIG, endpoint, replay

    # The four development cuts cover idle, claim, mark and off-DB acknowledgement.
    for condition, index in [
        ("reference", 0),
        ("reference", 3),
        ("contention", 0),
        ("publisher", 0),
    ]:
        interval = replay_development(condition, index)[-1]
        for branch, share in [("hold", CONFIG.hold), ("promote", CONFIG.promote)]:
            full, _, future, _ = replay(condition, "development", index, branch)
            expected = endpoint(full)
            result = replay_hypothesis(
                interval.end,
                interval.regime,
                future,
                share=share,
                horizon_ns=120 * SECOND,
            )
            assert result.breach == expected["breach"]
            assert result.eligible == tuple(w["eligible"] for w in expected["windows"])
            assert result.misses == tuple(w["misses"] for w in expected["windows"])
            assert result.end == anonymous_boundary(full)


def test_deadlines_at_window_edges_and_five_percent_breach_threshold():
    from dataclasses import replace
    from particle_filter.model.physical import Boundary, CONFIG, replay_hypothesis
    from simulator.config import MS

    # First write is late by one tick; second commits at its deadline.
    # A deadline at exactly 30 s is in the first window and a deadline at 120 s is eligible.
    arrivals = [i * SECOND for i in range(1, 20)] + [
        30 * SECOND - 250 * MS,
        120 * SECOND - 250 * MS,
        120 * SECOND - 10,
    ]
    tape = [
        (t, "write", i, 1.0, 249.000001 if i == 0 else 249.0)
        for i, t in enumerate(arrivals)
    ]
    config = replace(CONFIG, ruby_ms=1, java_ms=1)
    result = replay_hypothesis(
        Boundary(), "A", tape, share=0.25, horizon_ns=120 * SECOND, config=config
    )
    assert result.eligible == (20, 0, 0, 1)
    assert result.misses == (1, 0, 0, 0)
    assert not result.breach
    result = replay_hypothesis(
        Boundary(),
        "A",
        tape[:2] + tape[3:],
        share=0.25,
        horizon_ns=120 * SECOND,
        config=config,
    )
    assert result.eligible[0] == 19 and result.breach


def test_matched_forecast_is_reproducible_and_equal_route_work_matches_per_future():
    from dataclasses import replace
    from numpy.random import default_rng
    from particle_filter.model.physical import Boundary, CONFIG, matched_pair

    equal_work = replace(CONFIG, java_ms=CONFIG.ruby_ms)
    pair = matched_pair(Boundary(), "B", default_rng(44), config=equal_work)
    assert pair[0] == pair[1]
    assert pair == matched_pair(Boundary(), "B", default_rng(44), config=equal_work)


def test_sparse_cell_fallback_accounts_for_only_admitted_work_and_retains_known_wait():
    from numpy.random import default_rng
    from particle_filter.model.physical import Boundary, Closure, Visit
    from simulator.config import MS
    from simulator.model import CLAIM, QUERY

    closure = Closure([("A", Boundary(1, (Visit(CLAIM, MS),), "claim"))])
    prior = Boundary(2, (), "ack", 30 * MS)
    recovered, fallback = closure.reconstruct("A", 7, 2, default_rng(1), prior=prior)
    assert fallback
    assert recovered == Boundary(2, (Visit(QUERY, 7),), "ack", 30 * MS)
    empty, fallback = closure.reconstruct("B", 0, 0, default_rng(1))
    assert empty == Boundary() and not fallback
