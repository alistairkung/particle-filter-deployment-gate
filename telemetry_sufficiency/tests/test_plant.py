from dataclasses import replace

import numpy as np
import pytest

from experiments.config import MS, SECOND
from simulator.inputs import arrivals, rng
from telemetry_sufficiency import plant


def test_publisher_receives_nominal_reference_queries_in_both_prefix_segments():
    raw = plant.raw_tape("publisher", "fixture", 0, "prefix")
    key = (57012026, "outbox-sufficiency-v1", "publisher", "fixture", 0, "prefix")
    times = arrivals(rng(*key, "reference-arrivals"), 3.5, 60)
    draws = rng(*key, "reference-work").exponential(40 * MS, len(times))
    expected = [
        (t - 60 * SECOND, "reference", -(i + 1), max(1, round(x)))
        for i, (t, x) in enumerate(zip(times, draws))
    ]
    actual = [r for r in raw if r[1] == "reference"]
    assert actual == expected
    assert min(r[0] for r in actual) < -30 * SECOND < max(r[0] for r in actual)
    assert all(r[1] != "extra" for r in raw)
    extra = [
        r
        for r in plant.raw_tape("contention", "fixture", 7, "prefix")
        if r[1] == "extra"
    ]
    assert extra and min(r[0] for r in extra) > -30 * SECOND
    with pytest.raises(ValueError):
        plant.raw_tape("reference/development", "fixture", 0, "prefix")


def test_threshold_and_atomic_insert_on_paired_business_draws():
    uniforms = [0.1, 0.25, 0.4, 0.6, 0.99]
    raw = [(i + 1, "write", i, u, 1.0) for i, u in enumerate(uniforms)]
    raw.append(
        (
            9,
            "reference",
            0,
            331,
        )
    )
    hold = plant.admissions(raw, 0.25)
    promote = plant.admissions(raw, 0.60)
    assert [r[3] for r in hold] == [16 * MS, 12 * MS, 12 * MS, 12 * MS, 12 * MS, 331]
    assert [r[3] for r in promote] == [16 * MS, 16 * MS, 16 * MS, 12 * MS, 12 * MS, 331]
    assert [r[4] for r in hold] == [True, False, False, False, False, False]
    assert [r[4] for r in promote] == [True, True, True, False, False, False]


@pytest.mark.parametrize("began_delta,expected_ms", [(-1, 2), (0, 2), (1, 50)])
def test_ack_wait_started_at_onset_keeps_assigned_duration(began_delta, expected_ms):
    onset = -30 * SECOND
    # Zero business + atomic insert + one claim visit = 2 ms before ACK begins.
    prefix = [(onset - 2 * MS + began_delta, "write", -1, 0.9, 0.0)]
    model, cut = plant.start_prefix("publisher", prefix)
    w = model.writes[-1]
    assert w.claim == onset + began_delta
    assert w.ack - w.claim == expected_ms * MS
    assert cut["truth"]["demand"] == cut["truth"]["unfinished"] == 0
    assert len(cut["history"]) == 12


def test_cut_retains_acknowledged_unmarked_rows_and_excludes_uncommitted_writes():
    # One eligible row has acknowledged the sink but its mark waits behind a long query.
    prefix = [
        (-5 * MS, "write", -2, 0.9, 0.0),
        (-2 * MS, "reference", -1, 100 * MS),
        (-MS, "write", -1, 0.1, 0.0),
    ]
    model, cut = plant.start_prefix("reference", prefix)
    assert model.writes[-2].ack == -MS and model.writes[-2].mark is None
    assert model.writes[-1].commit is None
    assert cut["history"][-1] == dict(
        interval_start_ns=-5 * SECOND,
        interval_end_ns=0,
        available_at_ns=0,
        pending_count=1,
        oldest_pending_age_s=0.004,
        publication_throughput_per_s=0.0,
        completed_delay_median_s=None,
    )
    assert cut["truth"] == dict(
        demand=0.1 / 30,
        unfinished=0.1,
        duration_ns=30 * SECOND,
        query_work_ns=100 * MS,
        queued_ns=2 * MS,
        active_residual_ns=98 * MS,
    )
    model.advance(SECOND)
    assert model.writes[-2].mark is not None and model.writes[-1].commit is not None
    assert cut["history"][-1]["pending_count"] == 1


def test_cut_and_endpoint_match_uninterrupted_native_pilot_and_passive_control():
    from simulator.model import Model
    from experiments.observations import horizon_records, score_window

    prefix = plant.raw_tape("reference", "fixture", 5, "prefix")
    future = plant.raw_tape("reference", "fixture", 5, "future")
    cfg = plant.CONFIG
    direct = Model(cfg, -60 * SECOND)
    direct.schedule(
        plant.admissions(prefix, cfg.hold) + plant.admissions(future, cfg.promote)
    )
    direct.advance(120 * SECOND)
    cohort = [
        (i, w)
        for i, write in direct.writes.items()
        if (w := score_window(write.arrival, write.eligible, cfg)) is not None
    ]
    expected = horizon_records(direct, cohort)
    for passive in (True, False):
        model, _, _, cut = plant.replay(
            "reference", "fixture", 5, "promote", passive=passive
        )
        assert model.db.visits == direct.db.visits and model.writes == direct.writes
        assert model.physical_digest(True) == direct.physical_digest(True)
        assert plant.endpoint(model)["windows"] == [
            {"eligible": w["eligible"], "misses": w["misses"]} for w in expected
        ]
        assert bool(cut["history"]) is passive


def test_paired_root_surprise_and_action_order_share_only_permitted_draws():
    ordinary = plant.simulate_root("reference", "fixture", 3)
    reverse = plant.simulate_root(
        "reference", "fixture", 3, order=list(reversed(ordinary["branches"]))
    )
    assert ordinary == reverse
    branches = ordinary["branches"]
    assert branches["hold"]["future_digest"] == branches["promote"]["future_digest"]
    assert (
        branches["surprise_hold"]["future_digest"]
        == branches["surprise_promote"]["future_digest"]
    )
    assert (
        branches["hold"]["future_digest"] != branches["surprise_hold"]["future_digest"]
    )
    assert len({v["cut_digest"] for v in branches.values()}) == 1
    assert len(set(ordinary["history"][0])) == 7


@pytest.mark.parametrize(
    "arrival,phase,ack_due",
    [
        (-8 * MS, "idle", None),
        (-MS, "claim", None),
        (-2 * MS, "ack", 2 * MS),
        (-4 * MS, "mark", None),
        (0, "idle", None),
    ],
)
def test_exact_cut_phase_ack_due_and_tied_zero_arrival(arrival, phase, ack_due):
    prefix = [(arrival, "write", -1, 0.25, 0.0)]
    model, cut = plant.start_prefix("reference", prefix)
    assert model.phase == phase and model.ack_due == ack_due
    assert model.writes[-1].java is False
    reading = cut["history"][-1]
    assert reading["pending_count"] == int(phase in ("claim", "ack", "mark"))
    if phase == "claim":
        assert reading["oldest_pending_age_s"] == 0
    if phase == "idle" and arrival == 0:
        assert cut["truth"]["active_residual_ns"] == MS
        assert model.writes[-1].commit is None


def test_active_ack_at_cut_keeps_publisher_timeout_under_both_actions():
    prefix = [(-3 * MS, "write", -1, 0.3, 0.0)]
    future = [(MS, "write", 0, 0.3, 0.0)]
    hold, *_ = plant.replay(
        "publisher", "fixture", 0, "hold", prefix=prefix, future=future
    )
    promote, *_ = plant.replay(
        "publisher", "fixture", 0, "promote", prefix=prefix, future=future
    )
    assert hold.writes[-1] == promote.writes[-1]
    assert hold.writes[-1].ack == 49 * MS
    assert hold.writes[0].java is False and promote.writes[0].java is True


def test_equal_work_arms_have_identical_physics_and_endpoints():
    cfg = replace(plant.CONFIG, java_ms=plant.CONFIG.ruby_ms)
    hold, *_ = plant.replay("contention", "fixture", 11, "hold", config=cfg)
    promote, *_ = plant.replay("contention", "fixture", 11, "promote", config=cfg)
    assert hold.physical_digest() == promote.physical_digest()
    assert hold.physical_digest(True) != promote.physical_digest(True)
    assert plant.endpoint(hold) == plant.endpoint(promote)


def test_ineligible_writes_do_not_publish_and_ack_only_rows_stay_visible():
    from telemetry_sufficiency.observation import Collector

    collector = Collector()
    model = plant.ObservedModel(plant.CONFIG, collector)
    end = -55 * SECOND
    model.schedule(
        [
            (end - 5 * MS, "write", -2, MS, False, True),
            (end - 2 * MS, "reference", -1, 10 * MS, False),
            (end - MS, "write", -1, MS, False, False),
        ]
    )
    model.advance(end)
    assert model.writes[-2].ack == end - MS and model.writes[-2].mark is None
    reading = collector.seal(end)
    assert (
        reading.pending_count,
        reading.oldest_pending_age_s,
        reading.publication_throughput_per_s,
        reading.completed_delay_median_s,
    ) == (1, 0.004, 0, None)
    model.advance(end + 5 * SECOND)
    reading = collector.seal(end + 5 * SECOND)
    assert model.writes[-1].commit is not None and model.writes[-1].mark is None
    assert (reading.pending_count, reading.publication_throughput_per_s) == (0, 0.2)


def test_changed_future_action_query_shock_and_drain_preserve_live_cut_bytes():
    import json

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    prefix = [
        (-5 * MS, "write", -2, 0.4, 0.0),
        (-2 * MS, "reference", -1, 100 * MS),
        (-MS, "write", -1, 0.1, 0.0),
    ]
    _, cut = plant.start_prefix("reference", prefix)
    sealed = canonical(cut["history"])
    for action, source in (
        ("hold", "reference"),
        ("promote", "reference"),
        ("surprise_hold", "extra"),
    ):
        future = [(MS, "write", 0, 0.3, 6.0), (2 * MS, source, 0, 5 * SECOND)]
        model, _, _, sibling = plant.replay(
            "reference", "fixture", 8, action, prefix=prefix, future=future
        )
        model.advance((120 + 39) * SECOND)
        assert all(w.mark is not None for w in model.writes.values())
        assert sibling["physical_digest"] == cut["physical_digest"]
        assert canonical(sibling["history"]) == sealed
        # Independent history rebuilt from horizon writes, ignoring post-bin timestamps.
        rows = []
        for end in range(-55 * SECOND, 1, 5 * SECOND):
            left = end - 5 * SECOND
            pending = [
                w.commit
                for w in model.writes.values()
                if w.eligible
                and w.commit is not None
                and w.commit <= end
                and (w.mark is None or w.mark > end)
            ]
            delays = [
                (w.mark - w.commit)
                for w in model.writes.values()
                if w.eligible and w.mark is not None and left < w.mark <= end
            ]
            rows.append(
                dict(
                    interval_start_ns=left,
                    interval_end_ns=end,
                    available_at_ns=end,
                    pending_count=len(pending),
                    oldest_pending_age_s=(end - min(pending)) / SECOND
                    if pending
                    else 0.0,
                    publication_throughput_per_s=len(delays) / 5,
                    completed_delay_median_s=float(np.median(delays)) / SECOND
                    if delays
                    else None,
                )
            )
        assert canonical(rows) == sealed


def test_actual_admission_binding_detects_changed_draws_and_query_work():
    m, prefix, future, _ = plant.replay("reference", "fixture", 1, "hold")
    for source in ("write", "reference"):
        corrupt = list(future)
        j = next(i for i, r in enumerate(corrupt) if r[1] == source)
        corrupt[j] = (*corrupt[j][:-1], corrupt[j][-1] + 17)
        with pytest.raises(ValueError, match="admissions"):
            plant.verify_admissions(m, prefix, corrupt, plant.CONFIG.hold)


def test_parallel_fixture_roots_match_serial_records():
    from concurrent.futures import ThreadPoolExecutor

    def run(index):
        return plant.simulate_root("publisher", "fixture", index)

    indices = [21, 22]
    serial = [run(i) for i in indices]
    with ThreadPoolExecutor(max_workers=2) as workers:
        parallel = list(workers.map(run, reversed(indices)))
    assert list(reversed(parallel)) == serial


def test_final_bin_includes_mark_and_excludes_uncommitted_write_at_zero():
    cfg = plant.CONFIG
    # The mark completes at zero; the other write admits at zero and cannot be committed yet.
    prefix = [(-5 * MS, "write", -2, 0.8, 0.0), (0, "write", -1, 0.5, 0.0)]
    model, cut = plant.start_prefix("reference", prefix)
    assert model.writes[-2].mark == 0 and model.writes[-1].commit is None
    reading = cut["history"][-1]
    assert (
        reading["pending_count"] == 0 and reading["publication_throughput_per_s"] == 0.2
    )
    assert (
        reading["completed_delay_median_s"]
        == (cfg.claim_ms + cfg.ack_ms + cfg.mark_ms) * MS / SECOND
    )


def test_cut_query_demand_excludes_left_end_and_includes_zero_even_if_queued():
    prefix = [
        (-30 * SECOND, "reference", -3, SECOND),
        (-30 * SECOND + 1, "extra", -2, 2 * SECOND),
        (-MS, "write", -1, 0.9, 1.0),
        (0, "reference", -1, 3 * SECOND),
    ]
    _, cut = plant.start_prefix("contention", prefix)
    assert cut["truth"]["query_work_ns"] == 5 * SECOND
    assert cut["truth"]["queued_ns"] == 3 * SECOND
    assert cut["truth"]["active_residual_ns"] == 11 * MS
    # Same primitive tape can yield the same physical state with different ongoing input rates.
    _, nominal = plant.start_prefix("reference", prefix)
    assert nominal["physical_digest"] == cut["physical_digest"]
    assert nominal["state_digest"] != cut["state_digest"]
