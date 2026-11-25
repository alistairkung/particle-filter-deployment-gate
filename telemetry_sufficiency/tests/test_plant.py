from dataclasses import asdict, replace

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
    expected = [(t - 60 * SECOND, "reference", -(i+1), max(1, round(x)))
                for i, (t, x) in enumerate(zip(times, draws))]
    actual = [r for r in raw if r[1] == "reference"]
    assert actual == expected
    assert min(r[0] for r in actual) < -30 * SECOND < max(r[0] for r in actual)
    assert all(r[1] != "extra" for r in raw)
    extra = [r for r in plant.raw_tape("contention", "fixture", 7, "prefix") if r[1] == "extra"]
    assert extra and min(r[0] for r in extra) > -30 * SECOND
    with pytest.raises(ValueError):
        plant.raw_tape("reference/development", "fixture", 0, "prefix")


def test_threshold_and_atomic_insert_on_paired_business_draws():
    uniforms = [0.1, 0.25, 0.4, 0.6, 0.99]
    raw = [(i+1, "write", i, u, 1.0) for i, u in enumerate(uniforms)]
    raw.append((9, "reference", 0, 331, ))
    hold = plant.admissions(raw, 0.25)
    promote = plant.admissions(raw, 0.60)
    assert [r[3] for r in hold] == [16*MS, 12*MS, 12*MS, 12*MS, 12*MS, 331]
    assert [r[3] for r in promote] == [16*MS, 16*MS, 16*MS, 12*MS, 12*MS, 331]
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
    prefix = [(-5*MS, "write", -2, 0.9, 0.0),
              (-2*MS, "reference", -1, 100*MS),
              (-MS, "write", -1, 0.1, 0.0)]
    model, cut = plant.start_prefix("reference", prefix)
    assert model.writes[-2].ack == -MS and model.writes[-2].mark is None
    assert model.writes[-1].commit is None
    assert cut["history"][-1] == dict(
        interval_start_ns=-5*SECOND, interval_end_ns=0, available_at_ns=0,
        pending_count=1, oldest_pending_age_s=0.004,
        publication_throughput_per_s=0.0, completed_delay_median_s=None,
    )
    assert cut["truth"] == dict(demand=0.1 / 30, unfinished=0.1, duration_ns=30*SECOND,
                                query_work_ns=100*MS, queued_ns=2*MS, active_residual_ns=98*MS)
    model.advance(SECOND)
    assert model.writes[-2].mark is not None and model.writes[-1].commit is not None
    assert cut["history"][-1]["pending_count"] == 1


def test_cut_and_endpoint_match_uninterrupted_native_pilot_and_passive_control():
    from simulator.model import Model
    from experiments.observations import horizon_records, score_window

    prefix = plant.raw_tape("reference", "fixture", 5, "prefix")
    future = plant.raw_tape("reference", "fixture", 5, "future")
    cfg = plant.CONFIG
    direct = Model(cfg, -60*SECOND)
    direct.schedule(plant.admissions(prefix, cfg.hold) + plant.admissions(future, cfg.promote))
    direct.advance(120*SECOND)
    cohort = [(i, w) for i, write in direct.writes.items()
              if (w := score_window(write.arrival, write.eligible, cfg)) is not None]
    expected = horizon_records(direct, cohort)
    for passive in (True, False):
        model, _, _, cut = plant.replay("reference", "fixture", 5, "promote", passive=passive)
        assert model.db.visits == direct.db.visits and model.writes == direct.writes
        assert model.physical_digest(True) == direct.physical_digest(True)
        assert plant.endpoint(model)["windows"] == [
            {"eligible": w["eligible"], "misses": w["misses"]} for w in expected]
        assert bool(cut["history"]) is passive


def test_paired_root_surprise_and_action_order_share_only_permitted_draws():
    ordinary = plant.simulate_root("reference", "fixture", 3)
    reverse = plant.simulate_root("reference", "fixture", 3,
                                  order=list(reversed(ordinary["branches"])))
    assert ordinary == reverse
    branches = ordinary["branches"]
    assert branches["hold"]["future_digest"] == branches["promote"]["future_digest"]
    assert branches["surprise_hold"]["future_digest"] == branches["surprise_promote"]["future_digest"]
    assert branches["hold"]["future_digest"] != branches["surprise_hold"]["future_digest"]
    assert len({v["cut_digest"] for v in branches.values()}) == 1
    assert len(set(ordinary["history"][0])) == 7
