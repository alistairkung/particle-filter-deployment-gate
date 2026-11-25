"""Acceptance fixtures for operational physics, matched inputs, and scored horizons."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from experiments.config import MS, SECOND, read_config
from experiments.observations import breaches, finish_latencies, horizon_records, latency_quantiles, score_window
from experiments.validation import publisher_release_check
from simulator.inputs import inputs
from simulator.model import CLAIM, MARK, Model


@pytest.fixture(scope="module")
def config():
    return read_config("configs/pilot.toml")


def test_arrival_eligibility_and_open_decision_boundary(config):
    assert [score_window(t, True, config) for t in (-1, 0, 1)] == [None, None, 0]
    assert score_window(1, False, config) is None
    model = Model(config, -MS)
    model.schedule([(-1, "write", -2, MS, False), (0, "write", -1, MS, False),
                    (1, "write", 0, MS, False)])
    model.advance(5 * MS)
    assert model.writes[0].commit == 3 * MS - 1


@pytest.mark.parametrize("boundary,ending", [(30, 0), (60, 1), (90, 2), (120, 3)])
def test_deadline_boundary_maps_to_ending_window_once(config, boundary, ending):
    arrival = boundary * SECOND - 250 * MS
    following = ending + 1 if ending < 3 else None
    assert [score_window(arrival + delta, True, config) for delta in (-1, 0, 1)] == [
        ending, ending, following
    ]


@pytest.mark.parametrize("duration,misses", [(250, 0), (251, 1)])
def test_horizon_commit_tie_is_on_time(config, duration, misses):
    model = Model(config, 0)
    model.schedule([(119_750 * MS, "write", 0, duration * MS, False)])
    model.advance(120 * SECOND)
    cohort = [(0, 3)]
    last_window = horizon_records(model, cohort)[-1]
    assert (last_window["eligible"], last_window["misses"]) == (1, misses)
    elapsed, _, censored = finish_latencies(model, cohort)
    assert elapsed.tolist() == [duration * MS]
    assert not censored.any()


def test_drain_preserves_queued_work_and_frozen_horizon(config):
    model = Model(config, 0, slow=True)
    # First event is acknowledged after 119.731 s. Its mark waits behind the
    # long query and the measured write; neither publisher phase is truncated.
    model.schedule([(119_679 * MS, "write", 0, MS, False),
                    (119_682 * MS, "reference", 7, 616 * MS, False),
                    (119_700 * MS, "write", 1, 102 * MS, False)])
    model.advance(120 * SECOND)
    cohort = [(0, 3), (1, 3)]
    before = horizon_records(model, cohort)
    frozen = deepcopy(before)
    assert model.phase == "mark"
    assert (before[-1]["eligible"], before[-1]["misses"], before[-1]["pending"]) == (2, 1, 1)
    assert before[-1]["db"]["query"]["busy_s"] == pytest.approx(.318)
    assert before[-1]["queue"] == 2
    assert before[-1]["work_s"] == pytest.approx(.401)
    elapsed, _, censored = finish_latencies(model, cohort)
    assert elapsed.tolist() == [MS, 700 * MS]
    assert not censored.any()
    assert model.env.now == 120_400 * MS
    assert before == frozen and list(model.writes) == [0, 1]


def test_drain_cap_censors_outstanding_and_counts_completion_at_cap(config):
    model = Model(config, 0)
    model.schedule([(MS, "write", 11, 240 * SECOND - MS, False),
                    (2 * MS, "write", 12, MS, True)])
    model.advance(120 * SECOND)
    cohort = [(11, 0), (12, 0)]
    windows = horizon_records(model, cohort)
    assert (windows[0]["eligible"], windows[0]["misses"]) == (2, 2)
    elapsed, java, censored = finish_latencies(model, cohort)
    assert model.env.now == 240 * SECOND
    assert java.tolist() == [False, True] and censored.tolist() == [False, True]
    assert elapsed.tolist() == [240 * SECOND - MS, 240 * SECOND - 2 * MS]
    assert latency_quantiles(elapsed, censored, [.99]) == [{"lower": float(elapsed[0])}]


def test_strict_window_threshold(config):
    assert not breaches(0, 0, config.miss_fraction)
    assert not breaches(1, 20, config.miss_fraction)
    assert breaches(2, 20, config.miss_fraction)


def test_acknowledgement_runs_off_db_then_mark_visits_db(config):
    assert publisher_release_check(config)["passed"]


def test_ack_already_in_flight_keeps_duration_across_fork(config):
    completions = []
    for slow in (False, True):
        model = Model(config, -3 * MS, slow=slow)
        model.schedule([(-3 * MS, "write", -1, MS, False)])
        model.advance(0)
        assert model.phase == "ack" and model.ack_due == MS
        model.schedule([(3 * MS, "write", 1, MS, False)])
        model.advance(100 * MS)
        completions.append((model.writes[-1].ack, model.writes[1].ack))
    assert completions == [(MS, 7 * MS), (MS, 55 * MS)]


def test_outbox_conservation_across_cut_with_all_publisher_phases(config):
    model = Model(config, 0)
    model.schedule([(MS, "write", 1, MS, False), (2 * MS, "write", 2, MS, True)])
    for tick, phase, pending in [(2, "claim", [1]), (4, "ack", [1, 2]),
                                 (6, "mark", [1, 2]), (7, "claim", [2]),
                                 (11, "idle", [])]:
        model.advance(tick * MS)
        assert (model.phase, list(model.pending)) == (phase, pending)
        completed = [w.commit for w in model.writes.values() if w.commit is not None]
        marked = [w.mark for w in model.writes.values() if w.mark is not None]
        assert len(completed) - len(marked) == len(pending)
    assert [visit[1] for visit in model.db.visits if visit[0] == CLAIM] == [1, 2]
    assert [visit[1] for visit in model.db.visits if visit[0] == MARK] == [1, 2]


def test_empty_and_inflight_age_at_window_end(config):
    # The event is acknowledged before the horizon but its mark is still queued.
    model = Model(config, 0)
    model.schedule([(119_995 * MS, "write", 0, MS, False),
                    (119_998 * MS, "reference", 1, 100 * MS, False)])
    model.advance(120 * SECOND)
    rows = horizon_records(model, [])
    assert (rows[0]["empty"], rows[0]["age_s"], rows[0]["delay_p50_s"]) == (True, 0, None)
    assert (rows[-1]["empty"], rows[-1]["age_s"], rows[-1]["pending"]) == (False, .004, 1)
    assert model.phase == "mark" and rows[-1]["marks"] == 0


def test_isolating_extra_stream_preserves_reference_queries_and_shared_draws(config):
    c = replace(config, seed=1001)
    hold = inputs(c, "C23", 0, "future", c.hold)
    high = inputs(c, "C23", 0, "future", c.hold, c.extra)
    promoted = inputs(c, "C23", 0, "future", c.promote, c.extra)
    assert [row for row in high if row[1] != "extra"] == hold
    assert [row for row in high if row[1] != "write"] == [row for row in promoted if row[1] != "write"]
    assert high != inputs(c, "C23", 1, "future", c.hold, c.extra)
    changed = 0
    for original, promoted_row in zip(high, promoted, strict=True):
        assert original[:3] == promoted_row[:3]
        if original[4] == promoted_row[4]:
            assert original[3] == promoted_row[3]
        else:
            changed += 1
            assert promoted_row[4]
            ruby_business = original[3] - c.insert_ms * MS
            java_business = promoted_row[3] - c.insert_ms * MS
            assert abs(java_business - ruby_business * c.java_ms / c.ruby_ms) <= 2
    assert changed > 0


def test_load_equations_include_relay_and_displaced_ruby_work(config):
    assert config.load_fraction(.25, 3.5) == pytest.approx(.740)
    assert config.load_fraction(.60, 3.5) == pytest.approx(.796)
    assert config.load_fraction(.25, 5.5) == pytest.approx(.820)
    assert config.load_fraction(.60, 5.5) == pytest.approx(.876)
    assert config.load_fraction(0, 12.5) == pytest.approx(1.06)
    assert np.isclose(config.load_fraction(.60, 3.5, True), config.load_fraction(.25, 3.5, True))
