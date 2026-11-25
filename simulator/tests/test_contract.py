from dataclasses import FrozenInstanceError, asdict

import pytest

from simulator.config import MS, SECOND, PlantConfig
from simulator.observations import parse_history
from simulator.replay import replay_prefix, run_future
from simulator.workload import admissions, primitive_tape


CONFIG = PlantConfig(
    seed=57012026,
    warmup=60,
    horizon=120,
    writes=40,
    ruby_ms=11,
    java_ms=15,
    insert_ms=1,
    claim_ms=1,
    mark_ms=1,
    ack_ms=2,
    slow_ms=50,
    query_ms=40,
)


def test_reconstructed_branches_own_their_queue_and_inflight_ack():
    # The claim ends at -1 ms. A query and a waiting write span the cut.
    prefix = [
        (-3 * MS, "write", -2, 0.4, 0.0),
        (-MS // 2, "reference", -1, 10 * MS),
        (-MS // 4, "write", -1, 0.2, 0.0),
    ]
    future = [(MS, "write", 0, 0.4, 1.0)]
    hold, history = replay_prefix(CONFIG, iter(prefix), 0.25)
    promote, sibling_history = replay_prefix(CONFIG, prefix, 0.25)
    cut = promote.physical_digest(with_routes=True)
    assert hold.physical_digest(with_routes=True) == cut
    assert history == sibling_history and history[-1].pending_count == 1
    assert hold.ack_due == MS and hold.writes[-1].commit is None

    run_future(hold, iter(future), 0.25)
    assert promote.physical_digest(with_routes=True) == cut
    run_future(promote, future, 0.60)
    assert hold.writes[-2].ack == promote.writes[-2].ack == MS
    assert (hold.writes[0].demand, promote.writes[0].demand) == (12 * MS, 16 * MS)
    assert history[-1].pending_count == 1


def test_controller_receives_only_available_immutable_measurements():
    deliveries = {-10 * SECOND: None, -5 * SECOND: 1, 0: 0}
    _, history = replay_prefix(CONFIG, [], 0.25, deliveries=deliveries)
    assert len(history) == 10
    assert [r.interval_end_ns for r in history[-2:]] == [-15 * SECOND, 0]
    row = asdict(history[-1])
    assert row == dict(
        interval_start_ns=-5 * SECOND,
        interval_end_ns=0,
        available_at_ns=0,
        pending_count=0,
        oldest_pending_age_s=0.0,
        publication_throughput_per_s=0.0,
        completed_delay_median_s=None,
    )
    assert parse_history(history, cutoff=-1) == history[:-1]
    with pytest.raises(ValueError, match="fields"):
        parse_history([dict(row, ack_due=None)])
    with pytest.raises(FrozenInstanceError):
        history[-1].pending_count = 1


def test_tapes_reproduce_by_root_and_match_before_routing():
    key = ("v1-fixture", 0)

    def future(root, extra=0):
        return primitive_tape(
            CONFIG, root, "future", reference_rate=3.5, extra_rate=extra
        )

    tape = future(key)
    assert tape == future(iter(key))
    assert tape != future(("v1-fixture", 1))
    assert tape == [row for row in future(key, extra=2) if row[1] != "extra"]
    hold = admissions(tape, 0.25, CONFIG)
    promote = admissions(tape, 0.60, CONFIG)
    assert [row[:3] for row in hold] == [row[:3] for row in promote]
    assert all(not old[-1] or new[-1] for old, new in zip(hold, promote, strict=True))


def test_replay_rejects_wrong_phase_before_advancing():
    with pytest.raises(ValueError, match="pre-decision"):
        replay_prefix(CONFIG, [(1, "reference", 0, MS)], 0.25)
    cut, _ = replay_prefix(CONFIG, [], 0.25)
    with pytest.raises(ValueError, match="post-cut"):
        run_future(cut, [(0, "reference", 0, MS)], 0.60)
    assert cut.env.now == 0 and not cut.db.visits


@pytest.mark.parametrize("key", [("ref/study", 0), ("study", "0"), ("study", True)])
def test_ambiguous_names_cannot_alias_an_independent_root(key):
    with pytest.raises(ValueError, match="Root"):
        primitive_tape(CONFIG, key, "prefix", reference_rate=3.5)


def test_admission_at_start_of_warmup_is_preserved():
    plant, _ = replay_prefix(CONFIG, [(-60 * SECOND, "write", -1, 0.3, 1.0)], 0.25)
    assert plant.writes[-1].commit == -60 * SECOND + 12 * MS
