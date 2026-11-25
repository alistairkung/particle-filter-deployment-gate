from types import SimpleNamespace

import pytest

from telemetry_sufficiency import audit

S = 1_000_000_000


def write(arrival, commit, mark=None, eligible=True):
    return SimpleNamespace(arrival=arrival, commit=commit, mark=mark, eligible=eligible)


def test_history_reconstruction_uses_completed_effects_and_deliveries():
    writes = {
        1: write(-7 * S, -6 * S, -5 * S),
        2: write(-6 * S, -5 * S, 0),
        3: write(-6 * S, -3 * S, 0),
        4: write(-3 * S, 0, 3 * S),
        5: write(-4 * S, 2 * S),
        6: write(-6 * S, -4 * S, eligible=False),
    }
    delayed = {-10 * S: None, -5 * S: S, 0: 0}
    assert audit.history_from_writes(writes, delayed) == [
        {
            "interval_start_ns": -5 * S,
            "interval_end_ns": 0,
            "available_at_ns": 0,
            "pending_count": 1,
            "oldest_pending_age_s": 0.0,
            "publication_throughput_per_s": 0.4,
            "completed_delay_median_s": 4.0,
        }
    ]


def test_endpoint_reconstruction_preserves_open_left_deadline_bins_and_strict_five_percent():
    writes = {i: write(S, S) for i in range(19)}
    writes[19] = write(30 * S - S // 4, 30 * S + 1)
    writes[20] = write(30 * S, 30 * S + S // 4)
    writes[21] = write(120 * S - S // 4, None)
    writes[22] = write(120 * S - S // 4 + 1, None)
    writes[23] = write(0, None)
    assert audit.windows_from_writes(writes) == ([(20, 1), (1, 0), (0, 0), (1, 1)], 1)
    del writes[21]
    assert audit.windows_from_writes(writes)[1] == 0


def test_ineligible_arrival_is_excluded_and_cut_work_counts_only_enqueued_service():
    assert audit.windows_from_writes({1: write(S, None, eligible=False)})[0][0] == (0, 0)
    visits = [
        (1, 1, -30 * S, -30 * S, -29 * S),
        (1, 2, -S, -S, 2 * S),
        (0, 3, 0, 2 * S, 3 * S),
        (1, 4, 0, 3 * S, 5 * S),
        (3, 5, 1, 5 * S, 6 * S),
    ]
    assert audit.truth_from_visits(visits) == dict(
        query_work_ns=5 * S,
        duration_ns=30 * S,
        queued_ns=3 * S,
        active_residual_ns=2 * S,
        demand=5 / 30,
        unfinished=5.0,
    )


def test_raw_admissions_reject_an_unswitched_route_with_plausible_work():
    c = SimpleNamespace(ruby_ms=10, java_ms=20, insert_ms=2)
    tape = [(1, "write", 3, 0.25, 1.0), (1, "extra", 5, 15)]
    w = write(1, None)
    w.java, w.demand = True, 22_000_000
    db = SimpleNamespace(visits=[(0, 3, 1, 1, 22_000_001), (1, 11, 1, 22_000_001, 22_000_016)])
    m = SimpleNamespace(c=c, db=db, writes={3: w})
    assert audit.check_admissions(m, [], tape, "promote")[1] == 2
    with pytest.raises(ValueError, match="admission"):
        audit.check_admissions(m, [], tape, "hold")


def test_raw_replay_detects_swapped_histories_changed_work_windows_and_hidden_fields():
    from copy import deepcopy
    from telemetry_sufficiency.plant import simulate_root

    record = simulate_root("publisher", "fixture", 2)
    assert audit.audit_root("publisher", "fixture", 2, record)["branches"] == 2
    for mutation in ("join", "query_work", "window", "hidden_field"):
        changed = deepcopy(record)
        if mutation == "join":
            sibling = simulate_root("publisher", "fixture", 3)
            changed["history"], sibling["history"] = sibling["history"], changed["history"]
        elif mutation == "query_work":
            changed["truth"]["query_work_ns"] += 1
        elif mutation == "window":
            changed["branches"]["hold"]["windows"][0]["misses"] += 1
        else:
            changed["history"][0]["java_share"] = 0.25
        with pytest.raises(ValueError):
            audit.audit_root("publisher", "fixture", 2, changed)


def test_prediction_audit_rebuilds_history_slopes_with_missing_delay_and_late_arrival():
    from telemetry_sufficiency.audit_models import features

    history = [
        dict(
            interval_start_ns=-10 * S,
            interval_end_ns=-5 * S,
            available_at_ns=0,
            pending_count=7,
            oldest_pending_age_s=2.0,
            publication_throughput_per_s=0.0,
            completed_delay_median_s=None,
        ),
        dict(
            interval_start_ns=-5 * S,
            interval_end_ns=0,
            available_at_ns=0,
            pending_count=2,
            oldest_pending_age_s=3.0,
            publication_throughput_per_s=0.2,
            completed_delay_median_s=0.5,
        ),
    ]
    row = features("history", {"history": history})
    assert row["slope_pending_count"] == -1.0
    assert row["defined_completed_delay_median_s"] == 1
    assert row["undefined_slope_completed_delay_median_s"] == 1
    assert row["deliveries"] == 2


def test_reference_fixture_independently_replays_ordinary_and_surprise_pairs():
    from telemetry_sufficiency.plant import simulate_root

    record = simulate_root("reference", "fixture", 1)
    result = audit.audit_root("reference", "fixture", 1, record)
    assert result["branches"] == 4
    assert result["admissions"] > 0


def test_audit_claim_gain_uses_best_evaluation_constant_and_requires_uncertainty():
    from telemetry_sufficiency.audit_statistics import claim
    import numpy as np

    groups = {
        name: ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [4.0, 4.0, 4.0])
        for name in ("reference", "contention")
    }
    result = claim({k: tuple(map(np.asarray, v)) for k, v in groups.items()}, "age")
    assert result["constant"] == 1.0
    assert result["prior_gain"] == 9.0
    assert result["gain"] == result["se"] == 0.0
    assert result["passes"] is False


def test_complete_fixture_checks_every_fold_and_detects_prediction_corruption():
    from telemetry_sufficiency import methods, scoring

    dev, forecast = [], []
    for j, condition in enumerate(("reference", "contention", "publisher")):
        for i in range(5):
            history = [
                dict(
                    interval_start_ns=-5 * S,
                    interval_end_ns=0,
                    available_at_ns=0,
                    pending_count=i,
                    oldest_pending_age_s=i / 2.0,
                    publication_throughput_per_s=0.0,
                    completed_delay_median_s=None,
                )
            ]
            row = dict(
                key=f"dev-{j}-{i}",
                partition="development",
                condition=condition,
                demand=(i + j) / 10,
                unfinished=i / 10,
                hold=int(i % 2 == 0),
                promote=int(i % 3 != 0),
                history=history,
            )
            dev.append(row)
            scored = row | {"key": f"forecast-{j}-{i}"}
            del scored["partition"]
            if j == 0:
                scored.update(surprise_hold=1, surprise_promote=0)
            forecast.append(scored)
    fits = methods.fit_development(dev, "fixture-folds")
    for row in forecast:
        row["predictions"] = methods.predict(fits["models"], row["history"])
        row["predictions"]["evaluator"] = methods.predict_evaluator(
            fits["evaluator_models"], row["condition"], row["demand"], row["unfinished"]
        )
    scores = scoring.score(forecast, fits["selected"], 1234)
    receipt = audit.audit_analysis(dev, forecast, fits, scores, "fixture-folds", 1234)
    assert receipt["comparisons_checked"] == 77
    forecast[1]["predictions"]["age"]["demand"] += 0.005
    with pytest.raises(ValueError, match="prediction"):
        audit.audit_analysis(dev, forecast, fits, scores, "fixture-folds", 1234)
