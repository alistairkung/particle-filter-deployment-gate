import math

import pytest

from telemetry_sufficiency import methods


def reading(end, count, age, rate, delay):
    return {
        "interval_start_ns": (end - 5) * 10**9,
        "interval_end_ns": end * 10**9,
        "available_at_ns": end * 10**9,
        "pending_count": count,
        "oldest_pending_age_s": age,
        "publication_throughput_per_s": rate,
        "completed_delay_median_s": delay,
    }


def test_temporal_features_use_native_means_actual_times_and_measured_zero():
    history = [
        reading(-30, 50, 3, 2, 1),
        reading(-20, 2, 1, 1, 0.3),
        reading(-5, 8, 3, 0, None),
    ]
    result = methods.features(history, "history")
    assert result["deliveries"] == 2
    assert result["current_pending_count"] == math.log1p(8)
    assert result["mean_pending_count"] == math.log1p(5)
    assert result["slope_pending_count"] == pytest.approx(0.4)
    assert result["defined_completed_delay_median_s"] == 1
    assert result["mean_completed_delay_median_s"] == math.log1p(0.3)
    assert result["slope_completed_delay_median_s"] is None
    assert result["undefined_slope_completed_delay_median_s"] == 1
    assert result["current_completed_delay_median_s"] is None
    assert result["current_publication_throughput_per_s"] == 0
    assert result["undefined_current_publication_throughput_per_s"] == 0


def test_isolated_prediction_boundary_rejects_extra_fields_and_duplicate_intervals():
    sample = reading(0, 2, 1, 1, .3)
    for invalid in ([sample | {"demand": .9}], [sample, sample]):
        with pytest.raises(ValueError):
            methods.features(invalid, "age")
    none_delivered = methods.features([sample | {"available_at_ns": 1}], "history")
    assert none_delivered["deliveries"] == 0
    assert none_delivered["defined_pending_count"] == 0
    assert none_delivered["mean_pending_count"] is None
    assert none_delivered["slope_pending_count"] is None
    assert none_delivered["undefined_current_oldest_pending_age_s"] == 1


def development_rows():
    rows = []
    for group, condition in enumerate(methods.CONDITIONS):
        for i in range(5):
            rows.append(dict(key=f"opaque-{i}-{group}", condition=condition, partition="development",
                             history=[reading(0, i+1, i, 1, .25)], demand=2*i,
                             unfinished=0., hold=0, promote=int(i > 1)))
    return rows


def test_lock_preserves_fold_priors_and_fit_can_predict_from_a_history_alone():
    rows = development_rows()
    ledger = methods.fit_development(rows, "toy")
    public = methods.predict(ledger["models"], rows[-1]["history"])
    assert public["prior"]["hold"] == 0
    assert public["prior"]["promote"] == 0.6
    assert public["prior"]["demand"] == 4.0
    assert public["age"]["demand"] > 6
    assert public["current"]["increment"] > .8
    assert public["history"]["condition"] == pytest.approx([1/3]*3)
    assert public["history"]["unfinished"] == 0
    assert set(public) == {"prior", "age", "current", "history"}
    for row in ledger["oof"]:
        peers = [r for r in rows if ledger["folds"][r["key"]] != row["fold"]]
        assert row["predictions"]["prior"]["demand"] == sum(r["demand"] for r in peers)/len(peers)
    assert set(ledger["selected"]) == {"demand", "increment"}


def test_protected_partition_is_rejected_before_feature_extraction_or_fitting():
    rows = development_rows()
    rows[0]["history"] = [{"hidden_field": 1}]
    rows[-1]["partition"] = "protected"
    with pytest.raises(ValueError, match="development-only"):
        methods.fit_development(rows, "toy")


def test_identical_information_breaks_ties_in_favour_of_age_and_excludes_private_inputs():
    rows = [r | {"history": []} for r in development_rows()]
    lock = methods.fit_development(rows, "toy")
    assert lock["selected"] == {"demand": "age", "increment": "age"}
    row = rows[0]
    before = methods.predict(lock["models"], row["history"])
    row.update(demand=1e4, condition="publisher", promote=1)
    lock["evaluator_models"] = {}
    assert methods.predict(lock["models"], row["history"]) == before
    assert all(x == 1 for x in lock["models"]["history"]["scales"])


def test_constant_physical_target_is_preserved_exactly_in_each_fold():
    rows = [row | {"demand": 0.1} for row in development_rows()]
    lock = methods.fit_development(rows, "toy")
    assert all(entry["predictions"][level]["demand"] == 0.1
               for entry in lock["oof"] for level in methods.LEVELS)
