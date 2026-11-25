import pytest

from telemetry_sufficiency import scoring


def protected_rows():
    rows = []
    for c in ("reference", "contention", "publisher"):
        for i, (h, p) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            history = [dict(interval_start_ns=-5*10**9, interval_end_ns=0, available_at_ns=0,
                            pending_count=i, oldest_pending_age_s=float(i),
                            publication_throughput_per_s=1., completed_delay_median_s=0.2)]
            public = dict(hold=.5, promote=.5, increment=0, condition=[1/3]*3,
                          demand=3., unfinished=0.)
            predictions = {level: public.copy() for level in ("prior", "age", "current", "history")}
            predictions["current"].update(hold=float(h), promote=float(p), increment=p-h, demand=float(i))
            predictions["evaluator"] = dict(hold=.5, promote=.5, increment=0)
            row = dict(key=f"{c}-{i}", condition=c, hold=h, promote=p, demand=float(i),
                       unfinished=0., predictions=predictions, history=history)
            if c == "reference":
                row.update(surprise_hold=h, surprise_promote=1)
            rows.append(row)
    return rows


def test_scores_paired_sign_and_stronger_constant_comparator():
    result = scoring.score(protected_rows(), {"demand": "current", "increment": "current"}, 100)
    all_roots = result["populations"]["all"]
    assert all_roots["paired"] == dict(neither=3, promote_only=3, hold_only=3, both=3)
    assert all_roots["rates"] == dict(hold=.5, promote=.5, increment=0.)
    assert all_roots["models"]["prior"]["scores"]["increment"] == .5
    physical = result["claims"]["demand"]
    assert physical["n"] == 8
    assert physical["constant"] == 1.5
    assert physical["gain"] == 1.25
    assert physical["prior_gain"] == 3.5
    assert physical["se"] == pytest.approx((1/6)**.5)
    assert physical["lower"] == pytest.approx(.4498481, abs=1e-5)
    assert physical["passes"] is True
    assert result["claims"]["increment"]["gain"] == .5
    assert result["surprise_changes"]["rates"]["increment"]["change"] == .5


def test_diagnostics_preserve_root_pairing_and_refit_each_bootstrap_constant():
    rows = protected_rows()
    report = scoring.score(rows, {"demand":"age", "increment":"age"}, 100)
    reference = report["populations"]["reference"]["models"]
    assert reference["prior"]["confusion"] == [[4,0,0],[0,0,0],[0,0,0]]
    calibration = reference["current"]["calibration"]["promote"]
    assert calibration[0] == dict(count=2, mean_prediction=0., event_rate=0.)
    assert calibration[5] == dict(count=0, mean_prediction=None, event_rate=None)
    assert calibration[9] == dict(count=2, mean_prediction=1., event_rate=1.)
    assert all(row["overlap"] == 1 for row in report["overlap"])
    gains = {(d["population"], d["baseline"], d["candidate"], d["target"]): d
             for d in report["comparisons"]}
    gain = gains["reference", "stratum_mean", "current", "demand"]
    assert gain["gain"] == 1.25
    # The bootstrap variance of resampled 0,1,2,3 has small values below .25.
    # Fixed scoring around the original mean would have a lower bound of at least .25.
    assert gain["lower"] == .1875
    same_forecasts = gains["all", "prior", "history", "increment"]
    assert same_forecasts["lower"] == same_forecasts["upper"] == 0
    assert report["surprise_changes"]["scores"]["current"]["promote"]["change"] == .5
    assert report["claims"]["increment"]["se"] == 0
    assert report["claims"]["increment"]["passes"] is False


def test_constant_gains_within_strata_have_zero_se_even_with_decimal_roundoff():
    rows = []
    for original in protected_rows():
        for copy in range(3):
            demand = .2 if original["condition"] == "reference" else .7
            forecasts = {level: dict(values) for level, values in original["predictions"].items()}
            forecasts["current"]["demand"] = demand
            rows.append(original | dict(key=f"{original['key']}-{copy}", demand=demand, predictions=forecasts))
    claim = scoring.score(rows, {"demand": "current", "increment": "current"}, 1)["claims"]["demand"]
    assert claim["gain"] > 0
    assert claim["se"] == 0
    assert claim["passes"] is False


def test_improved_absolute_action_scores_can_leave_increment_unchanged():
    rows = protected_rows()
    for row in rows:
        row["promote"] = row["hold"]
        row["predictions"]["current"].update(hold=row["hold"], promote=row["hold"], increment=0)
    result = scoring.score(rows, {"demand": "current", "increment": "current"}, 1)
    gains = {d["target"]: d["gain"] for d in result["comparisons"]
             if (d["population"], d["baseline"], d["candidate"]) == ("all", "prior", "current")}
    assert gains == dict(hold=.25, promote=.25, increment=0)
    assert result["claims"]["increment"]["passes"] is False
