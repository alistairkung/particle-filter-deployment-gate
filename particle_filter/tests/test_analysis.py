import pytest

from particle_filter.analysis import score_roots


def test_matched_hold_only_breaches_contribute_to_paired_gain():
    rows = []
    for condition in "ABC":
        for index, endpoints in enumerate(((1, 0), (0, 1))):
            rows.append(
                dict(
                    key=f"{condition}{index}",
                    regime=condition,
                    hold=endpoints[0],
                    promote=endpoints[1],
                    pending=0,
                    work_s=0.25,
                    predictions={
                        name: {"hold": 0.5, "promote": 0.5}
                        for name in ("history", "prior", "current", "age", "ablated")
                    },
                )
            )
            rows[-1]["predictions"]["full"] = dict(
                hold=endpoints[0], promote=endpoints[1]
            )
    report = score_roots(rows, backlog_median=0, seed=13)
    assert report["primary"]["gain"] == 1
    assert report["ablation"]["gain"] == 1
    assert report["primary"]["interpretation"] == "degenerate"
    assert report["populations"]["A"]["models"]["full"]["hold_brier"] == 0
    assert report["populations"]["all"]["n"] == 6
    with pytest.raises(ValueError, match="duplicate"):
        score_roots(rows + rows[:1], backlog_median=0, seed=13)


def test_full_only_run_keeps_primary_without_requesting_the_ablation():
    rows = [
        dict(
            key=f"{z}{i}",
            regime=z,
            pending=1,
            backlog_median=i,
            hold=i,
            promote=1 - i,
            predictions={
                name: dict(hold=0.25, promote=0.75)
                for name in ("full", "history", "half-life-60")
            },
        )
        for z in "ABC"
        for i in range(2)
    ]
    report = score_roots(rows, seed=1)
    assert "primary" in report and "ablation" not in report
    calibration = report["populations"]["all"]["models"]["half-life-60"]["calibration"]
    assert sum(b["n"] for b in calibration["low"]["hold"]) == 3
