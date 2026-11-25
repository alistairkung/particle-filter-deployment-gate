import numpy as np

from particle_filter.validation.phase_challenge import interval_summary


def test_phase_comparison_preserves_paired_root_weight_and_has_zero_gain_for_identical_forecasts():
    rows = [
        dict(
            condition="reference",
            root=0,
            regime="A",
            phase="idle",
            backlog="low",
            target=(10, 0.10),
            ordinary=(8, 0.09),
            revealed=(8, 0.09),
        ),
        dict(
            condition="reference",
            root=1,
            regime="A",
            phase="ack",
            backlog="low",
            target=(1, 0.20),
            ordinary=(5, 0.18),
            revealed=(5, 0.18),
        ),
    ]
    scores = interval_summary(rows, np.random.default_rng(43))
    pooled = scores["all"]
    assert pooled["roots"] == pooled["intervals"] == 2
    assert pooled["marks"]["ordinary_mae"] == 3
    assert pooled["marks"]["mae_gain_95"] == [0.0, 0.0]
    assert scores["phase=ack,backlog=low"]["work_s"]["mae_gain"] == 0
