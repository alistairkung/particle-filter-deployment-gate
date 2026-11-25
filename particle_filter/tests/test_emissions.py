from dataclasses import replace
import math

import numpy as np
import pytest

from simulator.observations import INTERVAL, START, Reading
from particle_filter.emissions import Context, fit_emission


def reading(start, pending, age, marks, delay):
    return Reading(
        start, start + INTERVAL, start + INTERVAL, pending, age, marks / 5, delay
    )


def training():
    rng = np.random.default_rng(412)
    rows = []
    for regime in "ABC":
        for _ in range(80):
            n = int(rng.poisson(1.5))
            m = int(rng.poisson(18 if regime == "C" else 40))
            context = Context(
                regime, rng.exponential(0.1), rng.exponential(0.1), None, START
            )
            age = float(rng.uniform(0.001, 1.0)) if n else 0.0
            rows.append(
                (
                    context,
                    reading(
                        START,
                        n,
                        age,
                        m,
                        float(rng.uniform(0.052 if regime == "C" else 0.004, 0.6)),
                    ),
                )
            )
    return rows


def test_fitted_likelihood_preserves_zero_mark_support_and_exact_inherited_age():
    model = fit_emission(training())
    prior = reading(START, 2, 0.25, 8, 0.031)
    context = Context("C", 0.1, 0.15, prior, prior.interval_end_ns)
    observation = reading(context.start_ns, 2, 5.25, 0, None)
    assert math.isfinite(model.log_likelihood(context, observation))
    assert (
        model.log_likelihood(context, replace(observation, oldest_pending_age_s=5.24))
        == -math.inf
    )
    assert (
        model.log_likelihood(context, replace(observation, pending_count=1))
        == -math.inf
    )


def test_joint_predictive_diagnostics_respect_capacity_and_condition_timing():
    from particle_filter.validation.emission_diagnostics import (
        diagnostic_rows,
        training_thresholds,
    )

    rows = training()
    model = fit_emission(rows)
    context, observation = rows[-1]
    thresholds = training_thresholds(rows)
    assert set(thresholds) == {"A", "B", "C"}
    out = diagnostic_rows(
        model, [context], observation, [1.0], (97, 0), np.random.default_rng(42)
    )
    assert out["mark_tail"] == (0, 0.0)
    assert out["upper_joint_tail"] == (0, 0.0)
    assert out["backlog_tail"][1] + out["empty"][1] == pytest.approx(1.0)
    assert 0 <= out["mark_pit"][0] <= 1
    assert 0 <= out["pending_pit"][0] <= 1
    delay_study = [v for k, v in out.items() if k.startswith("delay_band_")]
    assert sum(x[0] for x in delay_study) == 1
    assert sum(x[1] for x in delay_study) == pytest.approx(1.0)


def test_root_intervals_preserve_temporal_pairing_and_flag_repeated_fold_errors():
    from particle_filter.validation.emission_diagnostics import (
        fold_diagnostics,
        repeatable_discrepancies,
    )

    records = []
    for root in ("first", "second", "third"):
        for start in (START, START + INTERVAL):
            rows = {"empty": (1.0, 0.1), "mark_pit": (0.9, 0.5)}
            records.append((root, "A", start, rows))
    report = fold_diagnostics(records, seed=32, resamples=500)
    assert report["A"]["empty"]["roots"] == 3
    assert report["A"]["empty"]["observations"] == 6
    assert report["A"]["empty"]["error_ci95"] == pytest.approx([0.9, 0.9])
    serial = report["A"]["mark_pit_lag1"]
    assert serial["observations"] == 3
    assert serial["error_ci95"] == pytest.approx([0.16, 0.16])
    assert not repeatable_discrepancies([report])
    assert {
        "regime": "A",
        "quantity": "empty",
        "direction": "underprediction",
        "folds": [0, 1],
    } in repeatable_discrepancies([report, report])


def test_completed_delay_obeys_minimum_publisher_cycle_and_never_has_zero_band_mass():
    from particle_filter.validation.emission_diagnostics import diagnostic_rows

    samples = training()
    model = fit_emission(samples)
    context, obs = samples[-1]
    assert (
        model.log_likelihood(context, replace(obs, completed_delay_median_s=0.003))
        == -math.inf
    )
    rows = diagnostic_rows(
        model, [context], obs, [1.0], (97, 0), np.random.default_rng(81)
    )
    assert (
        sum(v[1] for k, v in rows.items() if k.startswith("delay_band_0_given_")) == 0
    )


def test_fifo_timing_respects_age_of_oldest_possible_row_and_observation_horizon():
    model = fit_emission(training())
    p = reading(START, 8, 1.0, 2, 0.01)
    context = Context("A", 0.1, 0.1, p, p.interval_end_ns)
    observed = reading(context.start_ns, 7, 5.1, 1, 1.0)
    assert math.isfinite(model.log_likelihood(context, observed))
    assert (
        model.log_likelihood(context, replace(observed, oldest_pending_age_s=6.1))
        == -math.inf
    )
    assert (
        model.log_likelihood(context, replace(observed, completed_delay_median_s=6.1))
        == -math.inf
    )


def test_mixture_update_uses_hypothetical_regime_capacity_without_counting_throughput_twice():
    model = fit_emission(training())
    nominal = Context("A", 0.02, 0.07, None, START)
    slow = replace(nominal, regime="C")
    observed = reading(START, 0, 0.0, 98, 0.08)
    batch = model.log_likelihoods([nominal, slow], observed)
    assert np.isfinite(batch[0])
    assert batch[1] == -np.inf
    assert model.log_likelihood(nominal, observed) == pytest.approx(batch[0])
    fractional = replace(observed, publication_throughput_per_s=19.65)
    assert model.log_likelihood(nominal, fractional) == -np.inf
    with pytest.raises(ValueError, match="unavailable"):
        model.log_likelihood(replace(nominal, prior=observed), observed)


@pytest.mark.parametrize(("marks", "limit"), [(3, 5.0), (4, 4.974)])
def test_delay_median_respects_one_inherited_mark_and_new_commits(marks, limit):
    # One inherited row (age 12s), all other marks must be for new commits.
    # Three delay upper bounds are 16.896, 4.948, 5s, with median 5s.
    # Four are 16.844, 4.896, 4.948, 5s, with median 4.974s.
    model = fit_emission(training())
    p = reading(START + 3 * INTERVAL, 1, 12.0, 7, 0.02)
    c = Context("C", 0.1, 0.1, p, p.interval_end_ns)
    o = reading(c.start_ns, 0, 0.0, marks, limit)
    assert math.isfinite(model.log_likelihood(c, o))
    assert (
        model.log_likelihood(c, replace(o, completed_delay_median_s=limit + 1.0e-8))
        == -math.inf
    )
    # A single inherited mark can have started its acknowledgement in nominal A.
    assert math.isfinite(model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 1, 0.004)))
