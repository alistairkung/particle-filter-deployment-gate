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
    observed = reading(context.start_ns, 7, 5.1, 1, 1.000000001)
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


@pytest.mark.parametrize(("marks", "limit"), [(3, 4.998999999), (4, 4.972499999)])
def test_delay_median_respects_one_inherited_mark_and_new_commits(marks, limit):
    # One inherited row (age 12s), all other marks must be for new commits.
    # A new commit is at least one tick after start. Each later commit
    # also consumes a 1 ms insert on the same FIFO server.
    model = fit_emission(training())
    p = reading(START + 3 * INTERVAL, 1, 12.0, 7, 0.02)
    c = Context("C", 0.1, 0.1, p, p.interval_end_ns)
    o = reading(c.start_ns, 0, 0.0, marks, limit)
    assert math.isfinite(model.log_likelihood(c, o))
    assert (
        model.log_likelihood(c, replace(o, completed_delay_median_s=limit + 1.0e-8))
        == -math.inf
    )
    # An old row already at age 12s is necessarily the first completed mark.
    assert model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 1, 0.004)) == -math.inf
    assert math.isfinite(
        model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 1, 12.000000001))
    )
    young = replace(c, prior=replace(p, oldest_pending_age_s=0.003))
    assert math.isfinite(
        model.log_likelihood(young, reading(c.start_ns, 0, 0.0, 1, 0.004))
    )


def test_all_inherited_marks_have_committed_before_interval_start():
    model = fit_emission(training())
    p = reading(START + 3 * INTERVAL, 5, 1.0, 20, 0.1)
    c = Context("C", 0.01, 0.02, p, p.interval_end_ns)
    assert model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 5, 0.103)) == -math.inf
    # First row is at least 1s old. Even in earliest FIFO completion order,
    # remaining marks require 52,104,156,208 ms past the first open-left tick.
    assert model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 5, 0.156)) == -math.inf
    assert math.isfinite(model.log_likelihood(c, reading(c.start_ns, 0, 0.0, 5, 0.158)))


@pytest.mark.parametrize(("z", "marks"), [("C", 97), ("A", 1250)])
def test_empty_start_excludes_the_mark_requiring_an_inherited_visit(z, marks):
    from particle_filter.validation.emission_diagnostics import diagnostic_rows

    model = fit_emission(training())
    p = reading(START, 0, 0.0, 8, 0.031)
    c = Context(z, 0.01, 0.02, p, p.interval_end_ns)
    o = reading(c.start_ns, 0, 0.0, marks, 0.2)
    assert model.log_likelihood(c, o) == -math.inf
    permitted = reading(c.start_ns, 0, 0.0, 10, 0.2)
    checks = diagnostic_rows(
        model, [c], permitted, [1.0], (marks - 1, 5), np.random.default_rng(85)
    )
    assert checks["mark_tail"] == (0.0, 0.0)
    c = replace(c, prior=replace(p, pending_count=1, oldest_pending_age_s=0.003))
    assert math.isfinite(model.log_likelihood(c, o))


def test_age_bands_respect_insert_spacing_for_surviving_and_new_rows():
    model = fit_emission(training())
    p = reading(START, 5, 0.5, 8, 0.04)
    c = Context("A", 0.1, 0.12, p, p.interval_end_ns)
    survivors = reading(c.start_ns, 4, 5.001, 2, 0.51)
    # Three inherited rows survive. Even the latest possible commit of the
    # oldest survivor preceded the interval start by 2 ms.
    assert model.log_likelihood(c, survivors) == -math.inf
    assert math.isfinite(
        model.log_likelihood(c, replace(survivors, oldest_pending_age_s=5.003))
    )
    all_new = reading(c.start_ns, 2, 0.0, 6, 0.6)
    assert model.log_likelihood(c, all_new) == -math.inf
    assert math.isfinite(
        model.log_likelihood(c, replace(all_new, oldest_pending_age_s=0.002))
    )
    assert (
        model.log_likelihood(c, replace(all_new, oldest_pending_age_s=4.9995))
        == -math.inf
    )
