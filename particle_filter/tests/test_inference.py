import numpy as np
import pytest

from particle_filter.inference import transition_matrix


@pytest.mark.parametrize("half_life", [60, 120, 240])
def test_allowed_graph_stationarity_and_departure_half_life(half_life):
    matrix = transition_matrix(half_life)
    assert np.sum(matrix, axis=1) == pytest.approx(np.ones(3))
    assert matrix[0, 1] == matrix[0, 2] > 0
    assert matrix[1, 2] == matrix[2, 1] == 0
    assert np.array([0.5, 0.25, 0.25]) @ matrix == pytest.approx([0.5, 0.25, 0.25])
    assert np.diag(matrix) ** (half_life / 5) == pytest.approx([0.5] * 3)


def test_two_update_posterior_agrees_with_enumeration():
    from particle_filter.inference import normalise_weights

    # Three fixed trajectories with equal prior mass. Their joint data masses
    # after two intervals are proportional to 1*3, 2*3 and 3*1.
    first = normalise_weights(np.log([1, 2, 3]) + 10_000)
    final = normalise_weights(first + np.log([3, 3, 1]))
    assert np.exp(final) == pytest.approx([0.25, 0.50, 0.25])
    assert normalise_weights(final) == pytest.approx(final)
    with pytest.raises(ValueError, match="zero"):
        normalise_weights(np.array([-np.inf, -np.inf]))
    with pytest.raises(ValueError, match="nonfinite"):
        normalise_weights(np.array([1, np.nan]))


def test_systematic_resampling_skips_zero_mass_at_boundaries():
    from particle_filter.inference import systematic_ancestors

    assert list(systematic_ancestors([0, 0.5, 0.25, 0.25], 0)) == [1, 1, 2, 3]
    assert list(systematic_ancestors([0.8, 0.2, 0, 0], 0.5)) == [0, 0, 0, 1]


def test_inference_rejects_evaluator_fields_before_simulating():
    from particle_filter.inference import infer
    from simulator.observations import START, INTERVAL

    with pytest.raises(ValueError, match="public observation"):
        infer(
            [{"background_intensity": 3.5}],
            None,
            None,
            particles=4,
            seed=9,
            cutoff=START + INTERVAL,
        )


def test_missing_first_reading_preserves_prior_and_generates_no_evidence():
    from particle_filter.inference import infer
    from simulator.observations import START, INTERVAL

    posterior = infer([], None, None, particles=96, seed=731, cutoff=START + INTERVAL)
    (step,) = posterior.steps
    assert step.evidence_intervals == ()
    assert step.ess == pytest.approx(96)
    assert all(0 < probability < 1 for probability in posterior.regime_probabilities)
    assert sum(posterior.regime_probabilities) == pytest.approx(1)
    assert posterior.mean_work_s > 0
    assert posterior.cut_reading is None


def test_all_four_paired_cells_keep_a_negative_promotion_increment():
    from particle_filter.forecast import Forecast

    result = Forecast(hold_only=4, promote_only=1, both=3, neither=2, fallbacks=0)
    assert (result.hold, result.promote, result.delta) == pytest.approx(
        (0.7, 0.4, -0.3)
    )


def test_matched_forecast_reuses_draws_when_ablated_work_is_identical():
    from particle_filter.inference import Posterior
    from particle_filter.forecast import forecast
    from particle_filter.physical import Boundary, Closure

    posterior = Posterior(0, ("A",), (Boundary(),), (1.0,), None, ())
    full = forecast(posterior, Closure([]), pairs=5, seed=941)
    ablated = forecast(
        posterior,
        Closure([]),
        pairs=5,
        seed=941,
        work_by_regime={z: (0,) for z in ("A", "B", "C")},
    )
    assert full == ablated
    assert full.pairs == 5
    assert full.delta == pytest.approx(full.promote - full.hold)


def test_delayed_reading_updates_its_resampled_ancestor_once():
    from particle_filter.inference import infer
    from particle_filter.physical import Closure
    from simulator.observations import START, INTERVAL, Reading

    first, second, third = (START + n * INTERVAL for n in (1, 2, 3))
    late = Reading(START, first, third, 0, 0.0, 0.0, None)
    timely = Reading(first, second, second, 0, 0.0, 0.0, None)

    class FiniteLikelihood:
        def log_likelihoods(self, contexts, reading):
            work = np.array([c.work_end_s for c in contexts])
            if reading == timely:
                # This observation selects just one trajectory in the fixture.
                assert (work == work.max()).sum() == 1
                return np.where(work == work.max(), 0, -np.inf)
            # At the third boundary every descendant has the same ancestor.
            assert reading == late
            assert all(c.start_ns == START and c.prior is None for c in contexts)
            assert len(set(work)) == 1
            return -100 * work

    predictions = []
    model = Closure([])
    shared = dict(particles=32, seed=35, cutoff=third)
    observed = infer(
        [timely, late],
        model,
        FiniteLikelihood(),
        **shared,
        on_prediction=lambda end, contexts, *_: predictions.append(contexts),
    )
    missing = infer([timely], model, FiniteLikelihood(), **shared)
    assert observed.regimes == missing.regimes
    assert observed.boundaries == missing.boundaries
    assert observed.weights == pytest.approx(missing.weights)
    assert [step.evidence_intervals for step in observed.steps] == [
        (),
        (second,),
        (first,),
    ]
    assert observed.steps[1].unique_ancestors == 1
    assert observed.steps[2].ess == pytest.approx(32)
    assert all(c.prior is None for c in predictions[1])
    assert all(c.prior == timely for c in predictions[2])


def test_common_public_history_gives_identical_posteriors_and_action_forecasts():
    from dataclasses import replace
    from particle_filter.inference import infer
    from particle_filter.forecast import forecast
    from particle_filter.physical import Closure
    from simulator.observations import START, INTERVAL, Reading

    first = Reading(START, START + INTERVAL, START + INTERVAL, 0, 0.0, 2.0, 0.08)
    unseen = Reading(-INTERVAL, 0, INTERVAL, 1, 0.3, 2.0, 0.15)

    class KnownRatio:
        def log_likelihoods(self, contexts, reading):
            assert reading == first
            return np.log([2.0 if c.regime == "B" else 1.0 for c in contexts])

    closure = Closure([])
    forecasts = []
    posteriors = []
    for alternative in (
        unseen,
        replace(unseen, pending_count=497, oldest_pending_age_s=40),
    ):
        posterior = infer(
            [first, alternative], closure, KnownRatio(), particles=8, seed=31
        )
        posteriors.append(posterior)
        forecasts.append(forecast(posterior, closure, pairs=3, seed=33))
    assert posteriors[0] == posteriors[1]
    assert forecasts[0] == forecasts[1]


def test_fitted_forecast_runs_with_private_plant_and_evidence_access_denied(
    monkeypatch,
):
    """Fit on separate development roots; deny the forecast all evaluator adapters."""
    import builtins
    from pathlib import Path
    from particle_filter.validation.development import training_samples
    from particle_filter.validation.development_replay import replay_development
    from particle_filter.emissions import fit_emission
    from particle_filter.forecast import forecast
    from particle_filter.inference import infer
    from particle_filter.physical import Closure
    from telemetry_sufficiency import CONDITIONS, custody, plant

    histories = [replay_development(z, i) for z in CONDITIONS for i in range(3)]
    model = fit_emission(list(training_samples(histories)))
    closure = Closure((t.regime, t.end) for root in histories for t in root)
    observed = tuple(t.reading for t in replay_development("reference", 15))

    def denied(*args, **kwargs):
        raise AssertionError("Forecaster crossed the evaluator boundary")

    with monkeypatch.context() as guard:
        guard.setattr(plant, "raw_tape", denied)
        guard.setattr(plant, "replay", denied)
        guard.setattr(plant, "simulate_root", denied)
        guard.setattr(custody, "read", denied)
        guard.setattr(builtins, "open", denied)
        guard.setattr(Path, "open", denied)
        posterior = infer(observed, closure, model, particles=12, seed=7654)
        result = forecast(posterior, closure, pairs=2, seed=567)
    assert result.pairs == 2
    assert result.delta == result.promote - result.hold


def test_systematic_resampling_upper_roundoff_stays_within_positive_support():
    from particle_filter.inference import systematic_ancestors

    # The last grid point rounds to 1 in float64 before the open-end correction.
    assert list(systematic_ancestors([0.5, 0.5, 0, 0], np.nextafter(1.0, 0.0))) == [
        0,
        0,
        1,
        1,
    ]
