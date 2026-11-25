import numpy as np
import pytest


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


def test_one_experiment_can_repeat_a_posterior_across_independent_forecast_seeds():
    from types import SimpleNamespace
    from particle_filter.config import load_variant
    from particle_filter.experiment import predict_variants
    from particle_filter.physical import Closure

    fit = SimpleNamespace(closure=Closure([]), likelihood=None)
    full = [load_variant("full")]
    settings = dict(particles=6, pairs=3, seed=131)
    default = predict_variants([], fit, full, **settings)["full"]
    same = predict_variants([], fit, full, **settings, forecast_seed=131)["full"]
    other = predict_variants([], fit, full, **settings, forecast_seed=95)["full"]
    assert same == default
    assert other["regime_probabilities"] == same["regime_probabilities"]
    assert other["mean_work_s"] == same["mean_work_s"]
