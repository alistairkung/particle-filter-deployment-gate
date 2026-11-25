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


def test_systematic_resampling_upper_roundoff_stays_within_positive_support():
    from particle_filter.inference import systematic_ancestors

    # The last grid point rounds to 1 in float64 before the open-end correction.
    assert list(systematic_ancestors([0.5, 0.5, 0, 0], np.nextafter(1.0, 0.0))) == [
        0,
        0,
        1,
        1,
    ]


def test_zero_weight_regime_is_extinct_before_resampling_threshold():
    from particle_filter.inference import infer
    from simulator.observations import START, INTERVAL, Reading

    class ExcludeSlowPublisher:
        def log_likelihoods(self, contexts, reading):
            return np.array([-np.inf if c.regime == "C" else 0.0 for c in contexts])

    reading = Reading(START, START + INTERVAL, START + INTERVAL, 0, 0, 98 / 5, 0.01)
    post = infer(
        [reading],
        None,
        ExcludeSlowPublisher(),
        particles=96,
        seed=731,
        cutoff=START + INTERVAL,
    )
    step = post.steps[0]
    assert not step.resampled and step.ess >= 48
    assert step.regime_counts[2] > 0 and step.regime_probabilities[2] == 0
    assert step.unique_ancestors == 96 - step.regime_counts[2]
