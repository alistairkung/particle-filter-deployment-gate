import numpy as np
import pytest

from particle_filter.validation.budgets import particle_change, rollout_variation


def test_one_unstable_regime_prevents_particle_budget_lock():
    smaller = np.zeros((3, 12, 4, 4))
    larger = smaller.copy()
    larger[2, :, :, -1] = 0.011
    report = particle_change(smaller, larger)
    assert report["passes"] is False
    assert report["C"]["work_rms_s"] == pytest.approx(0.011)


def test_rollout_noise_is_estimated_within_history_and_across_seeds():
    risks = np.ones((3, 12, 4, 3))
    risks[:, 0] = 0
    fixed = rollout_variation(risks)
    assert fixed["passes"] and fixed["increment_variance"] == 0
    risks[1, :, 2:, 2] -= 0.2
    assert not rollout_variation(risks)["passes"]
