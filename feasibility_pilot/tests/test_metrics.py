import numpy as np
import pytest
from scipy.stats import binom

from experiments.metrics import paired_effect


def test_discordant_pairs_keep_agreements_in_denominator_and_invert_binomial_tails():
    hold = np.array([0, 0, 0, 1, 1])
    promote = np.array([1, 1, 0, 1, 0])
    effect = paired_effect(hold, promote, 0.0125)
    assert [effect[key] for key in ("neither", "hold_only", "promote_only", "both")] == [1, 1, 2, 1]
    assert effect["estimate"] == 0.2
    reverse = paired_effect(promote, hold, 0.0125)
    assert reverse["estimate"] == -effect["estimate"]
    assert reverse["upper"] == pytest.approx(-effect["lower"])
    # All 20 pairs go in the same direction. CP lower reduces to an n-th root.
    all_harm = paired_effect(np.zeros(20), np.ones(20), 0.0125)
    assert all_harm["upper"] == 1
    assert all_harm["lower"] == pytest.approx(2 * 0.0125**(1 / 20) - 1)


from experiments.config import read_config
from experiments.metrics import migration_finding, classify_verdict, publisher_evidence, bootstrap_mean


@pytest.mark.parametrize("lo,hi,finding", [
    ([0.10, -0.02], [0.30, 0.20], "consequential"),
    ([-0.10, 0.01], [0.09, 0.0999], "excluded"),
    ([-0.10, 0.01], [0.09, 0.10], "inconclusive"),
])
def test_migration_decision_uses_directional_bounds_and_checks_separately(lo, hi, finding):
    comparisons = {i: {"lower": low, "upper": high} for i, (low, high) in enumerate(zip(lo, hi))}
    assert migration_finding(comparisons, .10) == finding
    assert classify_verdict(finding, False) == "MODIFY / inconclusive"
    assert classify_verdict(finding, True) == {
        "consequential":"GO", "excluded":"NO-GO", "inconclusive":"MODIFY / inconclusive"
    }[finding]


@pytest.mark.parametrize("successes,passes", [(216,False), (217,True)])
def test_publisher_exact_cutoff_and_population_health(successes, passes):
    c = read_config("configs/pilot.toml")
    outcomes = {"nominal": np.zeros(400), "slow": np.zeros(400)}
    age_difference = np.zeros(400)
    age_difference[:successes] = 5.0
    evidence = publisher_evidence(c, outcomes, age_difference)
    assert evidence["age"]["successes"] == successes
    assert evidence["age"]["lower"] == pytest.approx(
        binom_exact_lower(successes, 400, 0.05)
    )
    assert evidence["health"] == "demonstrated"
    assert evidence["passed"] is passes
    assert evidence["paired"]["upper"] == pytest.approx(1-0.025**(1/400))


def binom_exact_lower(count, n, tail):
    from scipy.optimize import brentq
    return brentq(lambda p: binom.sf(count-1,n,p)-tail, 0, 1)


def test_material_write_harm_fails_c5_even_when_age_deteriorates():
    c = read_config("configs/pilot.toml")
    outcomes = {"nominal":np.zeros(400), "slow":np.ones(400)}
    evidence = publisher_evidence(c, outcomes, np.full(400, 5))
    assert evidence["age"]["passed"] and not evidence["passed"]
    assert evidence["health"] == "violated"
    assert evidence["paired"]["lower"] >= c.harm_limit


def test_resampling_preserves_degenerate_root_interaction():
    c = read_config("configs/pilot.toml")
    assert bootstrap_mean(c, "C23", np.zeros(400)) == {"estimate":0, "lower":0, "upper":0}



def test_component_findings_survive_failed_publisher_control():
    from experiments.metrics import summarize
    c = read_config("configs/pilot.toml")

    def branch(breach):
        return {"breach": int(breach), "windows": [{"age_s": 0.0}]}

    roots = {
        "C23": [{name: branch(name == "high_promote") for name in (
            "ref_hold", "ref_promote", "high_hold", "high_promote"
        )} for _ in range(400)],
        "C4": [{name: branch(1) for name in ("ruby", "hold", "promote")} for _ in range(400)],
        "C5": [{name: branch(0) for name in ("nominal", "slow")} for _ in range(400)],
    }
    result = summarize(c, roots, validation_passed=True)
    assert result["migration_finding"] == "consequential"
    assert result["interaction"] == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}
    assert result["overload"]["upper"] == 0.0
    assert result["risks"]["C4"]["ruby"]["risk"] == 1
    assert result["publisher"]["health"] == "demonstrated"
    assert not result["publisher"]["passed"]
    assert result["verdict"] == "MODIFY / inconclusive"
    roots["C23"].pop()
    with pytest.raises(ValueError, match="399"):
        summarize(c, roots, validation_passed=True)

