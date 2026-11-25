"""Frozen paired decision rules and whole-history descriptive intervals."""

import numpy as np
from scipy.stats import beta

from simulator.inputs import rng


def cp_lower(count, n, tail):
    return 0.0 if count == 0 else float(beta.ppf(tail, count, n - count + 1))


def cp_upper(count, n, tail):
    return 1.0 if count == n else float(beta.isf(tail, count + 1, n - count))


def paired_effect(hold, promote, component_tail):
    """Count all pairs. Each directional bound spends two component tails."""
    hold = np.asarray(hold)
    promote = np.asarray(promote)
    if hold.ndim != 1 or promote.shape != hold.shape or not len(hold):
        raise ValueError("Each root must supply one outcome for each intervention")
    if not np.isin(hold, [0, 1]).all() or not np.isin(promote, [0, 1]).all():
        raise ValueError("Breach outcomes must be binary")
    n = len(hold)
    hold_only = int(np.count_nonzero(hold > promote))
    promote_only = int(np.count_nonzero(promote > hold))
    both = int(np.count_nonzero((hold == 1) & (promote == 1)))
    return {
        "n": n,
        "neither": n - both - hold_only - promote_only,
        "hold_only": hold_only,
        "promote_only": promote_only,
        "both": both,
        "estimate": (promote_only - hold_only) / n,
        "lower": cp_lower(promote_only, n, component_tail)
        - cp_upper(hold_only, n, component_tail),
        "upper": cp_upper(promote_only, n, component_tail)
        - cp_lower(hold_only, n, component_tail),
    }


def bootstrap_mean(c, family, root_contrasts):
    """Each contrast combines all of one root's branches before resampling."""
    root_contrasts = np.asarray(root_contrasts)
    indices = rng(c.seed, "bootstrap", family).integers(
        len(root_contrasts), size=(c.bootstrap, len(root_contrasts))
    )
    lower, upper = np.quantile(root_contrasts[indices].mean(axis=1), [0.025, 0.975])
    return {
        "estimate": float(root_contrasts.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }


def migration_finding(comparisons, margin):
    if any(pair["lower"] >= margin for pair in comparisons.values()):
        return "consequential"
    if all(pair["upper"] < margin for pair in comparisons.values()):
        return "excluded"
    return "inconclusive"


def classify_verdict(finding, controls_pass):
    if controls_pass:
        if finding == "consequential":
            return "GO"
        if finding == "excluded":
            return "NO-GO"
    return "MODIFY / inconclusive"


def publisher_evidence(c, outcomes, age_differences):
    paired = paired_effect(outcomes["nominal"], outcomes["slow"], 0.025)
    paired["nominal_only"] = paired.pop("hold_only")
    paired["slow_only"] = paired.pop("promote_only")
    n = paired["n"]
    absolute = {}
    for name, series in outcomes.items():
        count = int(series.sum())
        absolute[name] = {
            "lower": cp_lower(count, n, 0.05),
            "upper": cp_upper(count, n, 0.05),
        }
    health = "inconclusive"
    if (
        all(risk["upper"] < c.health_limit for risk in absolute.values())
        and paired["upper"] < c.harm_limit
    ):
        health = "demonstrated"
    elif (
        any(risk["lower"] >= c.health_limit for risk in absolute.values())
        or paired["lower"] >= c.harm_limit
    ):
        health = "violated"

    if len(age_differences) != n or not np.isfinite(age_differences).all():
        raise ValueError("C5 requires both horizon ages in every pair")
    successes = int(np.count_nonzero(age_differences >= c.age_margin))
    lower = cp_lower(successes, n, 0.05)
    age = {
        "successes": successes,
        "fraction": successes / n,
        "lower": lower,
        "passed": lower > c.success_fraction,
    }
    return {
        "n": n,
        "paired": paired,
        "absolute": absolute,
        "health": health,
        "age": age,
        "passed": health == "demonstrated" and age["passed"],
    }


def _binary_outcomes(roots, family, arms, expected):
    records = roots.get(family, [])
    if len(records) != expected:
        raise ValueError(
            f"{family}: received {len(records)} of {expected} required roots"
        )
    if any(set(row) != set(arms) for row in records):
        raise ValueError(f"{family}: unexpected or missing branches")
    return {
        arm: np.asarray([row[arm]["breach"] for row in records], dtype=int)
        for arm in arms
    }


def _risk(series):
    n = len(series)
    count = int(series.sum())
    return {
        "n": n,
        "breaches": count,
        "risk": count / n,
        "lower": cp_lower(count, n, 0.025),
        "upper": cp_upper(count, n, 0.025),
    }


def summarize(c, roots, validation_passed):
    """Classify each question on its own evidence, then apply the joint protocol gate."""
    outcomes = {
        "C23": _binary_outcomes(
            roots,
            "C23",
            ("ref_hold", "ref_promote", "high_hold", "high_promote"),
            c.migration_roots,
        ),
        "C4": _binary_outcomes(
            roots, "C4", ("ruby", "hold", "promote"), c.overload_roots
        ),
        "C5": _binary_outcomes(roots, "C5", ("nominal", "slow"), c.publisher_roots),
    }
    risks = {
        family: {arm: _risk(series) for arm, series in group.items()}
        for family, group in outcomes.items()
    }

    migration, contrasts = {}, {}
    for setting in ("ref", "high"):
        hold = outcomes["C23"][f"{setting}_hold"]
        promote = outcomes["C23"][f"{setting}_promote"]
        migration[setting] = paired_effect(hold, promote, c.migration_tail)
        contrasts[setting] = promote - hold
    finding = migration_finding(migration, c.margin)
    interaction = bootstrap_mean(c, "C23", contrasts["high"] - contrasts["ref"])

    hold, promote = outcomes["C4"]["hold"], outcomes["C4"]["promote"]
    overload = paired_effect(hold, promote, 0.025)
    overload.update(bootstrap_mean(c, "C4", promote - hold))

    # Saved age is in seconds; recover integer ticks before testing exact margin equality.
    age_differences = [
        (
            round(row["slow"]["windows"][-1]["age_s"] * 1e9)
            - round(row["nominal"]["windows"][-1]["age_s"] * 1e9)
        )
        / 1e9
        for row in roots["C5"]
    ]
    publisher = publisher_evidence(c, outcomes["C5"], np.array(age_differences))
    verdict = classify_verdict(finding, validation_passed and publisher["passed"])
    reasons = [
        f"Migration evidence relative to the {100 * c.margin:g}-point margin: {finding}."
    ]
    if not validation_passed:
        reasons.append("One or more required validation or workload checks failed.")
    reasons.append(f"C5 write health: {publisher['health']}.")
    reasons.append(
        "C5 outbox age criterion "
        + ("passed." if publisher["age"]["passed"] else "did not pass.")
    )
    return {
        "validation_passed": validation_passed,
        "risks": risks,
        "migration": migration,
        "migration_finding": finding,
        "interaction": interaction,
        "overload": overload,
        "publisher": publisher,
        "verdict": verdict,
        "reasons": reasons,
    }
