"""Reconstruct decisions directly from horizon window counts, without importing the simulator or metrics."""

import argparse
import hashlib
import json
from collections import Counter
from functools import cache
from math import isclose
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, sem, t

from experiments.config import read_config


def _read_evidence(directory, family, arm_names, n, c):
    outcomes = {arm: [] for arm in arm_names}
    ages = []
    for root_id in range(n):
        path = directory / "branches" / family / f"{root_id:04d}.json"
        record = json.loads(path.read_text())
        if (record["family"], record["root"]) != (family, root_id):
            raise ValueError(f"{path}: root identity mismatch")
        if set(record["branches"]) != set(arm_names):
            raise ValueError(f"{path}: invalid arm set")
        oldest = {}
        for arm, branch in record["branches"].items():
            scored = 0
            late_windows = 0
            windows = branch["windows"]
            if [w["window"] for w in windows] != list(
                range(1, 1 + c.horizon // c.window)
            ):
                raise ValueError(f"{path}: incomplete windows for {arm}")
            for w in windows:
                denominator = w["eligible"]
                misses = w["misses"]
                if not 0 <= misses <= denominator:
                    raise ValueError(f"{path}: invalid miss count for {arm}")
                late = denominator > 0 and misses / denominator > c.miss_fraction
                if late != w["breach"]:
                    raise ValueError(f"{path}: inconsistent window score for {arm}")
                scored += denominator
                late_windows += late
            outcome = int(late_windows > 0)
            if outcome != branch["breach"] or scored != branch["scored"]:
                raise ValueError(
                    f"{path}: inconsistent cohort or horizon score for {arm}"
                )
            outcomes[arm].append(outcome)
            if family == "C5":
                end = windows[-1]
                if end["empty"] != (end["pending"] == 0) or (
                    end["empty"] and end["age_s"] != 0
                ):
                    raise ValueError(
                        f"{path}: inconsistent empty-outbox coding for {arm}"
                    )
                oldest[arm] = round(end["age_s"] * 1_000_000_000)
        if family == "C5":
            ages.append(
                oldest["slow"] - oldest["nominal"] >= c.age_margin * 1_000_000_000
            )
    return outcomes, ages


@cache
def _invert_binomial(count, n, tail):
    """Direct inversion of the two discrete binomial tests, including edge cells."""
    lower = (
        0.0
        if count == 0
        else brentq(lambda p: binom.sf(count - 1, n, p) - tail, 0, 1, xtol=1e-15)
    )
    upper = (
        1.0
        if count == n
        else brentq(lambda p: binom.cdf(count, n, p) - tail, 0, 1, xtol=1e-15)
    )
    return lower, upper


def _pair_table(first, second, tail, first_label="hold", second_label="promote"):
    counts = Counter(zip(first, second, strict=True))
    n = len(first)
    lower_gain, upper_gain = _invert_binomial(counts[0, 1], n, tail)
    lower_loss, upper_loss = _invert_binomial(counts[1, 0], n, tail)
    return {
        "n": n,
        "neither": counts[0, 0],
        f"{first_label}_only": counts[1, 0],
        f"{second_label}_only": counts[0, 1],
        "both": counts[1, 1],
        "estimate": (sum(second) - sum(first)) / n,
        "lower": lower_gain - upper_loss,
        "upper": upper_gain - lower_loss,
    }


def _bootstrap_columns(c, family, columns, weights):
    """Select the same root rows for all columns, then combine the arm means."""
    raw = np.column_stack(columns)
    name = f"{c.seed}/bootstrap/{family}".encode()
    seed = int.from_bytes(hashlib.sha256(name).digest(), "big")
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = []
    for start in range(0, c.bootstrap, 100):
        count = min(100, c.bootstrap - start)
        sampled = generator.choice(len(raw), (count, len(raw)), replace=True)
        arm_means = raw[sampled].mean(axis=1)
        draws.extend(arm_means @ weights)
    quantiles = np.percentile(draws, [2.5, 97.5])
    return {
        "estimate": float(raw.mean(axis=0) @ weights),
        "lower": float(quantiles[0]),
        "upper": float(quantiles[1]),
    }


def _publisher(c, group, age_success):
    n = len(age_success)
    both_bounds = {
        arm: _invert_binomial(sum(values), n, 0.05) for arm, values in group.items()
    }
    pair = _pair_table(group["nominal"], group["slow"], 0.025, "nominal", "slow")
    health = "inconclusive"
    if (
        max(upper for _, upper in both_bounds.values()) < c.health_limit
        and pair["upper"] < c.harm_limit
    ):
        health = "demonstrated"
    if (
        max(lower for lower, _ in both_bounds.values()) >= c.health_limit
        or pair["lower"] >= c.harm_limit
    ):
        health = "violated"
    k = sum(age_success)
    lower = _invert_binomial(k, n, 0.05)[0]
    age = {
        "successes": k,
        "fraction": k / n,
        "lower": lower,
        "passed": lower > c.success_fraction,
    }
    return {
        "n": n,
        "paired": pair,
        "absolute": {
            arm: {"lower": lo, "upper": hi} for arm, (lo, hi) in both_bounds.items()
        },
        "age": age,
        "health": health,
        "passed": health == "demonstrated" and age["passed"],
    }


def _compare(path, reconstructed, saved):
    if isinstance(reconstructed, dict):
        for name, value in reconstructed.items():
            _compare(f"{path}.{name}", value, saved[name])
    elif isinstance(reconstructed, float):
        if not isclose(reconstructed, saved, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"{path}: independently reconstructed {reconstructed}, saved {saved}"
            )
    elif reconstructed != saved:
        raise ValueError(
            f"{path}: independently reconstructed {reconstructed}, saved {saved}"
        )


def audit(c, directory, summary, validation_passed):
    directory = Path(directory)
    if (directory / "validation" / "result.json").is_file():
        if audit_prerequisites(c, directory) != validation_passed:
            raise ValueError("Caller and persisted prerequisite evidence disagree")
    families = {
        "C23": (
            ["ref_hold", "ref_promote", "high_hold", "high_promote"],
            c.migration_roots,
        ),
        "C4": (["ruby", "hold", "promote"], c.overload_roots),
        "C5": (["nominal", "slow"], c.publisher_roots),
    }
    outcomes = {}
    risks = {}
    publisher_ages = []
    for family, (arm_names, n) in families.items():
        group, ages = _read_evidence(directory, family, arm_names, n, c)
        outcomes[family] = group
        risks[family] = {}
        for arm, values in group.items():
            count = sum(values)
            low, high = _invert_binomial(count, n, 0.025)
            risks[family][arm] = {
                "n": n,
                "breaches": count,
                "risk": count / n,
                "lower": low,
                "upper": high,
            }
        if family == "C5":
            publisher_ages = ages

    migration = {}
    for regime in ("ref", "high"):
        first, second = (
            outcomes["C23"][regime + "_" + arm] for arm in ("hold", "promote")
        )
        migration[regime] = _pair_table(first, second, c.migration_tail)
    interaction = _bootstrap_columns(
        c, "C23", [outcomes["C23"][arm] for arm in families["C23"][0]], [1, -1, -1, 1]
    )
    hold, promote = (outcomes["C4"][arm] for arm in ("hold", "promote"))
    overload = _pair_table(hold, promote, 0.025)
    overload.update(_bootstrap_columns(c, "C4", [hold, promote], [-1, 1]))
    publisher = _publisher(c, outcomes["C5"], publisher_ages)

    finding = "inconclusive"
    if max(row["lower"] for row in migration.values()) >= c.margin:
        finding = "consequential"
    elif max(row["upper"] for row in migration.values()) < c.margin:
        finding = "excluded"
    verdict = "MODIFY / inconclusive"
    if validation_passed and publisher["passed"]:
        if finding == "consequential":
            verdict = "GO"
        elif finding == "excluded":
            verdict = "NO-GO"
    rebuilt = {
        "risks": risks,
        "migration": migration,
        "migration_finding": finding,
        "interaction": interaction,
        "overload": overload,
        "publisher": publisher,
        "validation_passed": validation_passed,
        "verdict": verdict,
    }
    _compare("summary", rebuilt, summary)
    return {
        "verified": True,
        "method": "Window counts, independent binomial-tail inversion and paired column bootstrap",
        "reconstructed": rebuilt,
    }


def audit_prerequisites(c, directory):
    """Recompute the queue intervals and placebo comparisons from the saved validation evidence."""
    directory = Path(directory)
    validation = json.loads((directory / "validation" / "result.json").read_text())
    cells = []
    expected_cells = {
        (name, rate) for name in ("M/M/1", "M/D/1") for rate in c.queue_rates
    }
    observed_cells = {
        (cell["distribution"], cell["rate"]) for cell in validation["queue"]
    }
    if observed_cells != expected_cells or len(validation["queue"]) != len(
        expected_cells
    ):
        raise ValueError("Missing or duplicated analytical queue cell")
    for cell in validation["queue"]:
        means = cell["seed_means_ms"]
        if len(means) != c.queue_roots:
            raise ValueError("Incorrect analytical queue replication count")
        rho = cell["rate"] / c.queue_service_rate
        expected = 1000 / c.queue_service_rate * rho / (1 - rho)
        if cell["distribution"] == "M/D/1":
            expected *= 0.5
        low, high = t.interval(
            0.95, len(means) - 1, loc=np.mean(means), scale=sem(means)
        )
        passed = expected * (1 - c.queue_tolerance) <= low and high <= expected * (
            1 + c.queue_tolerance
        )
        cells.append(bool(passed))
        _compare("queue", {"passed": bool(passed), "expected_ms": expected}, cell)
        if not np.allclose([low, high], cell["interval_ms"], rtol=1e-11):
            raise ValueError(
                "Saved queue confidence interval differs from seed reconstruction"
            )

    placebos = validation["placebo"]
    if sorted(row["root_id"] for row in placebos) != list(range(c.placebo_roots)):
        raise ValueError("Incorrect placebo root list")
    for row in placebos:
        arms = row["arms"]
        matched = set(arms) == {"zero", "hold", "promote", "all"} and all(
            branch["trace"] == arms["hold"]["trace"]
            and branch["windows"] == arms["hold"]["windows"]
            for branch in arms.values()
        )
        if matched != row["passed"]:
            raise ValueError(
                "Saved placebo conclusion differs from route-blind comparisons"
            )
        cells.append(matched)
    checks = validation["checks"]
    cells.append(
        checks["replay"]["replay_end"] == checks["replay"]["uninterrupted_end"]
    )
    release = checks["release"]
    cells.append(
        release["claim_ns"]
        < release["intervening_commit_ns"]
        < release["ack_ns"]
        <= release["mark_ns"]
    )
    passed = all(cells) and all(item["passed"] for item in checks.values())
    if passed != validation["passed"]:
        raise ValueError("Validation reconstruction disagrees with saved status")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="directory", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Defaults to the saved pilot.toml")
    args = parser.parse_args()
    saved = json.loads((args.directory / "summary" / "verdict.json").read_text())
    c = read_config(args.config or args.directory / "pilot.toml")
    result = audit(c, args.directory, saved, audit_prerequisites(c, args.directory))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
