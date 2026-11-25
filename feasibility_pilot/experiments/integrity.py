"""Verify the frozen scenario assignments and accounting against saved observations."""

import math

import numpy as np

from experiments.config import MS
from experiments.runner import arms
from experiments.storage import root_paths


def verify_scientific_controls(c, out, roots):
    checked = {}
    for family, histories in roots.items():
        count = 0
        expected_arms = arms(c, family)
        for root_id, branches in enumerate(histories):
            _, path = root_paths(out, family, root_id)
            with np.load(path) as samples:
                assert_matched_history(c, family, expected_arms, branches, samples)
            count += 1
        checked[family] = count
    if c.load_fraction(0, c.overload) <= 1:
        raise ValueError("C4 all-Ruby workload must exceed DB capacity")
    return {
        "passed": True,
        "checked_roots": checked,
        "checks": [
            "Configuration-specific work accounting and preserved prefix digests",
            "Matched deadline cohorts, routing assignments and application demand",
            "Query work equality within each setting; C3 adds the shared extra stream",
            "Saved diagnostic latencies reproduce frozen horizon miss totals",
            "C4 all-Ruby offered overload and complete branch vectors",
        ],
    }


def _same_values(left, right, message):
    if not np.array_equal(left, right):
        raise ValueError(message)


def assert_matched_history(c, family, expected_arms, branches, samples):
    prefixes = {}
    cohort_sizes = None
    input_work = {}
    query_work = {}
    routing = {}
    for arm in expected_arms:
        branch = branches[arm.name]
        windows = branch["windows"]
        actual = [w["eligible"] for w in windows]
        if cohort_sizes is None:
            cohort_sizes = actual
        _same_values(actual, cohort_sizes, "Different eligible arrival cohorts")
        if prefixes.setdefault(arm.prefix_share, branch["cut"]) != branch["cut"]:
            raise ValueError("Branch prefix mismatch")
        query_rate = (0 if family == "C5" else c.reference) + arm.extra
        if (
            branch["share"] != arm.share
            or branch["query_rate"] != query_rate
            or not math.isclose(
                branch["expected_load"], c.load_fraction(arm.share, query_rate)
            )
        ):
            raise ValueError(
                "Routing or offered demand disagrees with frozen configuration"
            )
        demand = [w["db"]["write"]["offered_s"] for w in windows]
        _same_values(
            demand,
            input_work.setdefault(arm.share, demand),
            "Matched writes changed DB demand",
        )
        queries = [w["db"]["query"]["offered_s"] for w in windows]
        _same_values(
            queries, query_work.setdefault(query_rate, queries), "Unmatched query work"
        )
        elapsed = samples[arm.name + "_ns"]
        flags = samples[arm.name + "_censored"]
        java = samples[arm.name + "_java"]
        _same_values(java, routing.setdefault(arm.share, java), "Unmatched route flags")
        if (
            len(elapsed) != sum(actual)
            or len(java) != len(elapsed)
            or len(flags) != len(elapsed)
        ):
            raise ValueError("Incomplete stored latency cohort")
        if int(np.count_nonzero(flags)) != branch["censored"] or int(
            np.count_nonzero((elapsed > c.deadline_ms * MS) | flags)
        ) != sum(w["misses"] for w in windows):
            raise ValueError("Latency diagnostics disagree with frozen misses")
    if (
        c.hold in routing
        and c.promote in routing
        and np.any(routing[c.hold] & ~routing[c.promote])
    ):
        raise ValueError("Promotion lost requests already routed to Java")
    if family == "C23" and np.any(
        np.array(query_work[c.reference + c.extra]) < query_work[c.reference]
    ):
        raise ValueError("Higher-demand arm lost reference background work")
