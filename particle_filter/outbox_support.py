"""Deterministic count, age and median-delay support of a FIFO publisher interval."""

import numpy as np

from simulator.config import SECOND

CYCLE_NS = np.array([4_000_000, 4_000_000, 52_000_000])
MAX_MARKS = 1 + 5 * SECOND // CYCLE_NS
# Band zero is the singleton {0}; subsequent bands are open left, closed right.
BAND_UPPER = np.array([0.0, 0.025, 0.1, 0.5, 2.0, 5.0, 10.0, 20.0, np.inf])


def age_support(count, marks, pending, previous_age):
    lower = np.where((count >= 0) & (marks < count), 5.0, 0.0)
    upper = np.where((count >= 0) & (marks >= count), 5.0, previous_age + 5.0)
    exact = np.where(
        pending == 0,
        0.0,
        np.where((count > 0) & (marks == 0), previous_age + 5.0, np.nan),
    )
    return lower, upper, exact


def feasible_bands(lower, upper):
    left = np.r_[-np.inf, BAND_UPPER[:-1]]
    return (
        (lower[:, None] <= upper[:, None])
        & (BAND_UPPER >= lower[:, None])
        & (left < upper[:, None])
    )


def _delay_bounds(regime, marks, previous_count, previous_age, oldest_age):
    """Bounds on the median completed delay, including a partially processed first row."""
    # Work in integer ticks for band endpoints. For M=0 the delay factor is
    # omitted. A placeholder rank avoids indexing an empty order statistic.
    m = np.clip(marks, 1, MAX_MARKS[regime]).astype(int)
    inherited = np.where(previous_count < 0, m, np.minimum(previous_count, m))
    age_ns = np.rint(previous_age * SECOND).astype(int)
    oldest_ns = np.rint(oldest_age * SECOND)
    cycle = CYCLE_NS[regime]

    def middle_bounds(rank):
        # Mark i must complete by 5-(M-i)*cycle. Its delay upper bound adds
        # the prior oldest age only for an inherited row. Within the inherited
        # and new groups the bounds form two sorted sequences. Partition their
        # first 'rank' values where old[i] <= new[rank-i+1].
        old_used = np.floor_divide((inherited + rank + 1) * cycle - age_ns, 2 * cycle)
        old_used = np.clip(
            old_used, np.maximum(0, rank - (m - inherited)), np.minimum(rank, inherited)
        )
        new_used = rank - old_used
        last_old = 5 * SECOND + age_ns - (m - old_used) * cycle
        last_new = 5 * SECOND - (m - inherited - new_used) * cycle
        upper = np.maximum(
            np.where(old_used > 0, last_old, 0), np.where(new_used > 0, last_new, 0)
        )
        # Mark i cannot complete before (i-1)*cycle. Every marked commit
        # precedes any surviving pending row's commit. A first inherited
        # acknowledgement may have begun under the faster preceding regime.
        first_min = np.where((rank == 1) & (inherited > 0), 4_000_000, cycle)
        lower = np.maximum(first_min, oldest_ns - 5 * SECOND + (rank - 1) * cycle)
        return lower, upper

    left_min, left_max = middle_bounds((m + 1) // 2)
    right_min, right_max = middle_bounds(m // 2 + 1)
    return (left_min + right_min) / (2 * SECOND), (left_max + right_max) / (2 * SECOND)


def feasible_delay_bands(regime, marks, previous_count, previous_age, oldest_age):
    return feasible_bands(
        *_delay_bounds(regime, marks, previous_count, previous_age, oldest_age)
    )


def observation_support(marks, pending, age, delay, regime, prev, prev_age):
    delay_min, delay_max = _delay_bounds(regime, marks, prev, prev_age, age)
    lower = np.maximum(0, prev - marks)
    low_age, high_age, exact = age_support(prev, marks, pending, prev_age)
    # Measurements are produced on integer nanosecond ticks. Adding five seconds
    # to a previous floating-point age can differ by a few machine epsilons.
    same_age = np.isclose(age, exact, atol=0.1 / SECOND, rtol=0.0)
    valid = (
        (marks >= 0)
        & (marks <= MAX_MARKS[regime])
        & (marks == np.floor(marks))
        & (pending >= lower)
        & (pending == np.floor(pending))
        & np.isfinite(age)
        & (age >= low_age)
        & (age <= high_age + 1.0e-10)
        & (np.isnan(exact) | same_age)
        & np.where(
            marks == 0,
            np.isnan(delay),
            np.isfinite(delay)
            & (delay >= delay_min - 1.0e-10)
            & (delay <= delay_max + 1.0e-10),
        )
    )
    return valid, lower, exact, feasible_bands(low_age, high_age)
