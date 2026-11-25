"""Deterministic count, age and median-delay support of a FIFO publisher interval."""

import numpy as np

from simulator.config import SECOND, MS

CYCLE_NS = np.array([4_000_000, 4_000_000, 52_000_000])
HORIZON = 5 * SECOND
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


def mark_capacity(regime, previous_count):
    # First completion is strictly after start. A known empty outbox must first
    # receive a commit, then pay for the entire first publication cycle.
    return (HORIZON - 1) // CYCLE_NS[regime] + (previous_count != 0)


def _delay_bounds(regime, marks, previous_count, previous_age, oldest_age, pending):
    """Bound each ordered mark, then take medians (the delays need not be ordered)."""
    # No delay is scored without marks. A placeholder prevents an empty median.
    m = np.clip(marks, 1, mark_capacity(regime, previous_count)).astype(int)
    age_end = np.broadcast_to(np.rint(oldest_age * SECOND), m.shape).astype(int)
    survivor = np.broadcast_to(pending, m.shape) > 0
    age_start = np.rint(previous_age * SECOND).astype(int)
    lower = np.empty(len(m))
    upper = np.empty(len(m))
    for count in np.unique(m):
        rows = m == count
        index = np.arange(count)
        n0 = previous_count[rows, None]
        cycle = CYCLE_NS[regime[rows], None]
        # Only the first mark may inherit a request or an acknowledgement. Its
        # completion can occur at the first tick; subsequent cycles take c.
        first_finish = np.where(n0 == 0, 1 + cycle, 1)
        earliest_finish = first_finish + index * cycle
        latest_finish = HORIZON - (count - 1 - index) * cycle
        minimum_delay = np.where((index == 0) & (n0 != 0), 4 * MS, cycle)

        # Adjacent committed writes are separated by at least one fixed insert.
        # A stale count leaves the number of inherited rows unknown.
        earliest_commit = -age_start[rows, None] + index * MS
        known_new = (n0 >= 0) & (index >= n0)
        earliest_commit = np.where(
            known_new,
            np.maximum(earliest_commit, 1 + (index - n0) * MS),
            earliest_commit,
        )
        latest_commit = np.where(
            survivor[rows, None],
            HORIZON - age_end[rows, None] - (count - index) * MS,
            HORIZON,
        )
        latest_commit = np.where(
            index < n0, np.minimum(latest_commit, -(n0 - 1 - index) * MS), latest_commit
        )
        # FIFO gives the first marked row the *known* original commit when a
        # fresh reading had pending rows; every later inherited age is unknown.
        latest_commit[:, 0] = np.where(
            n0[:, 0] > 0,
            np.minimum(latest_commit[:, 0], -age_start[rows]),
            latest_commit[:, 0],
        )
        min_delay = np.maximum(minimum_delay, earliest_finish - latest_commit)
        max_delay = latest_finish - earliest_commit
        lower[rows] = np.median(min_delay, axis=1) / SECOND
        upper[rows] = np.median(max_delay, axis=1) / SECOND
    return lower, upper


def feasible_delay_bands(
    regime, marks, previous_count, previous_age, oldest_age, pending
):
    return feasible_bands(
        *_delay_bounds(regime, marks, previous_count, previous_age, oldest_age, pending)
    )


def observation_support(marks, pending, age, delay, regime, prev, prev_age):
    delay_min, delay_max = _delay_bounds(regime, marks, prev, prev_age, age, pending)
    lower = np.maximum(0, prev - marks)
    low_age, high_age, exact = age_support(prev, marks, pending, prev_age)
    # Measurements are produced on integer nanosecond ticks. Adding five seconds
    # to a previous floating-point age can differ by a few machine epsilons.
    same_age = np.isclose(age, exact, atol=0.1 / SECOND, rtol=0.0)
    valid = (
        (marks >= 0)
        & (marks <= mark_capacity(regime, prev))
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
