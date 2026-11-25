"""Public readings and the passive observer of committed inserts and completed marks."""

from dataclasses import asdict, dataclass, fields, replace
import math
from statistics import median

SECOND = 1_000_000_000
INTERVAL = 5 * SECOND
START = -60 * SECOND


@dataclass(slots=True, frozen=True)
class Reading:
    interval_start_ns: int
    interval_end_ns: int
    available_at_ns: int
    pending_count: int
    oldest_pending_age_s: float
    publication_throughput_per_s: float
    completed_delay_median_s: float | None


class Collector:
    """Consumes only (effect, identity, event tick). Identities stay inside this object."""

    def __init__(self):
        self._pending = {}
        self._delays = []
        self._left = START
        self._last_event = START

    def event(self, effect, identity, tick):
        if tick <= self._left or tick < self._last_event:
            raise ValueError("Lifecycle event arrived outside the open interval")
        self._last_event = tick
        if effect == "insert" and identity not in self._pending:
            self._pending[identity] = tick
        elif effect == "mark" and identity in self._pending:
            self._delays.append(tick - self._pending.pop(identity))
        else:
            raise ValueError("Unmatched or unknown lifecycle event")

    def seal(self, end):
        if end != self._left + INTERVAL or end < self._last_event or end > 0:
            raise ValueError("Invalid interval end")
        oldest = min(self._pending.values(), default=end)
        result = Reading(
            self._left,
            end,
            end,
            len(self._pending),
            (end - oldest) / SECOND,
            len(self._delays) / 5,
            median(self._delays) / SECOND if self._delays else None,
        )
        self._left = end
        self._delays.clear()
        return result


PUBLIC_FIELDS = frozenset(f.name for f in fields(Reading))


def _validated(row):
    if isinstance(row, Reading):
        row = asdict(row)
    if not isinstance(row, dict) or set(row) != PUBLIC_FIELDS:
        raise ValueError("Unexpected public observation fields")
    r = Reading(**row)
    if any(
        type(x) is not int
        for x in (
            r.interval_start_ns,
            r.interval_end_ns,
            r.available_at_ns,
            r.pending_count,
        )
    ):
        raise ValueError("Measurement ticks and pending count must be integers")
    if (
        r.interval_end_ns not in range(START + INTERVAL, 1, INTERVAL)
        or r.interval_start_ns != r.interval_end_ns - INTERVAL
        or r.available_at_ns < r.interval_end_ns
        or r.pending_count < 0
    ):
        raise ValueError("Invalid public interval, delivery time or count")
    metrics = [r.oldest_pending_age_s, r.publication_throughput_per_s]
    if r.completed_delay_median_s is not None:
        metrics.append(r.completed_delay_median_s)
    if any(
        type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in metrics
    ):
        raise ValueError("Invalid public measurement")
    if (r.publication_throughput_per_s == 0) != (r.completed_delay_median_s is None):
        raise ValueError("Completed delay is defined exactly when marks were observed")
    if not r.pending_count and r.oldest_pending_age_s:
        raise ValueError("An empty outbox must have age zero")
    return r


def parse_history(history, cutoff=0):
    """Validate public transport data, then return immutable values available as of cutoff."""
    readings = tuple(_validated(row) for row in history)
    if len({r.interval_end_ns for r in readings}) != len(readings):
        raise ValueError("Duplicate measurement interval")
    return tuple(
        sorted(
            (r for r in readings if r.available_at_ns <= cutoff),
            key=lambda r: r.interval_end_ns,
        )
    )


class DeliveryLog:
    """Evaluator-private measurement delivery schedule, including suppressed intervals."""

    def __init__(self):
        self._scheduled = {}
        self._viewed_through = START

    def schedule(self, reading, available_at_ns):
        end = reading.interval_end_ns
        if end in self._scheduled:
            raise ValueError("Duplicate measurement interval")
        if available_at_ns is None:
            self._scheduled[end] = None
            return
        if available_at_ns <= self._viewed_through:
            raise ValueError("Cannot change history retroactively")
        self._scheduled[end] = _validated(
            replace(reading, available_at_ns=available_at_ns)
        )

    def as_of(self, cutoff):
        self._viewed_through = max(self._viewed_through, cutoff)
        return tuple(
            r
            for _, r in sorted(self._scheduled.items())
            if r is not None and r.available_at_ns <= cutoff
        )
