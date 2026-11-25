"""Freeze horizon records before following outstanding writes for latency diagnostics."""

from dataclasses import dataclass

import numpy as np

from experiments.config import MS, SECOND
from simulator.model import KINDS, WRITE, CLAIM, MARK

MISSING = -(10**18)


def score_window(arrival, eligible, c):
    deadline = arrival + c.deadline_ms * MS
    if not eligible or arrival <= 0 or deadline > c.horizon * SECOND:
        return None
    return (deadline - 1) // (c.window * SECOND)


def breaches(misses, denominator, fraction):
    return misses > fraction * denominator


def latency_quantiles(latency, censored, probabilities=(0.5, 0.95, 0.99)):
    """Return a lower bound where a right-censored observation leaves a rank unresolved."""
    if not len(latency):
        return [None] * len(probabilities)
    lower = np.quantile(latency, probabilities, method="inverted_cdf")
    upper = np.quantile(
        np.where(censored, np.inf, latency), probabilities, method="inverted_cdf"
    )
    return [float(a) if a == b else {"lower": float(a)} for a, b in zip(lower, upper)]


@dataclass
class Lifecycle:
    arrivals: np.ndarray
    commits: np.ndarray
    marks: np.ndarray

    @classmethod
    def from_model(cls, model):
        rows = []
        for write in model.writes.values():
            previous = write.arrival
            incomplete = False
            for stamp in (write.commit, write.claim, write.ack, write.mark):
                if stamp is None:
                    incomplete = True
                elif incomplete or stamp < previous:
                    raise AssertionError(
                        "Invalid commit/claim/acknowledgement/mark order"
                    )
                else:
                    previous = stamp
            rows.append(
                (
                    write.arrival,
                    MISSING if write.commit is None else write.commit,
                    MISSING if write.mark is None else write.mark,
                )
            )
        return cls(*np.asarray(rows, dtype=np.int64).reshape(-1, 3).T)

    def pending(self, tick):
        pending = (
            (self.commits != MISSING)
            & (self.commits <= tick)
            & ((self.marks == MISSING) | (self.marks > tick))
        )
        count = int(pending.sum())
        age = (tick - int(self.commits[pending].min())) / SECOND if count else 0.0
        return count, age

    def window(self, left, right):
        commits = int(((self.commits > left) & (self.commits <= right)).sum())
        processed = (self.marks > left) & (self.marks <= right)
        marks = int(processed.sum())
        initial, initial_age = self.pending(left)
        final, final_age = self.pending(right)
        assert final == initial + commits - marks
        delays = (self.marks[processed] - self.commits[processed]) / SECOND
        return dict(
            commits=commits,
            marks=marks,
            pending=final,
            pending_start=initial,
            empty=final == 0,
            age_s=final_age,
            age_start_s=initial_age,
            throughput=marks * SECOND / (right - left),
            delay_p50_s=float(np.median(delays)) if marks else None,
            delay_p95_s=float(np.quantile(delays, 0.95)) if marks else None,
        )


@dataclass
class Visits:
    kinds: np.ndarray
    identities: np.ndarray
    queued: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    def queue(self, tick):
        outstanding = (self.queued <= tick) & (self.ends > tick)
        waiting = outstanding & (self.starts > tick)
        residual_ns = (
            self.ends[outstanding] - np.maximum(self.starts[outstanding], tick)
        ).sum()
        return int(waiting.sum()), float(residual_ns / SECOND)

    def deferred_publication(self, tick, c):
        # Each admitted write owes two visits. Subtract work as each visit joins the DB queue.
        offered = self.kinds[self.queued <= tick]
        writes = np.count_nonzero(offered == WRITE)
        claims_due = writes - np.count_nonzero(offered == CLAIM)
        marks_due = writes - np.count_nonzero(offered == MARK)
        return (claims_due * c.claim_ms + marks_due * c.mark_ms) / 1000

    def window(self, left, right):
        data = {}
        for kind, name in enumerate(KINDS):
            selected = self.kinds == kind
            began = selected & (self.starts > left) & (self.starts <= right)
            admitted = selected & (self.queued > left) & (self.queued <= right)
            completed = selected & (self.ends > left) & (self.ends <= right)
            overlap = np.minimum(self.ends[selected], right) - np.maximum(
                self.starts[selected], left
            )
            waits = (self.starts[began] - self.queued[began]) / MS
            data[name] = dict(
                visits=int(admitted.sum()),
                completed=int(completed.sum()),
                busy_s=float(np.maximum(0, overlap).sum() / SECOND),
                offered_s=float(
                    (self.ends[admitted] - self.starts[admitted]).sum() / SECOND
                ),
                waits=len(waits),
                wait_sum_ms=float(waits.sum()),
                wait_p95_ms=float(np.quantile(waits, 0.95)) if len(waits) else None,
                still_waiting=int(
                    (selected & (self.queued <= right) & (self.starts > right)).sum()
                ),
            )
        return data


def horizon_records(model, cohort):
    c = model.c
    events = Lifecycle.from_model(model)
    visits = Visits(*np.asarray(model.db.visits, dtype=np.int64).reshape(-1, 5).T)
    assert np.all(visits.starts >= visits.queued) and np.all(
        visits.ends[:-1] <= visits.starts[1:]
    )
    denominators = [0] * (c.horizon // c.window)
    misses = denominators.copy()
    for identity, window in cohort:
        write = model.writes[identity]
        denominators[window] += 1
        misses[window] += (
            write.commit is None or write.commit > write.arrival + c.deadline_ms * MS
        )

    records = []
    for i, (denominator, late) in enumerate(zip(denominators, misses)):
        left = i * c.window * SECOND
        right = left + c.window * SECOND
        outbox = events.window(left, right)
        db = visits.window(left, right)
        initial_queue, initial_work = visits.queue(left)
        queue, work = visits.queue(right)
        busy = sum(item["busy_s"] for item in db.values())
        offered = sum(item["offered_s"] for item in db.values())
        relay_owed = db["write"]["visits"] * (c.claim_ms + c.mark_ms) / 1000
        cohort_work = db["write"]["offered_s"] + db["query"]["offered_s"] + relay_owed
        initial_deferred = visits.deferred_publication(left, c)
        deferred = visits.deferred_publication(right, c)
        assert abs(initial_work + offered - busy - work) < 1e-8
        assert (
            abs(initial_work + initial_deferred + cohort_work - busy - work - deferred)
            < 1e-8
        )
        assert busy <= c.window + 1e-10 and outbox["marks"] == db["mark"]["completed"]
        records.append(
            dict(
                window=i + 1,
                eligible=denominator,
                misses=late,
                breach=breaches(late, denominator, c.miss_fraction),
                **outbox,
                queue_start=initial_queue,
                queue=queue,
                work_start_s=initial_work,
                work_s=work,
                realised_cohort_work_s=cohort_work,
                relay_owed_s=relay_owed,
                deferred_start_s=initial_deferred,
                deferred_s=deferred,
                utilisation=busy / c.window,
                db=db,
            )
        )
    assert records[-1]["pending"] == len(model.pending)
    return records


def finish_latencies(model, cohort):
    c = model.c
    cap = (c.horizon + c.drain) * SECOND
    unfinished = {i for i, _ in cohort if model.writes[i].commit is None}
    if unfinished:
        latest = max(
            end for k, i, _, _, end in model.db.visits if k == WRITE and i in unfinished
        )
        model.advance(min(cap, latest))
    rows = [model.writes[i] for i, _ in cohort]
    elapsed = np.array(
        [(w.commit if w.commit is not None else cap) - w.arrival for w in rows],
        dtype=np.int64,
    )
    java = np.array([w.java for w in rows], dtype=bool)
    censored = np.array([w.commit is None for w in rows], dtype=bool)
    return elapsed, java, censored
