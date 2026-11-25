"""Horizon records are frozen before the diagnostic drain."""
import numpy as np
from experiments.config import MS, SECOND
from simulator.model import WRITE, QUERY, CLAIM, MARK, KINDS

def score_window(arrival, eligible, c):
    deadline = arrival + c.deadline_ms * MS
    if not eligible or arrival <= 0 or deadline > c.horizon * SECOND:
        return None
    return (deadline-1) // (c.window*SECOND)

def breaches(misses, denominator, fraction):
    return misses > fraction * denominator

def latency_quantiles(latency, censored, probabilities=(.5,.95,.99)):
    """Empirical order statistics. A censored observation can lie anywhere above its bound."""
    if not len(latency):
        return [None] * len(probabilities)
    low = np.quantile(latency, probabilities, method="inverted_cdf")
    high = np.quantile(np.where(censored, np.inf, latency), probabilities, method="inverted_cdf")
    return [float(a) if a == b else {"lower":float(a)} for a,b in zip(low,high)]

def _lifecycle_array(model):
    rows = []
    for w in model.writes.values():
        chain = [w.commit,w.claim,w.ack,w.mark]
        previous = w.arrival
        seen_missing = False
        for stamp in chain:
            if stamp is None:
                seen_missing = True
            else:
                if seen_missing or stamp < previous:
                    raise AssertionError("Invalid outbox lifecycle")
                previous = stamp
        rows.append([w.arrival, *(stamp if stamp is not None else -10**18 for stamp in chain)])
    return np.array(rows, dtype=np.int64).reshape(-1,5).T

def horizon_records(model, cohort):
    c = model.c
    arrival, commit, claim, ack, mark = _lifecycle_array(model)
    missing = -10**18
    # Every offered write in the experiment is event-eligible.
    assert all(w.eligible for w in model.writes.values())
    visits = np.array(model.db.visits, dtype=np.int64).reshape(-1,5)
    kinds, ids, enqueued, starts, ends = visits.T
    assert np.all(starts >= enqueued) and np.all(ends[:-1] <= starts[1:])
    denom = [0]*(c.horizon//c.window)
    missed = denom.copy()
    for identity, window in cohort:
        w = model.writes[identity]
        denom[window] += 1
        missed[window] += w.commit is None or w.commit > w.arrival+c.deadline_ms*MS

    def pending_at(t):
        pending = (commit != missing) & (commit <= t) & ((mark == missing) | (mark > t))
        count = int(pending.sum())
        return count, (t-int(commit[pending].min()))/SECOND if count else 0.

    def queue_at(t):
        outstanding = (enqueued <= t) & (ends > t)
        queued = outstanding & (starts > t)
        remaining = np.maximum(ends[outstanding]-np.maximum(starts[outstanding], t),0).sum()
        return int(queued.sum()), float(remaining/SECOND)

    records = []
    for i,(n,m) in enumerate(zip(denom,missed)):
        left = i*c.window*SECOND
        right = left + c.window*SECOND
        commits = int(((commit > left) & (commit <= right)).sum())
        processed = (mark > left) & (mark <= right)
        marks = int(processed.sum())
        old, initial_age = pending_at(left)
        pending, age = pending_at(right)
        assert pending == old + commits - marks
        initial_queue, initial_work = queue_at(left)
        queue, work = queue_at(right)
        delays = (mark[processed]-commit[processed])/SECOND
        data = {}
        for k,name in enumerate(KINDS):
            mask = kinds == k
            began = mask & (starts > left) & (starts <= right)
            admitted = mask & (enqueued > left) & (enqueued <= right)
            completed = mask & (ends > left) & (ends <= right)
            busy_ns = np.maximum(0, np.minimum(ends[mask],right)-np.maximum(starts[mask],left)).sum()
            waits = (starts[began]-enqueued[began])/MS
            data[name] = dict(visits=int(admitted.sum()), completed=int(completed.sum()),
                              busy_s=float(busy_ns/SECOND), offered_s=float((ends[admitted]-starts[admitted]).sum()/SECOND),
                              waits=len(waits), wait_sum_ms=float(waits.sum()),
                              wait_p95_ms=float(np.quantile(waits,.95)) if len(waits) else None,
                              still_waiting=int((mask & (enqueued <= right) & (starts > right)).sum()))
        busy = sum(d["busy_s"] for d in data.values())
        admitted_work = sum(d["offered_s"] for d in data.values())
        assert abs(initial_work + admitted_work - busy - work) < 1e-8
        assert busy <= c.window + 1e-10 and marks == data["mark"]["completed"]
        relay_owed = data["write"]["visits"]*(c.claim_ms+c.mark_ms)/1000
        records.append(dict(window=i+1, eligible=n, misses=m, breach=breaches(m,n,c.miss_fraction),
                            commits=commits, marks=marks, pending=pending, pending_start=old,
                            empty=pending==0, age_s=age, age_start_s=initial_age, throughput=marks/c.window,
                            delay_p50_s=float(np.median(delays)) if marks else None,
                            delay_p95_s=float(np.quantile(delays,.95)) if marks else None,
                            queue_start=initial_queue,queue=queue, work_start_s=initial_work,work_s=work,
                            realised_cohort_work_s=data["write"]["offered_s"]+data["query"]["offered_s"]+relay_owed,
                            relay_owed_s=relay_owed, utilisation=busy/c.window, db=data))
    assert records[-1]["pending"] == len(model.pending)
    return records

def finish_latencies(model, cohort):
    c = model.c
    cap = (c.horizon+c.drain)*SECOND
    unfinished = {i for i,_ in cohort if model.writes[i].commit is None}
    if unfinished:
        latest = max(end for k,i,_,_,end in model.db.visits if k == WRITE and i in unfinished)
        model.advance(min(cap,latest))
    rows = [model.writes[i] for i,_ in cohort]
    elapsed = np.array([(w.commit if w.commit is not None else cap)-w.arrival for w in rows],dtype=np.int64)
    java = np.array([w.java for w in rows],dtype=bool)
    censored = np.array([w.commit is None for w in rows],dtype=bool)
    return elapsed,java,censored
