"""SimPy event model of one non-preemptive, capacity-one FIFO server."""
from collections import deque
from dataclasses import dataclass
import hashlib
import simpy
from experiments.config import MS

WRITE, QUERY, CLAIM, MARK = range(4)
KINDS = ("write", "query", "claim", "mark")

class SerialDB:
    """Reserve consecutive service intervals in FIFO enqueue order.

    Once enqueued a visit cannot be preempted or cancelled. Its start is therefore
    known exactly from the preceding visit's end. Only completion callbacks may
    commit a write, release an event to the publisher or change publisher phase.
    """
    def __init__(self, env):
        self.env = env
        self.available = env.now
        self.visits = []

    def visit(self, kind, identity, demand, complete=None):
        now = self.env.now
        start = max(now, self.available)
        end = start + demand
        self.available = end
        self.visits.append((kind, identity, now, start, end))
        if complete is not None:
            self.env.timeout(end-now).callbacks.append(complete)
        return end

@dataclass(slots=True)
class Write:
    arrival: int
    java: bool
    demand: int
    eligible: bool = True
    commit: int | None = None
    claim: int | None = None
    ack: int | None = None
    mark: int | None = None

class Model:
    def __init__(self, c, start, slow=False):
        self.c = c
        self.env = simpy.Environment(initial_time=start)
        self.db = SerialDB(self.env)
        self.writes = {}
        self.pending = deque()
        self.phase = "idle"
        self.ack_due = None
        self.slow = slow

    def schedule(self, tape):
        # Tape admissions at the same tick have priority over completion callbacks.
        # The stable source order is application, reference query, extra query.
        for tick, kind, identity, work, java in tape:
            event = simpy.Event(self.env)
            event._ok = True
            event._value = (kind, identity, work, java)
            event.callbacks.append(self._admit)
            self.env.schedule(event, priority=0, delay=tick-self.env.now)

    def _admit(self, event):
        kind, identity, work, java = event.value
        if kind == "write":
            self.writes[identity] = Write(self.env.now, java, work)
            self.db.visit(WRITE, identity, work, lambda _: self._commit(identity))
        else:
            # Reference and extra streams each have their own IDs.
            query_id = identity * 2 + (kind == "extra")
            self.db.visit(QUERY, query_id, work)

    def _commit(self, identity):
        w = self.writes[identity]
        w.commit = self.env.now
        if w.eligible:
            self.pending.append(identity)
            if self.phase == "idle":
                self._claim_oldest()

    def _claim_oldest(self):
        self.phase = "claim"
        self.db.visit(CLAIM, self.pending[0], self.c.claim_ms * MS, self._await_sink)

    def _await_sink(self, _):
        w = self.writes[self.pending[0]]
        w.claim = self.env.now
        self.phase = "ack"
        ms = self.c.slow_ms if self.slow and self.env.now > 0 else self.c.ack_ms
        self.ack_due = self.env.now + ms * MS
        self.env.timeout(ms * MS).callbacks.append(self._request_mark)

    def _request_mark(self, _):
        identity = self.pending[0]
        self.writes[identity].ack = self.env.now
        self.ack_due = None
        self.phase = "mark"
        self.db.visit(MARK, identity, self.c.mark_ms * MS, self._finish_mark)

    def _finish_mark(self, _):
        w = self.writes[self.pending.popleft()]
        w.mark = self.env.now
        if self.pending:
            self._claim_oldest()
        else:
            self.phase = "idle"

    def advance(self, tick):
        if tick < self.env.now:
            raise ValueError("Cannot move backwards")
        self.env.timeout(tick-self.env.now)
        while self.env.peek() <= tick:
            self.env.step()

    def physical_digest(self, with_routes=False):
        states = [(i, w.arrival, w.demand, w.commit, w.claim, w.ack, w.mark,
                   w.java if with_routes else None) for i,w in self.writes.items()]
        physical = (self.env.now, self.db.available, self.db.visits, states,
                    list(self.pending), self.phase, self.ack_due)
        return hashlib.sha256(repr(physical).encode()).hexdigest()
