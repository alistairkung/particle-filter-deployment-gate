"""Evaluator-owned prefix reconstruction and interventions on matched primitive tapes."""

from dataclasses import replace

from .config import SECOND, PlantConfig
from .model import Model
from .observations import Collector, DeliveryLog, INTERVAL, START, Reading
from .workload import admissions


class ObservedModel(Model):
    """Tap only two completed physical effects, without scheduling another DB visit."""

    def __init__(self, config, collector):
        super().__init__(config, START)
        self.collector = collector

    def _commit(self, identity):
        super()._commit(identity)
        if self.collector is not None and self.writes[identity].eligible:
            self.collector.event("insert", identity, self.env.now)

    def _finish_mark(self, event):
        identity = self.pending[0]
        super()._finish_mark(event)
        if self.collector is not None:
            self.collector.event("mark", identity, self.env.now)


def replay_prefix(
    config: PlantConfig,
    prefix,
    share,
    *,
    passive=True,
    deliveries=None,
    ack_change=None,
) -> tuple[Model, tuple[Reading, ...]]:
    """Build a fresh plant through zero and an immutable, available-at-zero history.

    ack_change is an optional (tick, milliseconds) switch after events at that tick.
    deliveries maps measurement ends to delivery ticks, or None for a missing reading.
    """
    if -config.warmup * SECOND != START:
        raise ValueError("Observed replay requires the v1 sixty-second prefix")
    routed = admissions(prefix, share, config)
    if any(not START < row[0] <= 0 for row in routed):
        raise ValueError("Prefix admissions must fall within the pre-decision interval")
    ends = range(START + INTERVAL, 1, INTERVAL)
    cuts = set(ends)
    if ack_change is not None:
        change_at, ack_ms = ack_change
        if not START <= change_at <= 0:
            raise ValueError("Acknowledgement change outside prefix")
        cuts.add(change_at)

    collector = Collector() if passive else None
    model = ObservedModel(config, collector) if passive else Model(config, START)
    feed = DeliveryLog()
    model.schedule(routed)
    for end in sorted(cuts):
        model.advance(end)
        if collector is not None and end in ends:
            available = end if deliveries is None else deliveries.get(end, end)
            feed.schedule(collector.seal(end), available)
        if ack_change is not None and end == change_at:
            model.c = replace(config, ack_ms=ack_ms)
    if collector is not None:
        model.collector = None
    return model, feed.as_of(0)


def run_future(model, future, share):
    """Continue this evaluator-owned cut to its horizon using the supplied primitives."""
    horizon = model.c.horizon * SECOND
    routed = admissions(future, share, model.c)
    if model.env.now != 0 or any(not 0 < row[0] <= horizon for row in routed):
        raise ValueError(
            "Future replay requires a decision cut and post-cut admissions"
        )
    model.schedule(routed)
    model.advance(horizon)
