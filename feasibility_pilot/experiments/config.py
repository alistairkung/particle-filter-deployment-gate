"""Frozen experiment settings. Time fields use the units in their names or seconds."""
from dataclasses import dataclass
from pathlib import Path
import tomllib

SECOND = 1_000_000_000
MS = 1_000_000

@dataclass(frozen=True)
class Config:
    seed: int
    warmup: int
    horizon: int
    window: int
    drain: int
    deadline_ms: int
    miss_fraction: float
    margin: float
    migration_tail: float
    hold: float
    promote: float
    writes: float
    ruby_ms: int
    java_ms: int
    insert_ms: int
    claim_ms: int
    mark_ms: int
    ack_ms: int
    slow_ms: int
    query_ms: int
    reference: float
    extra: float
    overload: float
    placebo_roots: int
    migration_roots: int
    overload_roots: int
    publisher_roots: int
    queue_roots: int
    queue_warmup: int
    queue_scored: int
    queue_service_rate: int
    queue_rates: list[int]
    queue_tolerance: float
    bootstrap: int
    health_limit: float
    harm_limit: float
    age_margin: int
    success_fraction: float

    def load_fraction(self, share, queries, placebo=False):
        diff = 0 if placebo else self.java_ms - self.ruby_ms
        foreground_ms = self.ruby_ms + share * diff + self.insert_ms + self.claim_ms + self.mark_ms
        return (self.writes * foreground_ms + queries * self.query_ms) / 1000

def read_config(path):
    config = Config(**tomllib.loads(Path(path).read_text()))
    if not 0 <= config.hold < config.promote <= 1 or config.horizon % config.window:
        raise ValueError("Invalid routing weights or window layout")
    if min(config.queue_rates) <= 0 or max(config.queue_rates) >= config.queue_service_rate:
        raise ValueError("Analytical queues must be stable")
    return config
