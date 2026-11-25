"""Physical workload parameters and integer clock units."""

from dataclasses import dataclass

SECOND = 1_000_000_000
MS = 1_000_000


@dataclass(frozen=True)
class PlantConfig:
    seed: int
    warmup: int
    horizon: int
    writes: float
    ruby_ms: int
    java_ms: int
    insert_ms: int
    claim_ms: int
    mark_ms: int
    ack_ms: int
    slow_ms: int
    query_ms: int
