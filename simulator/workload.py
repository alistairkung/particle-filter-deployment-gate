"""Generate route-independent primitives and apply a routing intervention."""

from collections.abc import Iterable
import hashlib
from typing import Literal

import numpy as np

from .config import MS, SECOND, PlantConfig


type Primitive = (
    tuple[int, Literal["write"], int, float, float]
    | tuple[int, Literal["reference", "extra"], int, int]
)


def rng(seed, *key):
    name = "/".join(map(str, (seed, *key))).encode()
    entropy = int.from_bytes(hashlib.sha256(name).digest(), "big")
    return np.random.Generator(np.random.PCG64(entropy))


def arrivals(generator, rate, seconds):
    if not rate:
        return []
    times = []
    elapsed = 0
    while elapsed < seconds * SECOND:
        gaps = np.maximum(1, np.rint(generator.exponential(SECOND / rate, 512))).astype(
            np.int64
        )
        batch = elapsed + gaps.cumsum()
        times.extend(batch[batch <= seconds * SECOND].tolist())
        elapsed = int(batch[-1])
    return times


def primitive_tape(
    config: PlantConfig,
    root_key: Iterable[str | int],
    phase: Literal["prefix", "future"],
    *,
    reference_rate,
    extra_rate=0,
    extra_after_ns=None,
) -> list[Primitive]:
    """Key every draw by seed, root components, phase and source; keep stable tie order."""
    if phase not in ("prefix", "future"):
        raise ValueError("Unknown tape phase")

    parts = tuple(root_key)
    if len(parts) < 2 or type(parts[-1]) is not int or parts[-1] < 0:
        raise ValueError(
            "Root key requires namespace labels and a nonnegative integer index"
        )
    if any(type(label) is not str or not label or "/" in label for label in parts[:-1]):
        raise ValueError("Root namespace labels must be nonempty slash-free strings")
    key = (config.seed, *parts, phase)

    def stream(name):
        return rng(*key, name)

    prefix = phase == "prefix"
    seconds = config.warmup if prefix else config.horizon
    offset = -seconds * SECOND if prefix else 0
    at = arrivals(stream("application"), config.writes, seconds)
    uniforms = stream("routing").random(len(at)).tolist()
    business = stream("business").exponential(size=len(at)).tolist()
    ids = range(-len(at), 0) if prefix else range(len(at))
    rows = [
        (offset + t, "write", i, u, e)
        for t, i, u, e in zip(at, ids, uniforms, business)
    ]
    rates = [("reference", reference_rate), ("extra", extra_rate)]
    for source, rate in rates:
        times = arrivals(stream(source + "-arrivals"), rate, seconds)
        draws = stream(source + "-work").exponential(config.query_ms * MS, len(times))
        sizes = np.maximum(1, np.rint(draws)).astype(np.int64).tolist()
        for i, (t, size) in enumerate(zip(times, sizes)):
            tick = offset + t
            if (
                source == "extra"
                and extra_after_ns is not None
                and tick <= extra_after_ns
            ):
                continue
            rows.append((tick, source, -i - 1 if prefix else i, size))
    # Python's stable sort keeps application/reference/extra ordering at exact ties.
    return sorted(rows, key=lambda r: r[0])


def admissions(tape: Iterable[Primitive], share, config: PlantConfig):
    """Map each request's shared primitive draw to its routed atomic write demand."""
    if not 0 <= share <= 1:
        raise ValueError("Routing share must be in [0, 1]")
    rows = []
    for tick, source, identity, draw, *extra in tape:
        if source == "write":
            (business,) = extra
            java = draw < share
            ms = config.java_ms if java else config.ruby_ms
            work = round(business * ms * MS) + config.insert_ms * MS
            rows.append((tick, source, identity, work, java))
        else:
            rows.append((tick, source, identity, draw, False))
    return rows
