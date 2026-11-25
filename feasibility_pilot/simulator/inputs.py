"""Independent named PCG64 streams. IDs index input tapes, never completion order."""

import hashlib
import numpy as np
from experiments.config import MS, SECOND


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


def inputs(c, family, root, phase, share, extra_rate=0):
    """Return time-ordered (tick, class, id, demand, route) admissions."""

    def source(name):
        return rng(c.seed, family, root, phase, name)

    prefix = phase == "prefix"
    seconds = c.warmup if prefix else c.horizon
    offset = -c.warmup * SECOND if prefix else 0
    times = arrivals(source("application"), c.writes, seconds)
    uniforms = source("routing").random(len(times))
    business = source("business").exponential(size=len(times))
    java = uniforms < share
    mean = np.where(java & (family != "C1"), c.java_ms, c.ruby_ms)
    demand = np.rint(business * mean * MS).astype(np.int64) + c.insert_ms * MS
    # Prefix and future IDs occupy disjoint ranges; the sign is never used for routing.
    ids = range(-len(times), 0) if prefix else range(len(times))
    tape = list(
        zip(
            (offset + t for t in times),
            ["write"] * len(times),
            ids,
            demand.tolist(),
            java.tolist(),
        )
    )
    rates = [("reference", 0 if family == "C5" else c.reference), ("extra", extra_rate)]
    for stream, rate in rates:
        qt = arrivals(source(stream + "-arrivals"), rate, seconds)
        work = source(stream + "-work").exponential(c.query_ms * MS, len(qt))
        sizes = np.maximum(1, np.rint(work)).astype(np.int64).tolist()
        for i, (t, size) in enumerate(zip(qt, sizes)):
            identity = -(i + 1) if prefix else i
            tape.append((offset + t, stream, identity, size, False))
    return sorted(tape, key=lambda row: row[0])
