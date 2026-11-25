"""Replay independent roots; only counterfactual branches share an input history."""

from dataclasses import dataclass
from experiments.config import SECOND
from experiments.observations import finish_latencies, horizon_records, score_window
from simulator.inputs import inputs
from simulator.model import Model


@dataclass(frozen=True)
class Arm:
    name: str
    share: float
    prefix_share: float
    extra: float = 0.0
    slow: bool = False


def arms(c, family):
    if family == "C1":
        return [
            Arm(name, s, c.hold)
            for name, s in zip(
                ("zero", "hold", "promote", "all"), (0, c.hold, c.promote, 1)
            )
        ]
    if family == "C5":
        return [Arm("nominal", c.hold, c.hold), Arm("slow", c.hold, c.hold, slow=True)]
    if family == "C4":
        extra = c.overload - c.reference
        return [
            Arm("ruby", 0, 0, extra),
            Arm("hold", c.hold, c.hold, extra),
            Arm("promote", c.promote, c.hold, extra),
        ]
    if family != "C23":
        raise ValueError(family)
    return [
        Arm(regime + "_" + name, s, c.hold, rate)
        for regime, rate in (("ref", 0), ("high", c.extra))
        for name, s in (("hold", c.hold), ("promote", c.promote))
    ]


def run_root(c, family, root):
    branches, samples, prefixes = {}, {}, {}
    for arm in arms(c, family):
        model = Model(c, -c.warmup * SECOND, arm.slow)
        model.schedule(inputs(c, family, root, "prefix", arm.prefix_share))
        model.advance(0)
        cut = model.physical_digest(with_routes=True)
        if prefixes.setdefault(arm.prefix_share, cut) != cut:
            raise AssertionError(f"Prefix diverged in {family}/{root}/{arm.name}")
        model.schedule(inputs(c, family, root, "future", arm.share, arm.extra))
        model.advance(c.horizon * SECOND)
        cohort = [
            (i, window)
            for i, w in model.writes.items()
            if (window := score_window(w.arrival, w.eligible, c)) is not None
        ]
        windows = horizon_records(model, cohort)
        actual_routes = sum(w.java for w in model.writes.values() if w.arrival > 0)
        elapsed, java, censored = finish_latencies(model, cohort)
        if arm.share in (0, 1):
            assert actual_routes == arm.share * sum(
                w.arrival > 0 for w in model.writes.values()
            )
        branch = dict(
            cut=cut,
            windows=windows,
            breach=int(any(w["breach"] for w in windows)),
            censored=int(censored.sum()),
            scored=len(cohort),
            java_arrivals=actual_routes,
            share=arm.share,
            query_rate=(0 if family == "C5" else c.reference) + arm.extra,
            expected_load=c.load_fraction(
                arm.share,
                (0 if family == "C5" else c.reference) + arm.extra,
                family == "C1",
            ),
        )
        if family == "C1":
            branch["trace"] = model.physical_digest()
        samples[arm.name] = (elapsed, java, censored)
        branches[arm.name] = branch
    if family == "C1" and len({b["trace"] for b in branches.values()}) != 1:
        raise AssertionError(f"Physical placebo traces diverged for root {root}")
    return branches, samples
