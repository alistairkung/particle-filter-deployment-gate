"""Matched posterior-predictive hold and promote breach probabilities."""

from dataclasses import dataclass

import numpy as np

from simulator.workload import rng
from .inference import REGIMES


@dataclass(frozen=True)
class Forecast:
    hold_only: int
    promote_only: int
    both: int
    neither: int
    fallbacks: int

    @property
    def pairs(self):
        return self.hold_only + self.promote_only + self.both + self.neither

    @property
    def hold(self):
        return (self.hold_only + self.both) / self.pairs

    @property
    def promote(self):
        return (self.promote_only + self.both) / self.pairs

    @property
    def delta(self):
        return (self.promote_only - self.hold_only) / self.pairs


def forecast(posterior, closure, *, pairs, seed, work_by_regime=None):
    """Forecast from a weighted cut. Ablation accepts development cut work in ns."""
    from .physical import matched_pair

    if posterior.cutoff != 0:
        raise ValueError("Action risk forecasts require the decision-cut posterior")
    if type(pairs) is not int or pairs <= 0:
        raise ValueError("A positive integer forecast-pair budget is required")
    if work_by_regime is not None:
        for regime in REGIMES:
            work = np.asarray(work_by_regime[regime])
            if (
                work.ndim != 1
                or not len(work)
                or work.dtype.kind not in "iu"
                or (work < 0).any()
            ):
                raise ValueError(
                    "Ablation requires nonnegative development cut work in ns per regime"
                )
    selected = rng(seed, "pf-forecast", "joint-draw").choice(
        len(posterior.weights),
        size=pairs,
        replace=True,
        p=posterior.weights,
    )
    cells = np.zeros((2, 2), dtype=int)
    fallbacks = 0
    for k, index in enumerate(selected):
        regime = posterior.regimes[index]
        prior = posterior.boundaries[index]
        pending = (
            prior.pending
            if posterior.cut_reading is None
            else posterior.cut_reading.pending_count
        )
        work = prior.work_ns
        if work_by_regime is not None:
            draws = work_by_regime[regime]
            i = rng(seed, "pf-forecast", "ablate", k).integers(len(draws))
            work = int(draws[i])
        boundary, fallback = closure.reconstruct(
            regime,
            work,
            pending,
            rng(seed, "pf-forecast", "reconstruct", k),
            prior=prior,
        )
        fallbacks += fallback
        hold, promote = matched_pair(
            boundary, regime, rng(seed, "pf-forecast", "future", k)
        )
        cells[hold, promote] += 1
    return Forecast(
        int(cells[1, 0]),
        int(cells[0, 1]),
        int(cells[1, 1]),
        int(cells[0, 0]),
        fallbacks,
    )
