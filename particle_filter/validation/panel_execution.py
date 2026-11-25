"""Run timed, root-disjoint development jobs and preserve the evidence of each completed stage."""

from dataclasses import dataclass
import hashlib
import resource
import sys
import time

from simulator.workload import rng
from telemetry_sufficiency.custody import save

from ..forecast import forecast
from ..inference import infer


@dataclass(frozen=True)
class PublicCase:
    history: tuple
    fold: int


# Each process receives read-only development fits once. Cases carry only the
# allowed readings; outcomes and evaluation labels remain in the coordinator.
def initialise_worker(fits, cases, direct_models):
    global _fits, _cases, _direct_models
    _fits, _cases, _direct_models = fits, cases, direct_models


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak / (1024**2 if sys.platform == "darwin" else 1024))


def compute_posterior(job):
    slot, n, seed, half_life = job
    case = _cases[slot]
    fit = _fits[case.fold]
    posterior = infer(
        case.history,
        fit.closure,
        fit.likelihood,
        particles=n,
        seed=seed,
        half_life=half_life,
    )
    return posterior, peak_rss_mib()


def compute_risk(job):
    slot, posterior, k, seed, ablated = job
    fit = _fits[_cases[slot].fold]
    result = forecast(
        posterior,
        fit.closure,
        pairs=k,
        seed=seed,
        work_by_regime=fit.cut_work_ns if ablated else None,
    )
    return (result.hold, result.promote, result.delta), peak_rss_mib()


class ResourceLimit(Exception):
    """Measured cost of the next required numerical comparison exceeds available time."""


class Panel:
    def __init__(self, pool, report, output, deadline, seed):
        self.pool = pool
        self.report = report
        self.output = output
        self.deadline = deadline
        self.seed = seed

    def run(self, name, worker, jobs, *, estimate=0):
        remaining = self.deadline - time.monotonic()
        if estimate > remaining:
            raise ResourceLimit(
                f"{name} projects {estimate:.1f}s with {remaining:.1f}s remaining"
            )
        start = time.monotonic()
        results = list(self.pool.map(worker, jobs))
        elapsed = time.monotonic() - start
        self.report["stages"][name] = dict(
            wall_s=elapsed,
            jobs=len(results),
            max_worker_rss_mib=max(item[1] for item in results),
        )
        save(self.output / "convergence.json", self.report)
        return [value for value, _ in results], elapsed

    def evidence(self, filename, data):
        path = self.output / filename
        save(path, data)
        self.report["evidence_digests"][filename] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        save(self.output / "convergence.json", self.report)

    def seeds(self, purpose, repeats):
        # Generate draws before assigning them to panel slots. No model seed is
        # derived from a condition, evaluator root key or outcome.
        generator = rng(self.seed, "development-convergence", purpose)
        return generator.integers(2**63 - 1, size=(36, repeats)).tolist()


def time_all_variants(job):
    from telemetry_sufficiency.methods import predict

    slot, n, k, filter_seed, forecast_seed = job
    case = _cases[slot]
    fit = _fits[case.fold]
    predict(_direct_models, case.history)
    for half_life in (120, 60, 240):
        posterior = infer(
            case.history,
            fit.closure,
            fit.likelihood,
            particles=n,
            seed=filter_seed,
            half_life=half_life,
        )
        forecast(posterior, fit.closure, pairs=k, seed=forecast_seed)
        if half_life == 120:
            forecast(
                posterior,
                fit.closure,
                pairs=k,
                seed=forecast_seed,
                work_by_regime=fit.cut_work_ns,
            )
    return slot, peak_rss_mib()
