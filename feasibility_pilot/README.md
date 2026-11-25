# AIMS5701 feasibility pilot

The [report](FEASIBILITY_REPORT.md) gives the frozen verdict, the C2/C3 migration finding and the independent C5 publisher finding. The approved [specification](SPEC.md) and [implementation plan](IMPLEMENTATION_PLAN.md) define the contract. All times in the simulator are integer nanoseconds. Reported work and delays use the units in their column headings.

## Reproduce the experiment

Use CPython 3.14.0 (tested on macOS arm64). From the repository root:

```sh
cd feasibility_pilot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pytest -q tests
.venv/bin/python -m experiments validate --config configs/pilot.toml --out results/reproduction
.venv/bin/python -m experiments run --config configs/pilot.toml --out results/reproduction
.venv/bin/python -m experiments report --from results/reproduction --plots results/reproduction/plots --output results/reproduction/FEASIBILITY_REPORT.md
```

For an independent numerical reconstruction use `.venv/bin/python -m experiments.audit --from results/frozen`. For an exact comparison of a complete second execution use `.venv/bin/python -m experiments.compare results/frozen results/reproduction`. The second run replays the original seeds. It adds no statistical replications. The default worker count is at most eight. Supply `--workers 1` to either simulation command for serial execution.

## Evidence and retry behaviour

- [Manifest](results/frozen/manifest.json) and [input snapshot](results/frozen/pilot.toml) identify the frozen settings, source, seed and runtime. Scientific families have independent streams. The C0 seed labels are master/C0/distribution/arrival-rate/root/arrival-or-service. Bootstrap labels are master/bootstrap/family. All stream names are hashed using SHA-256, as documented in the manifest.
- [Queue validation](results/frozen/validation/queue.json), [placebo results](results/frozen/validation/placebo.json) and [replay and DB-release fixtures](results/frozen/validation/checks.json) precede the scientific run. Structural invariants and the [workload checks](results/frozen/validation/workload-integrity.json) cover the frozen branch data.
- Under `results/frozen/branches/`, each JSON file holds one independent root with all its branches and windows. The matching NPZ contains raw scored write latencies in nanoseconds, Java flags and censor flags in request order. `windows/` contains flat per-window exports. [Summary tables](results/frozen/summary/) and [plots](plots/) can be regenerated from these records.
- Each worker publishes samples before the corresponding branch record, which acts as the root completion marker. Retry with the same inputs to reuse valid roots. A changed source signature requires a fresh output directory. An interrupted or failed stage is visible in the manifest. Preserve the original directory when diagnosing invalid results.
- The generated report records one scientific verdict. Model calibration, production deployment tolerances and particle-filter performance are outside this experiment.

## Engineering choices and deviations

The simulation uses SimPy's event scheduler and a capacity-one FIFO DB. At enqueue time the DB reserves the interval beginning at the later of arrival or the preceding reservation's end. This is exact for the specified non-preemptive queue without cancellations. Commit, claim, acknowledgement and mark effects occur only at their simulated event times. Reserving a future interval gives the evaluator a way to inspect residual work. The publisher holds no DB reservation while waiting on the sink.

Poisson interarrivals and exponential service draws are rounded to the nearest nanosecond. A zero interarrival or background service draw is clamped to one tick. All boundary checks use integer arithmetic. At a shared tick admissions have priority over completions, with stable insertion order for ties. Acknowledgement duration is chosen when it starts. The horizon snapshot includes all events at its boundary.

There were no GitHub tickets to close. The implementation used the approved milestones on one integration branch. The science budgets, workload, SLO, routing intervention and thresholds are those in the approved documents. The single logical publisher is a deliberate experimental control.
