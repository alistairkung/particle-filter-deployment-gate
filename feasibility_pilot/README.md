# AIMS5701 feasibility pilot

The [report](FEASIBILITY_REPORT.md) gives the frozen verdict, the C2/C3 migration finding and the independent C5 publisher finding. The approved [specification](SPEC.md) and [implementation plan](IMPLEMENTATION_PLAN.md) define the contract. All times in the simulator are integer nanoseconds. Reported work and delays use the units in their column headings.

## Reproduce the experiment

Use CPython 3.14.0 (tested on macOS arm64). From the repository root:

```sh
cd feasibility_pilot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock -r ../requirements-storage.lock
.venv/bin/python -m pytest -q tests
.venv/bin/python -m experiments validate --config configs/pilot.toml --out results/reproduction
.venv/bin/python -m experiments run --config configs/pilot.toml --out results/reproduction
.venv/bin/python -m experiments report --from results/reproduction --plots results/reproduction/plots --output results/reproduction/FEASIBILITY_REPORT.md
```

Use a fresh output directory for each clean reproduction. For an independent numerical reconstruction use `.venv/bin/python -m experiments.audit --from results/frozen`. For an exact comparison of a complete second execution use `.venv/bin/python -m experiments.compare results/frozen results/reproduction`. The second run replays the original seeds. It adds no statistical replications. To regenerate the report directly from the archived observations, use `.venv/bin/python -m experiments report --from results/frozen --plots /tmp/pilot-review/plots --output /tmp/pilot-review/report.md`. This checks every regenerated summary against its frozen counterpart and writes the report and plots at the supplied destinations. The default worker count is at most eight. Supply `--workers 1` to either simulation command for serial execution.

## Evidence and retry behaviour

- [Manifest](results/frozen/manifest.json) and [input snapshot](results/frozen/pilot.toml) identify the frozen settings, source, seed and runtime. Scientific families have independent streams. The C0 seed labels are master/C0/distribution/arrival-rate/root/arrival-or-service. Bootstrap labels are master/bootstrap/family. All stream names are hashed using SHA-256, as documented in the manifest.
- [Queue validation](results/frozen/validation/queue.json), [placebo results](results/frozen/validation/placebo.json) and [replay and DB-release fixtures](results/frozen/validation/checks.json) precede the scientific run. Structural invariants and the [workload checks](results/frozen/validation/workload-integrity.json) cover the frozen branch data.
- The [canonical observations](results/frozen/evidence/) have one partition per family (C1, C23, C4 and C5). Each contains three Zstandard-compressed Parquet tables. Branch rows identify the family, root and arm and preserve the original workload signature, cut digest, C1 trace and source sample digest. Window rows add the one-based measurement window and all outbox, queue and workload measurements, including a typed DB struct for each visit kind.
- In `samples.parquet`, each scored request has a zero-based `sample_index` within its root and arm, integer `elapsed_ns`, and boolean `java` and `censored` flags. A row group holds one root across its arms. Empty cohorts retain their branch row with a scored count of zero. Readers verify the key sequence, cohort size and exact sample digest before pooling. The four small [CSV tables](results/frozen/summary/) are report exports.
- A scientific worker saves its completion marker and sample archive under a private `.workers` directory for retry. The coordinator checks the sample checksums and publishes each complete family by directory rename. The completed run removes worker files. Failed validation attempts and failed family consolidations retain their raw worker evidence under `failures/` for diagnosis. A retry validates completed partitions and resumes unfinished roots with the same scientific signature. The manifest records interrupted and failed stages.
- To reconstruct the Parquet tables from an archived pre-migration result, run `.venv/bin/python -m experiments.migrate /path/to/original-run /path/to/destination` from this directory. It retains its inputs and checks checkpoint inventories, source fields, types and sample digests before publishing each family. New run manifests identify these tables with `evidence_format: parquet-v1`. The frozen manifest and [original reproduction receipt](results/frozen/summary/clean-reproduction.json) identify the historical experiment. Compare a new run with `experiments.compare` and optionally supply `--receipt /path/to/new-receipt.json` to save its canonical inventory digest.
- The generated report records one scientific verdict. Model calibration, production deployment tolerances and particle-filter performance are outside this experiment.

## Engineering choices and deviations

The simulation uses SimPy's event scheduler and a capacity-one FIFO DB. At enqueue time the DB reserves the interval beginning at the later of arrival or the preceding reservation's end. This is exact for the specified non-preemptive queue without cancellations. Commit, claim, acknowledgement and mark effects occur only at their simulated event times. Reserving a future interval gives the evaluator a way to inspect residual work. The publisher holds no DB reservation while waiting on the sink.

Poisson interarrivals and exponential service draws are rounded to the nearest nanosecond. A zero interarrival or background service draw is clamped to one tick. All boundary checks use integer arithmetic. At a shared tick admissions have priority over completions, with stable insertion order for ties. Acknowledgement duration is chosen when it starts. The horizon snapshot includes all events at its boundary.

There were no GitHub tickets to close. The implementation used the approved milestones on one integration branch. The science budgets, workload, SLO, routing intervention and thresholds are those in the approved documents. The single logical publisher is a deliberate experimental control.

## Implementation review

The complete pilot change was reviewed against the approved scientific specification and the repository's engineering conventions. Review reproduced a missing end-to-end case for admitting an ineligible write, two verification paths that could accept incomplete evidence, and a broken reproduction-report link. The [pre-fix evidence](results/review/pre-fix-checks.json) and [first saved verdict](results/review/pre-fix-verdict.json) identify the original run. An ineligible write still occupies the DB and can delay eligible traffic. Its business commit now contributes no outbox row or relay debt. All frozen workload writes are eligible.

The corrected paths have dedicated regression tests. Additional retry checks cover a mistyped config path, an unreadable manifest and invalidation of a previously published verdict after a failed rerender. The post-review run and its clean reproduction use the same config, root IDs, fork inputs and sampling budgets. The [pre/post review comparison](results/review/frozen-comparison.json) covers all 1,200 scientific root records and their latency sample digests. The [clean comparison receipt](results/frozen/summary/clean-reproduction.json) records deterministic file comparisons. Scientific settings and decision rules were unchanged throughout the review.

The physical model and the independent reconstruction intentionally calculate some of the same quantities separately. For example, the report uses beta quantiles while the audit directly inverts binomial tails. Keeping those implementations separate makes accidental agreement from a shared statistical helper less likely.
