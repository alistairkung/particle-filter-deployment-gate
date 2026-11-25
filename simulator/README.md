# Simulator v1 contract

Simulator v1 represents a synthetic Ruby-to-Java migration sharing a database and a transactional outbox. An experiment owns its condition assignments and its evaluator. The controller receives an immutable reading history and the controls and workload assumptions declared in its protocol.

## Evaluator entry points

| Import | Contract |
|---|---|
| `simulator.config.PlantConfig` | Frozen physical settings. Durations `warmup` and `horizon` are seconds, `writes` is a Poisson rate per second, and service parameters ending in `_ms` are integer milliseconds. The historical pilot config supplies the same attributes. |
| `simulator.workload.primitive_tape` | Takes config, root key, phase (`prefix` or `future`), and reference/optional extra query rates. Returns time-ordered primitive tuples. A root key consists of one or more nonempty slash-free string labels followed by a nonnegative integer index. Reserve a distinct namespace for each study and keep a root family together. |
| `simulator.workload.admissions` | Applies the Java share to a primitive tape. A routing uniform strictly below the share selects Java. Route changes scale the same business draw and retain the fixed atomic insert cost. |
| `simulator.replay.replay_prefix` | Reconstructs a fresh plant from the supplied prefix through zero, including all events at the cut. Returns the evaluator's live plant and a separate tuple of frozen readings available at zero. A delivery override maps a measurement end to a delivery tick or to `None` for suppression. An optional acknowledgement-duration change takes effect after events at its specified tick. |
| `simulator.replay.run_future` | Advances one live cut in place with the supplied future primitives and routing share through the inclusive horizon. Reconstruct a separate cut for each sibling. Scoring and any subsequent drain belong to the evaluator. |
| `simulator.model.Model` | Low-level evaluator access to scheduled DB visits and write lifecycles. Its `schedule` and `advance` methods also support deterministic physics fixtures. |

Generate tapes with the sampler for scientific runs. A hand-built write primitive has the shape `(tick_ns, "write", id, routing_uniform, unit_exponential_work)`. A query uses `(tick_ns, "reference" | "extra", id, work_ns)`. Supply sorted integer-nanosecond times and valid nonnegative service work. IDs must be unique within each source across both phases. The sampler assigns negative prefix IDs and nonnegative future IDs. It draws reference and extra queries independently, preserving identities if extra arrivals before an onset are discarded.

Named streams use PCG64 seeded by the big-endian SHA-256 of slash-joined seed, root components, phase and source. Poisson gaps and background service round to the nearest nanosecond with a one-tick minimum. Stable ties place writes before reference queries before extra queries. At the same tick, admissions precede completions, which follow SimPy insertion order. A single non-preemptive FIFO server reserves each service interval when the visit joins the queue. An eligible business commit creates exactly one pending event. One logical publisher claims the oldest event, waits for sink acknowledgement off the DB, and removes it on completion of its DB mark. The wait keeps the duration chosen when it began. Replaying the prefix reconstructs outstanding reservations, active residual service and pending acknowledgement timeouts.

## Controller input

`simulator.observations.parse_history(rows, cutoff)` accepts only the seven fields of `Reading` and returns a measurement-ordered tuple of frozen readings available by the cutoff. The v1 observed prefix starts empty at −60 seconds and ends at zero. Each interval is five seconds, open on the left and closed on the right. The observer sees only committed inserts and completed marks. It adds no DB visit.

| Fields | Meaning |
|---|---|
| `interval_start_ns`, `interval_end_ns`, `available_at_ns` | Measurement interval and delivery time relative to the decision. A delivery at the cutoff is visible. Missing and later deliveries are absent. |
| `pending_count`, `oldest_pending_age_s` | Rows committed and still unmarked at interval end, including claimed and acknowledged rows. Age is measured from the earliest pending commit. An observed empty outbox has count and age zero. |
| `publication_throughput_per_s`, `completed_delay_median_s` | Completed marks per second and their median commit-to-mark delay. An interval with zero marks has zero throughput and an undefined delay (`None`). |

The evaluator keeps the seed, root identity, tapes, live plant, condition, delivery schedule, queue, physical targets and future outcomes private. Pass only the readings and explicitly permitted control values to a forecaster. The observation contract is a data interface within a trusted research process.

```python
from simulator.config import PlantConfig
from simulator.replay import replay_prefix, run_future
from simulator.workload import primitive_tape

config = PlantConfig(
    seed=57012026, warmup=60, horizon=120, writes=40,
    ruby_ms=11, java_ms=15, insert_ms=1, claim_ms=1, mark_ms=1,
    ack_ms=2, slow_ms=50, query_ms=40,
)
key = ("forecast-study", "development", 0)
prefix = primitive_tape(config, key, "prefix", reference_rate=3.5)
future = primitive_tape(config, key, "future", reference_rate=3.5)
hold, history = replay_prefix(config, prefix, 0.25)
promote, sibling_history = replay_prefix(config, prefix, 0.25)
assert history == sibling_history

# The external forecaster receives history, the two shares, and its declared law.
# After forecasts are fixed, the evaluator runs and scores the matched pair.
run_future(hold, future, 0.25)
run_future(promote, future, 0.60)
```

## Use and compatibility

Study commands run directly from a full checkout with their pinned environment. For a separate simulator-only environment, install the repository with `python -m pip install .` using CPython 3.14. The distribution contains the shared Python modules and depends on NumPy and SimPy. Historical result readers, diagnostic fits, reports and Parquet dependencies belong to the study environments.

The serial DB, one logical publisher, exogenous Poisson arrivals, synthetic service distributions and passive observation feed are experimental controls. v1 represents a transient synthetic workload. It abstracts away DB locks, connection pools, parallel query execution and production publisher topology. The telemetry experiment assumes immediate delivery in its primary analysis.

Compatibility includes the ordered primitive draws, physical event times and observation semantics under the recorded seeds and runtime. Changes to those semantics require a documented scientific revision and new regression evidence. Subsequent particle-filter work needs fresh protected root families. [Validation and proposed freeze](../docs/simulator-v1-promotion.md)
