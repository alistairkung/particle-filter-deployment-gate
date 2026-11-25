# AIMS5701 feasibility pilot implementation plan

**Ready for approval · 25 November 2026.** Execute after the researcher explicitly approves this plan and [SPEC.md](SPEC.md). The spec owns the physics, cohorts, inputs, statistical methods and verdict. The commands below are the interface to build. The deliverable is a reproducible five-control feasibility report preceded by an isolated analytical validation.

## Milestones and human budget

Budget approximately four hours of researcher involvement for the entire pilot, including the specification grill. Agent implementation and batch execution can run unattended. Count time already spent on the contract against the first allocation.

| Milestone | Agent work and completion evidence | Researcher allocation |
| --- | --- | ---: |
| 1. Freeze inputs | Review the two documents, especially the execution defaults. After approval, capture the source baseline, frozen config and seed layout. | 60 min including grill |
| 2. Verify the physical model | Build the FIFO service path, isolated queue runner, request/commit path and publisher. Produce passing analytical and deterministic traces, including one DB release during sink wait. | 45 min |
| 3. Verify branching and scoring | Replay one root to the cut. Compare state and route-blind traces, hand-check a deadline-boundary example, verify conservation and inspect a C4 and a C5 fixture. | 45 min |
| 4. Run the frozen matrix | Complete the prerequisite and C1 before interpreting scientific outcomes. Execute all planned roots. Check manifest completeness, workload accounting and branch pairing. | 30 min |
| 5. Audit and decide | Rebuild the tables and verdict from saved evidence. Inspect the full diff, a paired discordance table, the two directional bound families and aligned traces. Deliver the report, including failed or inconclusive findings. | 60 min |

## Files and interfaces

Keep the environment, dependency lock, configuration, simulator, runner, tests and generated evidence under `feasibility_pilot/`. Use small Python modules and plain functions. Keep stateful operations with the SimPy processes that own them. Validate persisted inputs and results at the boundaries. Use the relevant conventions and final-diff review in the repository's [code style guide](../CODE_STYLE.md). The PCG extraction, alias resolution, graph construction, provenance and legacy CLI requirements belong to another codebase.

| Area | Responsibility and interface |
| --- | --- |
| `configs/pilot.toml`, `experiments/config.py` | Load and validate the typed scientific configuration. Include units, control matrix, warm-ups, horizon, drain limit, root counts, seed and thresholds. Generate the load-accounting table directly from these values. |
| `simulator/inputs.py` | Given the config, family and root ID, produce the prefix and future inputs with stable job IDs. Preserve shared uniforms, unit-mean exponential business-work draws and matched query demands. |
| `simulator/model.py` | Implement a SimPy environment, one capacity-one FIFO resource and one FIFO relay. Expose timestamps, DB visits, pending rows, publisher phase and a deterministic state summary. The isolated queue uses the same DB service and instrumentation path. |
| `experiments/runner.py` | For each root, replay its prefix for the prescribed branches, verify the cut-state identity where required, apply the intervention, freeze horizon outputs and perform the bounded diagnostic drain. Return root-labelled branch and window records. |
| `experiments/metrics.py` | Score immutable horizon records and the separate diagnostic follow-up. Convert root-labelled outcomes into absolute risks, discordant counts, exact directional bounds, descriptive intervals and the spec verdict. Keep this logic independent of simulation execution. |
| `experiments/report.py`, `experiments/__main__.py` | Expose the commands below, persist results, and generate the report and plots from stored records. The report command can run without launching SimPy. |
| `tests/` | Small deterministic physical, replay, scoring and analysis fixtures. Use separate development seeds. Generate hand-checkable records covering all decision branches. |

## Automated acceptance checks

| Check | Required evidence |
| --- | --- |
| Queue and workload accounting | Reproduce the four reference waits and the warm-up exclusion in the spec. Check busy-time clipping across a boundary. Recompute C2/C3's +0.056 offered-load step and C4's all-Ruby 1.06 demand. Track realised synchronous, background and publisher work separately. |
| Commit and publication | One stable event per eligible commit on either route. Timestamp order and per-window conservation hold with a row in every publisher phase. Check oldest age when empty, claimed or acknowledged but unmarked. A write must commit during another event's ack wait. |
| Forks and streams | Cover an active long query, queued writes, pending rows and a partially elapsed sink wait. Compare same-policy prefix replay against uninterrupted execution and check independence of future root streams. In C3, removing extra-stream identities recovers the reference query stream exactly. In C5, an ack spanning the cut keeps its original completion time. |
| Equal-work control | Compare every physical timestamp and outcome by stable identity across all four C1 shares after excluding route labels. Show one real event per eligible commit at both extremes. On mismatch, save the first divergent event and its root. |
| Deadline scoring and drain | Implement every deterministic fixture in the spec's boundary-test table. Verify that the write cohort is the same for branch risks and latency diagnostics, the 120 s snapshot is immutable, and a right-censored sample can make a tail percentile indeterminate. |
| Statistical calculations | Use hand-constructed paired tables, including zero discordance and extreme cells. Check beta-quantile bounds against direct binomial-tail inversion. Interchanging hold/promote must negate the point estimate and exchange the signs of the upper/lower bounds. Exercise paired root resampling of the C3 interaction. |
| Decision rules | On synthetic summaries, test lower bound equal to +0.10, both uppers below +0.10, one upper equal to +0.10, and a failing prerequisite with otherwise qualifying bounds. Check the strict C5 health margins, a paired age difference exactly 5 s, and 216 versus 217 successes. Require both C5 health and outbox criteria. |

## Commands to deliver

Run from the repository root after implementation. Pin the tested Python version and direct/transitive dependencies in the pilot README and lock file. The runner must create its output directories.

```sh
cd feasibility_pilot
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pytest -q tests
.venv/bin/python -m experiments validate --config configs/pilot.toml --out results/frozen
.venv/bin/python -m experiments run --config configs/pilot.toml --out results/frozen
.venv/bin/python -m experiments report --from results/frozen --plots plots --output FEASIBILITY_REPORT.md
```

`validate` runs the four analytical cells and C1, writes the validation evidence and returns a failing status for failed checks. `run` consumes successful validation for the same config and simulator source, executes C2/C3, C4 and C5, and records incomplete or invalid runs explicitly. Write results atomically by family and root. On retry, reuse only complete records with matching input and source digests. Preserve evidence from failed attempts separately. `report` checks the expected roots/arms, reconstructs verdicts using full precision and labels missing or invalid evidence. Include an isolated clean reproduction using a second output directory and compare deterministic scientific outputs, excluding wall-clock metadata.

## Expected outputs

| Artifact | Contents |
| --- | --- |
| `results/frozen/manifest.json` | Config and spec digests, source revision and file digests, dependency and runtime versions, seed derivation, run status, expected/completed roots and elapsed times. Include the exact config snapshot. |
| `results/frozen/validation/` | Analytical references, seed means, tolerances and intervals, replay and conservation checks, C1 path comparisons and diagnostic traces for failures. |
| `results/frozen/branches/` and `windows/` under the same result root | One row per root and branch, plus per-window denominators, misses, breach indicators, queue and publisher telemetry, DB work and busy time. Retain compact scored latency samples/censor flags and DB wait summaries for distribution reporting. |
| `results/frozen/summary/` | Control statuses, paired outcome counts, branch risks, both directional migration bounds, signed C3 interaction, C4 absolute risks and queue growth, and every C5 health and age calculation. Preserve a machine-readable verdict with its reasons. |
| `plots/` | Branch latency distributions, risk and paired effect plots with labelled bounds, and aligned write misses, queue lengths/work, outbox backlog, age and throughput. Use the same preselected root (ID 0) in each family for illustrative traces and pooled summaries for inference. |
| `README.md` and `FEASIBILITY_REPORT.md` | Reproduction commands and assumptions, then the analytical prerequisite, five-control table, diagnostics, limitations and a single scoped verdict. Include all-Ruby risk and both branch risks even if they saturate. |

## Implementation risks and stopping conditions

- **SimPy state and ties.** Use deterministic replay of the same input prefix rather than cloning live generators. Capture the cut after processing events at that timestamp. Instrument residual service and ordering. A matching queue length alone cannot establish identical state.
- **Publisher pressure.** Offered work includes future relay visits owed by arrivals. Actual claims and marks can be deferred, especially in C4/C5. Keep the DB queue and pending relay work separately visible, and reconcile the difference before interpreting results.
- **Runtime and storage.** Use development fixtures to measure per-root runtime and output size before the frozen run. Stream root summaries and keep detailed traces for preselected roots and failures. Parallelise independent roots if needed. Preserve the seed-to-root mapping, model and fixed replication budgets.
- **Invalid or incomplete evidence.** Record the failing invariant, fixture or missing root. Correct implementation defects under the contract and rerun the affected evidence with the original seeds. If the evidence cannot be made valid within the pilot budget, report MODIFY / inconclusive with validation status and the specific limitation.
- **Valid negative or ambiguous evidence.** Apply the verdict table once to the complete frozen run. Preserve all findings, including saturated outcomes and a failed C5 isolation. Scientific changes require a separate follow-up. The pilot ends at the report and an optional feature-branch commit after approval.
