# Simulator v1 release candidate

The reviewed implementation is `141b0f2fb30dfcf60697885a3bc0b32ddb6d7971`, based on `main` at `28be222cf34414b200b23b5faf16f525d14cde88`. The proposed release name is `simulator-v1.0.0`. After approval and merge, verify the merged source and evidence against this candidate and place the tag on that merged commit. The merge SHA will be known at that point.

The root package owns the physical plant, stochastic tapes, replay and passive observations. The studies own assignment, scoring, locks, statistics and storage. The [simulator contract](../simulator/README.md) documents the interfaces and controller boundary. The authoritative scientific baselines are the [pilot at bb20ebe](https://github.com/alistairkung/particle-filter-deployment-gate/tree/bb20ebed72471f43443c873a11b2ac6e32fbbc00/feasibility_pilot) and the [telemetry publication at 624541b](https://github.com/alistairkung/particle-filter-deployment-gate/tree/624541bae2405fe47a59a193a55482775a7e0d0a/telemetry_sufficiency), generated with the fits locked by the study implementation at `fbb64198686b75e5ffcd7dcabf4935527fa7a659`.

## Reproduce the candidate

Use a full Git checkout with both historical commits available. From the root, create the study environment using the [pilot guide](../feasibility_pilot/README.md). The following commands use the original seed, budgets and locked fits. Scratch outputs belong outside the retained evidence.

```sh
python=feasibility_pilot/.venv/bin/python
"$python" -m pytest -q simulator/tests telemetry_sufficiency/tests evidence_migration/test_equivalence.py
"$python" -m evidence_migration.verify --require-cleanup --workers 6
"$python" -m evidence_migration.simulator_telemetry --workers 6

run=$(mktemp -d)
(
  cd feasibility_pilot
  python=.venv/bin/python
  "$python" -m pytest -q tests
  "$python" -m experiments validate --config configs/pilot.toml --out "$run"
  "$python" -m experiments run --config configs/pilot.toml --out "$run"
  "$python" -m experiments report --from "$run" --plots "$run/plots" --output "$run/FEASIBILITY_REPORT.md"
  "$python" -m experiments.compare results/frozen "$run" --receipt "$run/comparison.json"
)
```

The [candidate receipt](simulator-v1-verification.json) records the implementation, scope and file hashes. The [migration receipt](../evidence_migration/receipts/migration-verification.json) binds the canonical tables to immutable historical Git objects and 47 byte-identical artifacts, including the reports. The migration verifier also compares the relocated FIFO, write lifecycle, PCG64 constructor and Poisson sampler definitions against the historical source.

| Check | Evidence and interpretation |
|---|---|
| Pilot GO | All four C0 cells, 128 seed means, 16 C1 roots, 3,664 branches, 14,656 windows and 17,546,533 ordered latency, route and censor samples. The 23-file comparison covers the control findings, bounds and all paired outcomes. The C2/C3 paired lower bounds remain 0.324442812 and 0.121240824. C4 saturates and C5 passes its health and age criteria. |
| Pilot provenance and presentation | The extracted implementation has a new source signature. The comparator substitutes just that value into the expected bytes for four branch tables and one validation JSON record in memory. All other deterministic files must match byte for byte. The nine plots agree exactly. Moving the rendered Markdown changes its relative link destinations. |
| Telemetry READY | Both 1,200-root partitions are regenerated through the production observer and replay, including 800 surprise continuations. The 21 generated Parquet tables and scores match by SHA-256. The verifier applies the retained fold models to every development history and the locked final models to the protected histories. It uses the historical model and fold tables as inputs. |
| Independent reconstruction | The historical scientific audit checks 3,200 protected branches, 25,491,716 admissions, 64,800 forecast quantities and 77 comparisons. The simultaneous lower bounds remain 0.000725034 for demand and 0.000325930 for incremental risk. Counts, nulls, order, stored binary64 values and verdicts compare exactly. The independent numerical derivations retain their protocol tolerance of `1e-10` for both relative and absolute error. |

## Independent review and limits

A separate read-only reviewer inspected the complete change against the scientific contracts and coding standards. Differential replay against the pre-extraction Git tree matched complete branch records and latency arrays in C1, C23, C4 and C5, and complete protected records in all three telemetry conditions. With Python file-open audit events blocked during inference, all 4,800 protected public forecast records matched using only the public model bundles and the seven-field histories.

Review found ambiguous namespace components could map two new root keys onto a single stream and that the new prefix wrapper rejected an admission at the warmup start. The fixes preserve historical keys and allow that boundary tick. Focused regression tests cover both cases. The archived pilot comparison and render also bind the original manifest by SHA-256.

The reviewer sampled physical differentials and used the full numerical verifiers for population-wide evidence. Review and computation share a machine and dependency lock. The historical diagnostic models were reused. Every protected root has already been inspected. Future particle-filter evaluation requires fresh root families and development-only tuning. [Original promotion plan](https://github.com/alistairkung/particle-filter-deployment-gate/blob/28be222cf34414b200b23b5faf16f525d14cde88/docs/simulator-v1-promotion.md)
