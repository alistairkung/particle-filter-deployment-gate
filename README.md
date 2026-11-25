# Particle-filter deployment gate research

AIMS5701 studies whether transactional-outbox observations can inform a change in Java traffic from 25% to 60% when both routes share a FIFO database. The simulator represents a synthetic workload. The research separates the physical plant, the observations available at decision time, and the evaluator that knows the paired outcomes.

| Milestone | Finding and evidence |
|---|---|
| 1 · Feasibility | **GO.** The routing intervention can change write-latency breach risk. [Pilot report](feasibility_pilot/FEASIBILITY_REPORT.md) · [Protocol](feasibility_pilot/SPEC.md) · [Reproduction](feasibility_pilot/README.md) · [Independent audit](feasibility_pilot/results/frozen/summary/independent-audit.json) |
| 2 · Telemetry sufficiency | **READY** under the tested conditions. [Milestone closeout](docs/milestones/002-telemetry-sufficiency.md) · [Scientific specification](telemetry_sufficiency/SCIENTIFIC_SPEC.md) · [Experimental Readiness Report](telemetry_sufficiency/results/v1/Experimental_Readiness_Report.md) · [Run guide](telemetry_sufficiency/README.md) · [Independent verification](telemetry_sufficiency/results/v1/verification-review.json) |
| 3 · Particle filtering and counterfactual forecasting | Planned. Compare action-risk forecasts against the existing simple baselines on fresh protected roots. [Simulator v1 promotion and freeze plan](docs/simulator-v1-promotion.md) |

The telemetry study’s protected data have been inspected. Preserve entire root families in every split and reserve fresh independent roots for future evaluation.

## Reproduce the milestones

The [storage verification summary](evidence_migration/README.md) links the file inventories, typed schemas and independent equivalence receipts. To regenerate a report from the published Parquet tables, follow the [pilot instructions](feasibility_pilot/README.md) or the [telemetry instructions](telemetry_sufficiency/README.md). To check the migration itself, use a full Git checkout containing both original evidence commits. Historical Git objects remain part of the repository.

For the original source files, extract the result directories into an empty directory with `git archive`. For example:

```sh
mkdir -p /tmp/aims5701-originals
git archive bb20ebed72471f43443c873a11b2ac6e32fbbc00 feasibility_pilot/results/frozen | tar -x -C /tmp/aims5701-originals
git archive 624541bae2405fe47a59a193a55482775a7e0d0a telemetry_sufficiency/results/v1 | tar -x -C /tmp/aims5701-originals
```

The historical manifests and receipts include original machine paths. The scientific protocol also cites a local Codex review discussion. The linked repository reports, recorded checks and portable run commands provide the evidence for a fresh checkout. Create the Python environment locally. Worker caches, run locks and scratch reproduction directories are disposable.
