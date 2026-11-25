# Particle-filter deployment gate research

AIMS5701 studies whether transactional-outbox observations can inform a change in Java traffic from 25% to 60% when both routes share a FIFO database. The simulator represents a synthetic workload. The research separates the physical plant, the observations available at decision time, and the evaluator that knows the paired outcomes.

| Milestone | Finding and evidence |
|---|---|
| 1 · Feasibility | **GO.** The routing intervention can change write-latency breach risk. [Pilot report](feasibility_pilot/FEASIBILITY_REPORT.md) · [Protocol](feasibility_pilot/SPEC.md) · [Reproduction](feasibility_pilot/README.md) · [Independent audit](feasibility_pilot/results/frozen/summary/independent-audit.json) |
| 2 · Telemetry sufficiency | **READY** under the tested conditions. [Milestone closeout](docs/milestones/002-telemetry-sufficiency.md) · [Scientific specification](telemetry_sufficiency/SCIENTIFIC_SPEC.md) · [Experimental Readiness Report](telemetry_sufficiency/results/v1/Experimental_Readiness_Report.md) · [Run guide](telemetry_sufficiency/README.md) · [Independent verification](telemetry_sufficiency/results/v1/verification-review.json) |
| 3 · Particle filtering and counterfactual forecasting | Planned. Compare action-risk forecasts against the existing simple baselines on fresh protected roots. [Simulator v1 promotion and freeze plan](docs/simulator-v1-promotion.md) |

The telemetry study’s protected data have been inspected. Preserve entire root families in every split and reserve fresh independent roots for future evaluation.
