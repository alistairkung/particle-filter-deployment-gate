# Milestone 2 · Outbox telemetry sufficiency

**READY** for a controlled forecasting study under the tested synthetic workload and observation contract. The question was whether observations available before changing Java traffic could reveal hidden database demand and the reliability consequences of each available action. The [Experimental Readiness Report](../../telemetry_sufficiency/results/v1/Experimental_Readiness_Report.md) is the authoritative quantitative record.

## Protocol and verification

The study used 2,400 independent roots split equally between development and protected evaluation. Each condition had a nominal 30-second prefix followed by 30 seconds at reference load, established contention, or publisher slowdown. A passive outbox feed delivered five-second aggregates. Each root supplied matched hold and promote continuations, sharing external draws and the same physical state at decision time. Protected reference histories also supplied matched surprise-load continuations. [Scientific protocol](../../telemetry_sufficiency/SCIENTIFIC_SPEC.md) · [Implementation contract](../../telemetry_sufficiency/IMPLEMENTATION_SPEC.md)

Independent reconstruction covered 3,200 protected branches. Among 1,200 ordinary protected pairs, promotion breached in 835 and holding avoided 370 of those breaches on the matched futures. Both actions breached in 465 pairs. The original verification passed 68 study tests and 47 feasibility tests, and reproduced the pilot’s GO across 2,430 compared files. [Scores](../../telemetry_sufficiency/results/v1/scores.json) · [Independent audit](../../telemetry_sufficiency/results/v1/independent-audit.json) · [Review](../../telemetry_sufficiency/results/v1/verification-review.json) · [Pilot receipt](../../telemetry_sufficiency/results/v1/pilot-reproduction.json)

## What the evidence supports

Relative to the best constant forecast, outbox history reduced MSE by **0.000834** for realised background demand and **0.003591** for signed incremental promotion risk. The respective simultaneous lower bounds were **0.000725** and **0.000326**. Both predeclared information claims passed.

Between-condition differences drove the pooled gains. Within each relevant condition its own constant mean outperformed the selected outbox model on both targets. Publisher slowdown was unusually easy to identify in this synthetic workload (400/400 correctly classified). A diagnostic with privileged physical variables forecast both action risks and their difference more accurately. These limits matter for interpreting what a future estimator could learn.

History improved absolute hold and promote breach forecasts over the complete current observation. Its incremental-risk MSE gain over current was **0.00146**, with a descriptive 95% interval of **−0.00076 to +0.00362**. The additional value of temporal modelling for that target remains unresolved. An unexpected load increase beginning after the decision shares the ordinary reference prefix and its forecasts; pre-decision telemetry supplies no advance warning of which future occurs.

## Next milestone and data custody

**The protected results are historical evidence.** Future particle-filter experiments must use fresh independent protected roots, keep every descendant of a root in the same partition, and exclude historical protected outcomes from estimator, likelihood, threshold and model-selection tuning.

The next research question is whether particle filtering improves held-out counterfactual forecasts over current-value and simple temporal baselines. Deployment benefits depend on decision costs and a gate evaluation beyond this milestone. The [promotion plan](../simulator-v1-promotion.md) describes the proposed extraction and subsequent freeze approval.
