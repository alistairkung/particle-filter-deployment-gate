# AIMS5701 · Outbox telemetry sufficiency study

**Status — approved for implementation.** This protocol closes the design grill. It specifies a bounded readiness study at one decision point. The companion [implementation and verification contract](IMPLEMENTATION_SPEC.md) defines execution, evidence and acceptance. Implementation and protected evaluation follow approval.

## Question and scope

Can the outbox history available before increasing Java traffic from 25% to 60% reveal realised background DB demand or help predict the change in write-breach risk? Estimate operating-condition discrimination, physical-state estimation, absolute action risks and incremental-risk prediction separately. The temporal question asks how much six successive readings add beyond the complete latest reading. A synthetic finding can support a forecasting experiment. Deployment costs, production thresholds, gate policies and particle filters are outside this study.

The frozen pilot established that the extra work of the Java route can increase breaches and that a slow publisher can build a backlog while writes remain healthy. Its elevated-demand futures all began from reference prefixes. Its publisher control had no promotion branch. The new study changes when disturbances begin and obtains both actions in every condition. The frozen pilot remains the benchmark for its original questions. [Pilot protocol](../feasibility_pilot/SPEC.md) · [Results](../feasibility_pilot/FEASIBILITY_REPORT.md) · [Independent scientific review, sections C–F](codex://threads/01a211dd-5dcf-7210-850b-eeff94a9cdae)

## Population, interventions and independence

Use the decision as time zero. Start empty at −60 s. During (−60,−30] all roots receive reference conditions. During (−30,0] apply the assigned condition and continue it through (0,120]. Hold 25% Java throughout the prefix. The expected foreground arrival rate is 40 writes/s in every cell.

| Condition | Background queries/s | Off-DB acknowledgement | Development roots | Protected roots |
|---|---:|---:|---:|---:|
| Reference | 3.5 | 2 ms | 400 | 400 |
| Established contention | 3.5 + 2.0 | 2 ms | 400 | 400 |
| Publisher slowdown | 3.5 | 50 ms | 400 | 400 |

Keep the pilot's Poisson arrivals, exponential business and query work, atomic outbox insert, FIFO DB, serial publisher and timing rules. An acknowledgement keeps the duration assigned when it began. The three labels identify interventions. Every fresh root is eligible, including idle, unusually busy and overlapping histories. Fixed exposure and an empty start define a transient study. Report observed overlap and any separation without changing rates, noise or onset to influence it.

Each root receives one matched future for the primary study. Fork its full physical state into hold (25%) and promote (60%). Share external arrival times, query service draws, request routing uniforms and underlying business-work draws across siblings. Keep inherited work, routes and publisher progress. Only the routing threshold for new writes changes. Root is the independent sampling unit. Its 12 readings, two actions, and any robustness branches always travel together in partitions and uncertainty calculations.

For the 400 protected reference roots, add one surprise pair starting the extra 2 queries/s at the cut. Reuse the ordinary future's foreground and reference streams. Score the already locked reference forecasts against each surprise outcome. Report the change in outcomes and scores within these roots, separately from the primary mixture. Total budget is 2,400 roots, 4,800 ordinary continuations and 800 surprise continuations. No early stopping, added futures or replacement of unfavourable histories.

## Measurements and evaluator targets

Observe passive committed-insert and completed-mark events at five-second cadence. Each reading contains pending count, oldest pending age, mark throughput and median completed commit-to-mark delay. They share one lifecycle. At the cut use only delivered readings. Immediate interval-end delivery is an idealised primary instrumentation assumption. Availability, empty, no-mark, missing and delayed cases are defined in the [observation contract](IMPLEMENTATION_SPEC.md#workstream-1--time-causal-outbox-telemetry).

| Evaluator target | Definition and interpretation |
|---|---|
| Realised background demand, primary physical target | Sum the full assigned service requirements of background queries arriving in (−30,0], divide by 30 s. Unit is DB-service seconds per elapsed second. The realised value can exceed one. |
| Total unfinished DB work, secondary | Seconds of residual service on the active visit plus full service for visits already queued after processing all cut events. Future publisher visits owed by pending rows enter only once enqueued. |
| Operating condition | Three-way intervention label. Configured intensity generates stochastic demand. Publisher condition determines off-DB delay. Neither a class label nor its classifier alone establishes recovery of realised demand. |
| Action outcomes | For each continuation score the pilot's binary pooled-write endpoint. Include eligible writes arriving strictly after the cut with deadlines at or before 120 s. Assign their 250 ms deadlines to four consecutive intervals open on the left, closed on the right. Commit at deadline is on time. Any nonempty window with strictly more than 5% misses gives one breach. Freeze before any diagnostic drain. |

For available history H, let R(a|H) be the conditional breach probability for action a under the equal-condition study mixture. ΔR(H) = R(promote|H) − R(hold|H). A root supplies D = Y_promote − Y_hold, a noisy observation with E[D|H] = ΔR(H). One matched future permits population forecast assessment. It gives no exact individual risk label.

| Hold | Promote | Paired interpretation | D |
|---:|---:|---|---:|
| 0 | 0 | Neither breaches. Safe promotion opportunity. | 0 |
| 0 | 1 | Promote only. Avoidable breach under this coupling. | +1 |
| 1 | 0 | Hold only. Safe promotion opportunity. | −1 |
| 1 | 1 | Both breach. Holding does not avoid this breach. | 0 |

Report all four counts over all roots in each applicable population, absolute rates and signed mean D. The promote-only probability depends on the coupling. Its difference from the hold-only probability gives the incremental risk.

## Diagnostic information and fitting

| Level | Inputs |
|---|---|
| Prior | Development action-specific breach rates, physical target means or class frequencies, as appropriate. |
| Current oldest age | Age in the most recent delivered aggregate through the cut. |
| Current outbox | All four measurements in that same aggregate. |
| History | Current outbox plus mean and ordinary least-squares slope for each measurement over the final six scheduled interval ends (−25, −20, −15, −10, −5, 0 s). Use actual interval-end times in seconds. Include each measurement's defined count. An undefined delay and an unavailable sample both reduce counts, with separate delivery information as specified in the engineering contract. |

Compute summaries in native units before transformations. For current and mean count, age, rate and delay use log(1+x) with values in rows, seconds or marks/s as applicable. Keep slopes signed on their native scales. Include missingness indicators for undefined numeric features. Impute development-column medians of defined values, zero if none. Then centre and scale with training means and population standard deviations. Use scale one for a constant column. Fit transformations on each training fold and then on all development roots for final fits. Use the last reading at most once per interval. No future fill, regime features, or identifiers.

Fit separate ridge logistic regressions for hold and promote. Their difference supplies the increment. Use ridge linear regression for each physical target in native units, clipping negative predictions to zero before scoring. Use ridge multinomial logistic regression for conditions. Loss is the mean binary or multiclass negative log likelihood, or half mean squared error for a linear model, plus 0.01 times half the sum of squared nonintercept coefficients. Intercepts are unpenalised. All learned levels use identical fitting rules and one fixed penalty. There is no search over thresholds, windows or penalties. Class frequencies and all training steps use training rows only. For constant response in a training fold use its constant prediction. Report convergence and numerical failures. Deterministic fitting details and locks are part of the companion contract.

The evaluator-only action-risk diagnostic fits the same two logistic models, with three one-hot condition columns in table order, realised demand and unfinished work as inputs. Use training-only standardisation of all five columns, including the indicators; use native units for demand and work before standardisation. It follows the same split, fitting, scoring and calibration discipline. The evaluator independently supplies supervised development targets to each model. The evaluator-only fitted model supplies nothing to the outbox models. Its performance measures what this simple diagnostic extracts from these physical variables. It sets no upper bound on achievable prediction.

## Scores and two information claims for READY

**Forecast accuracy.** Promotion Brier score is the primary absolute forecast-accuracy measure. Also score hold with Brier and ΔR with mean squared error against D. Differences of these last losses on common roots compare conditional-mean prediction in expectation even though individual D values are noisy. Estimate demand and unfinished work with MSE and report RMSE in physical units. Score condition probabilities with the sum of the three squared probability errors per root, plus a confusion matrix (argmax, ties in table order). A good condition classifier alone cannot support the physical claim. [Gneiting and Raftery on proper scores](https://doi.org/10.1198/016214506000001437)

**Lock one outbox candidate per claim before generating protected roots.** Use five deterministic stratified development folds of 80 roots per condition each, assigned by sorting a seeded hash of the root key within condition and cycling through folds. For each of age, current outbox and history collect out-of-fold predictions. Choose the minimum out-of-fold MSE for each claim on its population below, breaking exact ties in order of increasing information level. Refit all levels on the full development set. Freeze both selections, all fitted models, preprocessing and the analysis rule in the lock. The action pair is selected jointly using its signed difference. Report the entire ladder regardless of selection.

| Confirmatory claim | Population and response | Reason |
|---|---|---|
| Physical information | Reference and established-contention protected roots with equal weights, response = realised background demand | Both have the same nominal publisher. Merely detecting the publisher intervention cannot establish this claim. Pooled three-condition and all per-condition results are also required. |
| Incremental-risk information | All three ordinary protected conditions with equal weights, response = D | Tests whether outbox observations improve signed action-risk prediction under the declared mixture. |

For each claim report the selected model's ordinary score gain over the development prior. Protect the information claim against a fortunate intercept correction with a stronger scoring comparator: the **best single constant** for the claim population. For physical demand it is the population mean demand, and for D it is the population mean increment. Neither varies with H. Estimate that constant by the equally weighted mean of protected stratum means, solely inside the scorer. It is an evaluation benchmark and never enters fitting or predictions. This benchmark's empirical loss is at most that of any development constant, including the action-only prior's signed difference.

For each protected root with target y, selected prediction f and estimated constant m, compute g = (y−m)² − (y−f)². The gain estimate G is the equally weighted mean of stratum means of g. Estimate SE² as the sum over strata of w² s²(g)/n, using sample variances. The derivative of expected constant-prediction loss with respect to m is zero at the population mean, so estimation of m has no first-order contribution. Treat fits and selections as fixed after development. For the two nulls of nonpositive population gain, report one-sided Bonferroni lower bounds G − 1.959964 × SE (individual one-sided 97.5%). Use unrounded values, with the normal quantile evaluated at 0.975. A claim passes only with positive lower bound, finite scores and positive SE. A genuine zero SE from complete, finite losses leaves that claim unsupported. A missing prediction, nonfinite loss, failed required fit or arithmetic failure invalidates the affected analysis. These are **large-sample, familywise 95% lower bounds** under independent roots with finite loss variance, not exact finite-sample bounds. Bonferroni permits dependence between the two claims. Publish the two gain estimates, SEs, bounds, candidate names and denominators.

**Descriptive comparisons.** Report score gains for current outbox versus prior and history versus current outbox on each action’s Brier score. On D report current outbox versus prior, history versus prior and history versus current outbox. Report scores for every level, target and condition and the evaluator-only action diagnostic. Also give paired history-versus-current score differences for both physical targets and for multiclass discrimination. In each condition give descriptive demand and D gains over that condition’s own protected response mean, recomputing that mean within each bootstrap resample. This measures information beyond knowing the condition and guards against interpreting differences between configured intensities as prediction of within-condition fluctuations. Include paired descriptive 95% intervals for score differences using 10,000 whole-root bootstrap draws within condition, fixed fits, fixed weights and linear-interpolated 2.5/97.5 percentiles. Preserve pairings across methods and scenarios. Report improvement as baseline loss minus candidate loss. Show reliability by action with fixed probability bins [0,.1), …, [.9,1], counts, average prediction and event rate. Display empty calibration bins with count zero and undefined means. For each pair of conditions report the histogram overlap of current age and, separately, count: sum the smaller of the two relative frequencies in each bin. For age in seconds use {0}, (0,0.1], (0.1,0.5], (0.5,1], (1,5], (5,15], (15,∞); for row counts use {0}, 1–4, 5–19, 20–99, 100–499, and ≥500. Report class confusions alongside these descriptive, bin-dependent overlap measures. Label temporal gains, secondary physical targets, surprise results, within-condition effects and calibration as diagnostics. Nominal intervals from these comparisons cannot qualify a READY claim.

## Stopping, interpretation and report

Score once after model lock, validation and complete inventories. Independent reconstruction and clean frozen-pilot reproduction precede the final verdict. An invalid run is retained with its failure and scope. A pre-unblinding engineering failure can be repaired with unchanged inputs. After access to protected results, fixes affecting fits, features, exclusions, hypotheses or conclusions require documenting exposure and a separately approved evaluation plan. Replaying the same results verifies computation and adds zero independent roots.

| Verdict | Frozen rule |
|---|---|
| READY | All material validity and reproducibility checks pass and both information claims pass the rule above, including positive SEs and strictly positive lower bounds. Current measurements can supply either or both claims. Temporal superiority is unnecessary. |
| READY WITH LIMITATIONS | The experiment is valid and complete, and at least one information claim is unsupported. Explain the strength, direction and precision of the effects. State the later research question still justified and the limitations it must address. |
| NOT READY | Material validity or reproducibility problems remain, including future leakage, mismatched continuations, split contamination, incomplete evidence, unrepaired failure of a required fit or score, or failure of independent reconstruction. |

The Experimental Readiness Report must answer the five mission questions individually. Distinguish absolute-risk skill from incremental-risk skill, publisher recognition from DB-demand estimation, information extracted by the evaluator from information extracted by outbox models, weak performance even for the evaluator, and temporal gain from snapshot sufficiency. Inspect per-condition errors and report when a pooled result is driven by between-condition differences. Where uncertainty spans both gains and losses, report that temporal improvement is unestablished. Claiming equivalence would require a prespecified equivalence margin. Limited power, floor/ceiling risks, one future per prefix, the assumed condition mixture and immediate delivery constrain inference. Report score units and physical units. Operational decision value requires costs or a policy threshold beyond this study. Reserve fresh, untouched evaluation roots for any subsequent particle-filter experiment.
