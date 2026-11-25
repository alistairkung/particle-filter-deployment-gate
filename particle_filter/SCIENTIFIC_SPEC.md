# AIMS5701 · Particle-filter scientific specification

**Scientific specification approved, 25 November 2026. Implementation and development-only validation may proceed.** Protected evaluation requires separate owner approval after development validation, method lock and runtime verification. Decisions 1–10, revised Decision 11 and revised Decision 12 are approved. Independent reviewers rechecked the proposed methods and evaluation plans after their findings were resolved. Decision 12 fixes the evaluation at 300 fresh protected roots, 100 per condition, with development timing before the freeze. [Scientific review record](SPEC_REVIEW.md)

## Primary model and scope

Infer $X_t=(Z_t,W_t)$ from available outbox readings under known routing controls. $W_t$ measures seconds of service remaining on active and queued DB visits. $Z_t$ encodes the original background intensity and publisher condition together. This keeps the three physical quantities interpretable. All forecasting methods receive the same permitted inputs and workload law. Actual condition assignments and intervention times stay with the evaluator.

| Regime | Background queries/s | Off-DB acknowledgement |
|---|---:|---:|
| A · Reference | 3.5 | 2 ms |
| B · Elevated background | 5.5 | 2 ms |
| C · Slow publisher | 3.5 | 50 ms |

**Approximate Markov state.** Each particle propagates hypothetical FIFO work at visit admissions and uses a pending count delivered by the interval start as context for sequential publisher work. The compact state omits the ordered DB queue, outstanding event timestamps, publisher phase and residual acknowledgement time. This bounds estimation and calibration within the MSc project, while allowing a direct test of whether an aggregate work estimate improves forecasts. Two histories with equal $Z_t$, $W_t$ and pending count can have different imminent marks and subsequent DB admissions. The joint observation likelihood must average over such histories. Errors in that approximation can bias the posterior over $Z_t$ and $W_t$ and propagate to the write-breach probability under either routing action. [Time-causal transition and likelihood factorisation](../docs/adr/0004-particle-workload-and-publisher-memory.md)

**Restricted regime space.** A, B and C exhaust the primary calibration and evaluation combinations. Under this experimental law, evidence for a slow publisher also weighs against elevated background intensity. That relationship limits what a good pooled score can establish about queue inference. If both disturbances occur together, the primary likelihood has no component for that combination and could explain a backlog using the wrong mechanism. That can distort queue estimates and both action forecasts. The bounded study makes no inference about whether the disturbances coexist in production. [Decision 3](../docs/adr/0005-compact-state-and-regime-support.md)

## Pre-decision regime persistence

Use a common half-life $h$ for A, B and C. It describes the time for the prior probability of staying continuously in the same regime to fall to one half. Query interarrival times are governed separately by the background intensity in that regime. The five-second departure probability is $q(h)=1-2^{-5/h}$ at every update, including those around the evaluator's intervention time. This time-homogeneous prior is a declared modelling assumption. The experiment provides no independent empirical basis for distinct regime half-lives, and the study uses neither production telemetry nor the fixed intervention schedule to estimate them. [Decision 4](../docs/adr/0006-pre-decision-regime-half-life.md)

| Analysis | Regime half-life | Five-second departure probability (rounded) |
|---|---:|---:|
| Primary | 120 s | 2.8468% |
| Shorter persistence | 60 s | 5.6126% |
| Longer persistence | 240 s | 1.4337% |

These settings apply only to inference before the cut. At the cut, integrate over the joint posterior, keeping each drawn regime fixed for both 120-second continuations while stochastic arrivals, service, DB workload and publisher processing evolve. Report separate hold and promote breach probabilities and their signed difference. [Forecast persistence](../docs/adr/0003-regime-persistence-in-action-forecasts.md)

### Transition destinations

For all three half-lives, rows are current regimes and columns are regimes five seconds later.

| From | To A | To B | To C |
|---|---:|---:|---:|
| A | $1-q$ | $q/2$ | $q/2$ |
| B | $q$ | $1-q$ | $0$ |
| C | $q$ | $0$ | $1-q$ |

The equal split on departure from A is an assumed symmetry. The graph and common departure probability imply stationary weights $(1/2,1/4,1/4)$ in A/B/C order. These weights describe the model in the long run. Each B-to-C or C-to-B particle path visits A for at least one sampled boundary. This can delay detection of a rapid handover even when the endpoint regime is supported. A simultaneous disturbance remains outside the state space. [Decision 5](../docs/adr/0007-regime-transition-destinations.md)

**Deterministic verification.** Run the executable examples below with `python3 -m doctest particle_filter/SCIENTIFIC_SPEC.md`. They check the specified matrix. At implementation verification, apply the same assertions to the matrices returned by the filter for all three half-lives.

```pycon
>>> from math import isclose
>>> def check_transition(h, matrix):
...     a, b, c = matrix
...     assert all(len(row) == 3 for row in matrix)
...     assert all(0 <= p <= 1 for row in matrix for p in row)
...     assert all(isclose(sum(row), 1.0, rel_tol=0, abs_tol=1e-12) for row in matrix)
...     # Staying continuously for one half-life has probability one half.
...     for stay in (a[0], b[1], c[2]):
...         assert isclose(stay ** (h / 5), 0.5, rel_tol=0, abs_tol=1e-12)
...     assert a[1] == a[2] > 0
...     assert b[0] == c[0] > 0
...     assert b[2] == c[1] == 0
...     stationary = (0.5, 0.25, 0.25)
...     for expected, column in zip(stationary, zip(*matrix)):
...         actual = sum(weight * p for weight, p in zip(stationary, column))
...         assert isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
>>> for h in (60, 120, 240):
...     q = 1 - 2 ** (-5 / h)
...     matrix = ((1-q, q/2, q/2), (q, 1-q, 0), (q, 0, 1-q))
...     check_transition(h, matrix)

```

### Initial regime prior

Before the first five-second interval, initialise the regime law once per root with $(1/3,1/3,1/3)$ in A/B/C order. Use it for every half-life configuration. Equal weight per hypothesis is a modelling choice among possible priors and assigns $2/3$ probability to a disturbance at startup. The choice is independent of production prevalence, stationary weights and the experiment's condition allocation. The filter has no access to the true initial regime. All baselines receive the same permitted information, while their use of priors may differ by model class. [Decision 6](../docs/adr/0008-initial-regime-prior.md)

### Physical initialisation

Immediately before the first interval, the joint law assigns probability $1/3$ to each $(Z,W)=(A,0),(B,0),(C,0)$. All hypothetical publishers start with zero inherited pending records, claims, marks and in-flight acknowledgements. Supply this cold start as a study constant to every method. Construct it independently of the evaluator. At −60 s the available reading history is empty and the measured pending count is unavailable. Propagate the first interval from the declared state. Under immediate delivery the first reading becomes available at −55 s. Missing or delayed readings preserve the particle's own evolving work and publisher progress. Apply an observation update only to delivered evidence. Later inference and both action forecasts use the estimated state throughout. [Decision 7](../docs/adr/0009-known-empty-start.md)

**Deterministic acceptance tests for the approved implementation phase**

| Fixture | Required result |
|---|---|
| Start at −60 s under each half-life. Use fixed regime draws exercising A, B and C. | Each initial particle has zero work and no inherited publisher obligations. The regime law is uniform. The history has zero readings and the likelihood has received zero calls. |
| Deliver the first reading at −55 s. Inspect immediately before and at delivery. | The interval prediction is identical for two different possible ending readings given the same particle draws. The likelihood receives the delivered reading exactly once, at the boundary. |
| End the first interval with an unfinished visit and publisher work still due on a fixed hypothetical trajectory. Suppress the first reading in one run and delay it in another. | Before delivery, both runs match uninterrupted prediction with the same draws. The outstanding visit drains from its residual and the publisher continues the obligation exactly once. No zero reading, reset or missing-reading likelihood is applied. |
| Run initialisation, subsequent updates and the action fork with only public inputs and fixed forecaster draws. Deny access to the live plant, private state, labels, tapes and evidence files. Repeat with different evaluator-only contents behind the same public input. | Both runs succeed with identical particle trajectories and forecasts. The deployed forecaster cannot read a private initial condition or refresh later state from the evaluator. |

## Joint outbox observation likelihood

Use $g(M,N,A,D\mid C)=p(M\mid C)\,p(N\mid M,C)\,p(A,D\mid M,N,C)$. Here $C$ is the particle's proposed transition with its permitted prior history and controls. In this equation $A$ denotes oldest pending age. The joint factorisation is exact by the chain rule. The fitted conditional families and the choice of conditioning summaries still require development validation. [Decision 8](../docs/adr/0010-counts-first-outbox-likelihood.md)

- Recover completed mark count $M$ from five-second throughput. Score it once. Score ending pending count $N$ conditional on $M$. A previous count delivered by the start of the interval may condition propagation as known context. The ending count can serve that role in subsequent predictions after it has been assimilated. [Transition and likelihood factorisation](../docs/adr/0004-particle-workload-and-publisher-memory.md)
- Use the joint timing law for oldest pending age $A$ and median completed delay $D$ on the components defined by the counts. For $N=0$, the conditional mass at $A=0$ is one. For $M=0$, $D$ is undefined. A completely undelivered reading has no measurement update. A delayed reading becomes eligible at its availability time and retains its original measurement interval.
- Distinguish a hypothetical visit from observed evidence. A particle's hypothetical publisher visits affect $W$ through FIFO propagation. Conditional on the proposed boundary states and permitted history, the empirical emission law marginalises over within-interval paths. Evaluate that law once at the delivered reading. Neither a deterministic match to one simulated publisher path nor a second stochastic count score belongs in the update.
- Replay development roots to align the physical regime and unfinished work with the start and end of each reading. A root's final cut-state target labels only that cut. Keep each root family in one fit or validation partition. Select families, smoothing and calibration settings on development data only. Use the particle's own proposed states for likelihood evaluation at inference. Physical labels, experimental clock features identifying the onset, root identifiers, future outcomes and protected state stay outside the forecaster.
- On root-disjoint development validation, check empty-outbox frequency, the joint law of marks and pending counts, and the relationship between age and delay given counts, separately for each regime. The retained 14,400 development readings include 3,822 empty outboxes and zero intervals without a mark. Preserve support for $M=0$ and document how much its fitted probability depends on extrapolation.

### Initial conditional distributions

| Conditional quantity | Initial family and support |
|---|---|
| Completed marks $M$ | Integrate a normal distribution over each integer's half-unit bin and normalise over physically feasible counts. Fit its centre and a positive regime-specific spread. Compute tail masses accurately enough to retain positive mass at zero when feasible. Throughput has already supplied this count. |
| Ending pending count $N$ given $M$ | If the starting boundary count $n_0$ is known, set $L=\max(0,n_0-M)$ since each mark removes one row. Otherwise use the nonnegative lower bound and account for staleness as context. Fit a logistic hurdle for $N=L$ and a zero-truncated negative binomial for the positive excess $N-L$. |
| Age and completed delay given $M,N$ | Fit an ordinal logistic model for the age band, then an ordinal logistic model for the delay band conditional on age band, counts and particle context. Apply deterministic lifecycle constraints first. For example, if a known oldest row was pending at the preceding boundary and there have been no marks, its age advances by exactly five seconds. An empty outbox fixes age at zero. Delay is undefined without marks. |

Estimate parameters with regularised conditional likelihood on interval-aligned development replays. Start with regime intercepts and shared effects of boundary work and available starting count, including its staleness. Use complete root families for training and validation. Initial candidate timing boundaries in seconds are 0, 0.025, 0.1, 0.5, 2, 5, 10 and 20, with an open upper band. Freeze the boundaries, regularisation and fitted parameters before protected access.

Within each regime, compare held-out observed and predicted mark distributions, empty frequencies, joint count and backlog tails, and delay-band frequencies conditional on age and counts. Check successive one-step residuals for persistent bias or dependence. Quantify uncertainty by resampling whole roots. A repeatable discrepancy calls for investigating the affected factor and its physical support. Keep the scope of any revision tied to that discrepancy and document it before locking the fit. [Decision 9](../docs/adr/0011-initial-outbox-likelihood-families.md)

## Posterior action forecasts

For each of $K$ pairs, sample $(Z_0,W_0)$ jointly using the decision-time particle weights. Reconstruct compatible publisher progress and pending obligations once from that draw and the available outbox history. Carry admitted work into both branches with identical residual work. An outstanding publisher visit already included in $W_0$ adds no second admission. The evaluator's true regime, queue, publisher phase, tapes and future random streams stay private. All methods receive the same permitted information and declared future workload assumptions. [Decision 10](../docs/adr/0012-matched-posterior-action-rollouts.md)

Generate one fresh, independent 120-second future per pair under its fixed regime. Match application and query arrival times, query service, request routing uniforms and business-work draws between siblings. Change only the routing threshold for new writes. Both branches evolve stochastic workload, queue and publisher progress under their own routing action.

Score eligible writes arriving strictly after the cut whose 250 ms deadlines fall at or before 120 s. Assign by deadline to four 30-second windows, open on the left and closed on the right. A commit at its deadline is on time. A write still uncommitted at its deadline is late. A branch breaches if any nonempty window has strictly more than 5% late writes. Inherited work affects waiting but adds no scored prefix writes. [Frozen endpoint](../telemetry_sufficiency/SCIENTIFIC_SPEC.md#measurements-and-evaluator-targets)

Keep counts of hold-only, promote-only, both and neither across the $K$ forecast pairs. Divide hold-only plus both by $K$ for hold risk. Divide promote-only plus both by $K$ for promote risk. Their signed difference is (promote-only minus hold-only) divided by $K$. Choose and lock $K$ using development-only convergence checks for all three outputs, repeating the forecasts with independent forecast seeds. Report Monte Carlo stability. Assess hold and promote calibration separately, including low- and high-backlog histories. Approximation errors in inherited DB work and the timing of owed publisher visits can propagate differently under the two actions. Each protected root supplies one independent evaluation unit, regardless of $K$.

## Baselines and attribution

**Primary comparison.** Compare the full PF-based forecasting pipeline with the locked temporal-history predictor on signed incremental-risk MSE. The latter uses the complete latest reading and the existing summaries of the final six scheduled intervals. Also report the locked action-only prior, latest oldest pending age and complete snapshot. These three supply context. The PF can assimilate up to twelve readings, whereas the locked temporal summaries use the final six scheduled intervals. A primary gain applies to the complete forecasting pipeline at its delivered finite rollout budget. The contribution of the particle algorithm in isolation remains unresolved. [Existing predictors](../telemetry_sufficiency/methods.py) · [Revised Decision 11](../docs/adr/0013-baselines-and-protected-evaluation.md)

**Explanatory ablation.** Retain the PF's cut regime probabilities. Given each sampled regime, substitute a draw from the training roots' empirical cut-work distribution for that regime, with equal weight per root. Keep publisher context and reconstruction, matched future primitives, routing, endpoint and forecast budget identical. Pair the regime and future random draws across variants. Reconstruct omitted obligations compatibly with each variant’s own work draw using the same conditional rule. Only the source of cut work changes. Use the ordinary primary half-life for this ablation. A gain measures what the history-specific estimate of unfinished work at the cut contributes under the shared rollout procedure.

## Protected evaluation

Evaluate once on identical fresh independent root families, balanced across A/B/C. Each ordinary root supplies one evaluator-matched pair of action outcomes. Report hold and promote Brier scores and calibration, signed increment MSE, regime and unfinished-work diagnostics, and development, filtering and rollout runtimes. Give pooled and per-condition results and paired whole-root intervals. The signed-increment contrast is the sole primary hypothesis. Promotion Brier remains the primary *absolute-risk* metric. [Fixed budget, scoring and custody plan](EVALUATION_PLAN.md)

## Implementation and independent verification plan

The companion [methods and implementation plan](METHODS_PLAN.md) defines the approximate transition, likelihood checks, bootstrap algorithm, numerical budgets, reproducibility records and ordered acceptance checks. Development convergence, calibration and runtime checks precede the protected evaluation freeze.

## Final-report limitations plan

- **Bound the startup claim.** Report that the protocol supplies a known empty physical start. Performance when monitoring attaches to an already-running service with an unknown DB queue and publisher state remains unmeasured. A joint warm-start prior is optional follow-up work.

- **Describe the effect of the starting prior.** With a finite observation history, the posterior can retain some dependence on the initial regime weights, especially when early readings are missing or ambiguous. State this limitation and retain the uniform prior independently of protected results.

- **Assess sensitivity to persistence.** Report all three predeclared half-lives on the same evaluation roots. State whether regime and unfinished-work estimates, calibration for either action, or improvement over baselines on signed incremental risk depend materially on the half-life. The 120-second setting retains primary status regardless of the protected results.

- **Test omitted queue and publisher memory.** On held-out development roots, compare predicted five-second mark counts and interval-end unfinished work with evaluator replay. Examine similar starting pending counts across publisher phases. Use evaluator-only phase and residual acknowledgement time to diagnose errors in the physical predictions and in hold and promote calibration. Report the size and repeatability of discrepancies. Require a consequential deficiency and evidence for the smallest inferable correction before adding latent dimensions.
- **Separate regime recognition from queue estimation.** On fresh protected roots, report results pooled and within each of A, B and C. Show regime inference, unfinished-work error and forecast accuracy for hold, promote and their signed increment. Compare the full forecast with one that retains the same regime probabilities and averages over queue work within regime. This isolates what the history-specific queue estimate contributes at the decision cut.
- **Describe the unsupported combination.** State that the primary design excludes simultaneous elevated demand and slow publishing, and explain the resulting risk of misattribution and miscalibration. If a separately specified combined-disturbance stress test is feasible, report its forecasts and calibration separately, including deterioration. Otherwise identify performance under that scenario as unmeasured.
